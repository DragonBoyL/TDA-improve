import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights): # 熵值的「归一化处理」函数
    max_entropy = math.log2(clip_weights.size(1)) #最大熵：当模型对所有类别完全不确定（均匀分布，\(p_i = 1/C\)），熵值达到理论最大值 \(H_{max} = \log_2(C)\)
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs): # [0.1B, C]  
    #「集合平均熵」：先把 B 个样本的概率分布取平均，再计算熵
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # 每个元素是对应类别的对数概率 log(p)  [0.1B, C]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # [C] 每个元素是平均概率的对数 
    min_real = torch.finfo(avg_logits.dtype).min # 获取当前数据类型（如 float32）的最小可表示值
    avg_logits = torch.clamp(avg_logits, min=min_real) # torch.clamp(input, min=None, max=None) 是 PyTorch 中用于「截断张量数值范围」的函数。将 avg_logits 张量的所有元素限制在「当前数据类型可表示的最小值」以上，避免因数值过小导致的计算错误
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1) # 标量，熵值


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template] # template = ["a photo of a {}."] → ["a photo of a tench."]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)    # [N, D]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # [N， D]
            class_embedding = class_embeddings.mean(dim=0) # [D]
            class_embedding /= class_embedding.norm() # [D]
            clip_weights.append(class_embedding) # c个元素，每个元素D维

        clip_weights = torch.stack(clip_weights, dim=1).cuda() # [D, C] dim = 1 相当于做了转置操作，便于后面直接相乘
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights # [B, C] 将相似度（范围 -1 ~ 1）缩放至 -100 ~ 100，适配分类得分的数值习惯 

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits) # [B]
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)] # 筛选熵值最小的10%样本（高置信度样本）
            output = clip_logits[selected_idx] # [0.1B, C]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0) # [0.1B, D] -> [1, D]
            clip_logits = output.mean(0).unsqueeze(0) # [1, C]

            loss = avg_entropy(output) 
            prob_map = output.softmax(1).mean(0).unsqueeze(0) # [1, C]
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t()) #在类别维度（dim=1）取 Top1 得分：第一个1：取前 1 个值；第二个1：类别维度；True：降序排列；True：返回值和索引
        else:
            loss = softmax_entropy(clip_logits) # [1]
            prob_map = clip_logits.softmax(1) # [1, C]
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template