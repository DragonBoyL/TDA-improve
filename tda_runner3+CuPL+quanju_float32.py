import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
    
def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='/root/dataset/TestTimeData', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

def compute_nullspace_from_hist(cov_hist, thres = 0.01):
    """
    基于「全局累计协方差」计算零空间基
    """
    thres = 0.1
    if cov_hist is None or torch.allclose(cov_hist, torch.zeros_like(cov_hist)):
            return None  # 无历史协方差时跳过投影

    # SVD
    eigvals, eigvecs = torch.linalg.eigh(cov_hist.float())  # 升序
    # 选择累计能量低于阈值的特征向量作为零空间基
    energy = eigvals / eigvals.sum()
    cum_energy = torch.cumsum(energy, dim=0)
    k = (cum_energy <= thres).sum().item()
    U2 = eigvecs[:, :k].to(cov_hist.dtype)

    # print(f"零空间向量数量/总特征维度：{U2.shape[1]}/{eigvals.shape[0]} ")

    return U2
    
# def compute_nullspace_from_hist(cov_hist, thres=10):
#     """
#     基于「全局累计协方差」计算零空间基
#     """
#     try:
#         if cov_hist is None or torch.allclose(cov_hist, torch.zeros_like(cov_hist)):
#             return None  # 无历史协方差时跳过投影
        
#         # 特征值分解（对称矩阵更高效）


#         eigvals, eigvecs = torch.linalg.eigh(cov_hist.float())  # eigvals: [D]，eigvecs: [D, D]
        
#         lambda_max = eigvals[-1]
#         mask = eigvals <= lambda_max*1e-4

#         # lambda_min = eigvals[0].clamp(min=1e-12)  # 避免极小值导致计算错误
#         # mask = eigvals <= (1e5 * lambda_min)  # 筛选零空间基
        
#         # print(mask.sum().item(), "个零空间基向量被选中。")
#         # print(lambda_min.item(), "最小特征值。")
#         # print(eigvals)
#         # print(a * lambda_min)

#         U2 = eigvecs[:, mask].to(cov_hist.dtype)
#         return U2
#     except Exception as e:
#         print(f"计算零空间失败：{str(e)}")
#         return None
    
# def compute_nullspace_from_hist(cov_hist, thres):
#     """
#     基于「全局累计协方差」计算零空间基
#     """
#     thres = 1e5
#     _, eigen_value, eigen_vector = torch.svd(cov_hist.float(), some=False) #降序
#     ind = eigen_value <= eigen_value[0] * 1e-4
#     # print(eigen_value)
#     # print('零空间向量数量/总特征维度 {}/{};radio:{}'.format(
#     # ind.sum(), eigen_value.shape[0],
#     # eigen_value[ind].sum(
#     # ) / eigen_value.sum()
#     # ))

#     U2 = eigen_vector[:, ind].to(cov_hist.dtype)
#     return U2

def project_feat_to_nullspace(feat, cov_hist, thres=0.01):
    """
    将特征投影到全局协方差的零空间
    """
    with torch.no_grad():
        U2 = compute_nullspace_from_hist(cov_hist, thres)
        if U2 is None or U2.size(1) == 0:
            return feat 
        
        transform = torch.mm(U2, U2.transpose(1, 0))
        # transform = transform / torch.norm(transform) 
        # print(transform)
        # I = torch.eye(feat.shape[1], device="cuda", dtype=torch.float32) - transform
        # print(I)
        proj_feat = torch.mm(feat.float(), transform)  # 投影到零空间
        # 投影到零空间 + 归一化
        proj_feat = proj_feat.to(feat.dtype)
        proj_feat = F.normalize(proj_feat, dim=1) 

        return proj_feat

def accumulate_global_cov(cov_hist, hist_count, new_feat):

    with torch.no_grad():
        new32 = new_feat.float()
        cov32 = cov_hist.float() 
        
        # 加权平均累计
        updated_cov = (hist_count * cov32 + new32.t() @ new32) / (hist_count + 1) 
        # updated_cov = cov32 + new32.t() @ new32

        # updated_cov = updated_cov / torch.norm(updated_cov, p='fro')  # 
        updated_count = hist_count + 1
        return updated_cov, updated_count
    
def verify_paper_condition(cov_hist, proj_update):
    # Eqn 3: Cov * delta_w = 0
    result = torch.mm(cov_hist, proj_update.t())
    
    # 由于浮点误差和近似零空间，这里看它是否比 cov_hist 本身的量级小很多
    print(f"Eqn(3) 验证值 (应该接近0): {result.abs().mean().item():.2e}")

def update_positive_cache(
    pos_cache, global_cov_hist, global_hist_count,
    image_features, class_idx, loss, pos_params, thres
):
    with torch.no_grad():
        cache_list = pos_cache.get(class_idx, [])   # 当前类的缓存列表

        # Case 1: 缓存没满 → 零空间投影
        if len(cache_list) < pos_params['shot_capacity']:
            # proj_feat = project_feat_to_nullspace(image_features, global_cov_hist, thres)
            proj_feat = image_features
            update_cache(pos_cache, class_idx, [proj_feat, loss], pos_params['shot_capacity'])
        # Case 2: 缓存已满 + 新样本熵更大 → 不更新
        elif loss >= cache_list[-1][1]:
            pass
        # Case 3: 缓存已满 + 新样本熵更小 → 零空间投影替换
        else:
            old_feat = cache_list[-1][0]  # 待替换的旧特征
            update = image_features - old_feat   # [1, D]

            # 只有在 Warm-up 结束且协方差有效时才投影
            if global_hist_count > 0:
                _update = project_feat_to_nullspace(update, global_cov_hist, thres)
            else:
                _update = update

            # # 验证
            # print(f"投影后更新量与历史协方差的内积: {torch.mm(global_cov_hist, _update.t()).abs().mean().item():.2e}")
            # print(f"投影后更新量与旧特征的内积 : {torch.mm(old_feat, _update.t()).abs().mean().item():.2e}")
            # print(f"投影前后updated的内积: {torch.mm(update, _update.t()).abs().mean().item()}")

            f_corr = old_feat + _update
            f_corr = F.normalize(f_corr, dim=1) #没有影响
            update_cache(pos_cache, class_idx, [f_corr, loss], pos_params['shot_capacity'])
            
        global_cov_hist, global_hist_count = accumulate_global_cov(global_cov_hist, global_hist_count, image_features)
        return global_cov_hist, global_hist_count

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity.
            features_loss：[image_features, loss, prob_map]：
                image_features：归一化的图像特征（[1, D]）；
                loss：熵值（越小→置信度越高）；
                prob_map：类别概率分布（[1, C]）
    """
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1)) # 根据熵值升序排序
        else:
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []  # 存储缓存中所有样本的特征 [item[0]]
        cache_values = [] # 正缓存存类别索引，负缓存存概率分布 [item[2]]
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0]) # image_feature[1, D]
                if neg_mask_thresholds:
                    cache_values.append(item[2]) # prob_map [1, C]
                else:
                    cache_values.append(class_index) # pred 

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0) # [1, D] -> [K, D] -> [D, K] permute交换两个维度
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0) # [K, C]
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().float() # [K, C] 掩码过滤：仅保留概率在 (min, max) 之间的类别
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().float() # [K, C]正缓存，生成one-hot编码  

        affinity = image_features @ cache_keys # [1, K]
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values # [1 ,C] 每一个类别的修正得分，每一列表示所有缓存样本对该类别的贡献总和（亲和度越高，贡献越大）
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        D = clip_weights.size(0)  # 特征维度（CLIP输出维度，RN50=1024，ViT-B/16=512）

        # 全局协方差
        global_cov_hist = torch.zeros((D, D), device=clip_weights.device, dtype=torch.float32)
        global_hist_count = 0

        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')): # tqdm显示测试集处理进度条
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights) # 对熵进行归一化

            if pos_enabled:
                global_cov_hist, global_hist_count = update_positive_cache(
                    pos_cache=pos_cache,
                    global_cov_hist=global_cov_hist,
                    global_hist_count=global_hist_count,
                    image_features=image_features,
                    class_idx=pred,
                    loss=loss,
                    pos_params=pos_params,
                    thres=0.01,
                )
                # update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone() # [1, K]
            if pos_enabled and pos_cache: # 第一次cache为空
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)

            if i%1000==0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model = clip_model.float()
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess) ## 测试图像的datacloader, 类别名称，模板
        clip_weights = clip_classifier(classnames, template, cupl_path, clip_model) # [D, C] 特征维度 x class类别数

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()