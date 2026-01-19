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
    
def compute_nullspace_from_hist(cov_hist, a=4):
    """
    基于「历史累计协方差」计算零空间基（核心：避开所有历史核心特征方向）
    Args:
        cov_hist: D×D 历史累计协方差（包含所有历史有效特征）
        a: 零空间筛选阈值（越小约束越严格）
    Returns:
        U2: D×K 零空间基矩阵
    """
    try:
        cov32 = cov_hist.float()
        # print(cov32)
        # 数值稳定性：添加极小单位矩阵避免奇异值退化
        cov_stable = cov32 + 1e-6 * torch.eye(cov32.size(0), device=cov32.device)
        # 特征值分解（对称矩阵更高效）
        eigvals, eigvecs = torch.linalg.eigh(cov_stable)  # eigvals: [D]，eigvecs: [D, D]
        
        # lambda_max = eigvals[-1].clamp(min=1e-12)
        # mask = eigvals <= lambda_max * a 
        lambda_min = eigvals[0].clamp(min=1e-12)  # 避免极小值导致计算错误
        mask = eigvals <= (a * lambda_min)  # 筛选零空间基
        
        # print(mask.sum().item(), "个零空间基向量被选中。")
        # print(lambda_min.item(), "最小特征值。")
        # print(eigvals)
        # print(a * lambda_min)
        # print(mask)
        
        # # 分16块打印（1024/64=16），每块64个值
        # for i in range(0, len(eigvals), 64):
        #     end_idx = min(i+64, len(eigvals))
        #     print(f"第{i}-{end_idx-1}个特征值：{eigvals[i:end_idx].cpu().numpy()}")

        U2 = eigvecs[:, mask].to(cov_hist.dtype)
        return U2
    except Exception as e:
        print(f"计算零空间失败：{str(e)}")
        return None

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

def replace_with_delta_projection(
    pos_cache, pos_cov_hist, pos_hist_count, 
    class_idx, new_feat, loss, pos_params, a=4, entropy_threshold=1
):
    """
    带历史累计协方差的零空间投影替换逻辑：
        累计高置信特征到历史协方差（永不遗忘）
        投影Δ到历史协方差的零空间（不干扰历史）
    """
    cache_list = pos_cache[class_idx]
    old_feat = cache_list[-1][0]  # 待替换的旧特征
    cov_hist = pos_cov_hist[class_idx]  # 历史累计协方差
    n_hist = pos_hist_count[class_idx]  # 历史累计样本数

    # 1. 累计特征到历史协方差
    new32 = new_feat.float()
    cov_hist32 = cov_hist.float()
    pos_cov_hist[class_idx] = (n_hist * cov_hist32 + new32.t() @ new32) / (n_hist + 1)
    # pos_cov_hist[class_idx] += new32.t() @ new32
    pos_hist_count[class_idx] += 1  # 累计数+1

    # 2. 计算原始特征更新量Δ
    Δ = new_feat - old_feat   # [1, D]

    # 3. 基于历史累计协方差计算零空间基（核心：全局不遗忘）
    U2 = compute_nullspace_from_hist(cov_hist, a=a)

    # 4. Δ投影到零空间 + 步长限制（控制更新幅度）
    Δ_proj = (Δ.float() @ U2.float()) @ U2.float().T
    Δ_proj = Δ_proj.to(Δ.dtype)
    
    # 5. 得到修正后的特征（保留历史核心）
    f_corr = old_feat + Δ_proj
    f_corr = F.normalize(f_corr, dim=1)

    # # 直接对新特征进行零空间投影修正
    # f_corr = (new_feat.float() @ U2.float()) @ U2.float().T
    # f_corr = f_corr.to(new_feat.dtype)
    # f_corr = F.normalize(f_corr, dim=1)
    
    # 6. 更新缓存并验证效果
    update_cache(pos_cache, class_idx, [f_corr, loss], pos_params['shot_capacity'])
    # verify_null_space_effect(pos_cache, class_idx, Δ_proj, old_feat, f_corr)

    return True

def verify_null_space_effect(pos_cache, class_idx, delta_f_proj, old_feat, f_corr):
    """验证零空间约束：Δf_proj与历史缓存特征正交（不干扰预测）"""
    with torch.no_grad():
        # 1. 提取历史缓存特征（不含被替换的old_feat）
        cache_list = pos_cache[class_idx]
        if len(cache_list) <= 1:
            return  # 无足够历史特征，跳过验证
        
        # 核心修复：统一转为float32，避免Half/float不匹配
        hist_feats = torch.cat([item[0].float() for item in cache_list[:-1]], dim=0)  # [k-1, D] float32
        delta_f_proj_32 = delta_f_proj.float()  # 确保更新量也是float32
        
        # 2. 验证：历史特征 · Δf_proj ≈ 0（正交，无干扰）
        if hist_feats.size(0) > 0:
            dot_product = (hist_feats @ delta_f_proj_32.T).abs().mean().item()
            print(f"零空间约束验证：历史特征与Δf_proj的平均点积={dot_product:.6f}（越接近0越好）")
        
        # 3. 验证：f_final与old_feat高相似（保留历史核心）
        old_feat = old_feat  # 被替换的旧特征（转float32）
        f_final = f_corr  # 最新更新的特征（转float32）
        similarity = F.cosine_similarity(old_feat, f_final, dim=-1).item()
        print(f"历史特征保留验证：f_old与f_final相似度={similarity:.4f}（>0.9表示保留核心）")

def update_positive_cache(
    pos_cache, pos_cov_hist, pos_hist_count,
    image_features, pred, loss, pos_params, pos_null_a, D, entropy_threshold=1
):
    """
    带历史累计协方差的正缓存更新函数：
        - 缓存未满：直接新增 + 累计历史协方差
        - 缓存已满：零空间投影替换 + 更新协方差
    """
    with torch.no_grad():
        new_feat = image_features.clone()  # 原始新特征 [1, D]
        class_idx = pred  # 当前样本的预测类别索引
        cache_list = pos_cache.get(class_idx, [])   # 当前类的缓存列表

        # 初始化协方差（当前缓存+历史累计）
        if class_idx not in pos_cov_hist:
            pos_cov_hist[class_idx] = torch.zeros((D, D), device=new_feat.device, dtype=new_feat.dtype)
            pos_hist_count[class_idx] = 0  # 历史累计数初始为0

        # Case 1: 缓存没满 → 直接加入 + 累计历史协方差
        if len(cache_list) < pos_params['shot_capacity']:
            new32 = new_feat.float()
            cov_hist32 = pos_cov_hist[class_idx].float()
            n_hist = pos_hist_count[class_idx]
            pos_cov_hist[class_idx] = (n_hist * cov_hist32 + new32.t() @ new32) / (n_hist + 1)
            # pos_cov_hist[class_idx] += new32.t() @ new32
            pos_hist_count[class_idx] += 1
            # 加入缓存
            update_cache(pos_cache, class_idx, [new_feat, loss], pos_params['shot_capacity'])
            return 
        
        # Case 2: 缓存已满 + 新样本熵更大 → 忽略，不更新
        last_loss = cache_list[-1][1]
        if loss >= last_loss:
            return  
        
        # Case 3: 缓存已满 + 新样本熵更小 → 零空间投影替换
        success = replace_with_delta_projection(
            pos_cache, pos_cov_hist, pos_hist_count,
            class_idx, new_feat, loss, pos_params,
            a=pos_null_a, entropy_threshold=entropy_threshold
        )
    
        if not success:
            print("跳过替换操作。")
            return
        
        return 

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
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half() # [K, C] 掩码过滤：仅保留概率在 (min, max) 之间的类别
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half() # [K, C]正缓存，生成one-hot编码  

        affinity = image_features @ cache_keys # [1, K]
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values # [1 ,C] 每一个类别的修正得分，每一列表示所有缓存样本对该类别的贡献总和（亲和度越高，贡献越大）
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        D = clip_weights.size(0)  # 特征维度（CLIP输出维度，RN50=1024，ViT-B/16=512）

        pos_cov_hist = {}  # 历史累计协方差（全局不遗忘）
        pos_hist_count = {}  # 历史累计样本数

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
                update_positive_cache(
                    pos_cache=pos_cache,
                    pos_cov_hist=pos_cov_hist,
                    pos_hist_count=pos_hist_count,
                    image_features=image_features,
                    pred=pred,
                    loss=loss,
                    pos_params=pos_params,
                    pos_null_a=5,
                    D=D,
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