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

# def compute_approx_null_space(cov_matrix, a=10):
#     """
#     计算协方差矩阵的近似零空间基U2（基于SVD）
#     Args:
#         cov_matrix: D×D 协方差矩阵 (torch.cuda.HalfTensor)，D为特征维度
#         a: 筛选阈值，仅保留奇异值≤a×最小奇异值的向量（平衡稳定与可塑性）
#     Returns:
#         U2: D×K 零空间基矩阵，K为零空间维度
#     """
#     with torch.no_grad():
#         # 数值稳定性：添加极小单位矩阵避免奇异值退化
#         cov_stable = cov_matrix + 1e-6 * torch.eye(
#             cov_matrix.size(0), 
#             device=cov_matrix.device, 
#             dtype=cov_matrix.dtype
#         )
#         # SVD分解（对称矩阵满足 U=V^T）
#         cov32 = cov_stable.float()
#         U, S, Vh = torch.linalg.svd(cov32)
#         U = U.to(cov_matrix.dtype)
#         # 筛选近似零空间基（奇异值足够小的向量）
#         min_S = S.min()
#         null_mask = S <= a * min_S  # 仅保留"贡献可忽略"的奇异向量
#         U2 = U[:, null_mask]  # 零空间基：D×K
#         return U2
    
def compute_nullspace_excluding_old(cov, old_feat, k, a=10):
    """
    基于 Σ_excl = (k * Σ_old - f_old f_old^T) / (k - 1) 计算零空间基
    """
    if k <= 1:
        return None  # 无法排除 old 特征

    try:
        cov32 = cov.float()
        old32 = old_feat.float()

        Sigma_excl = (k * cov32 - old32.t() @ old32) / (k - 1)
        Sigma_excl = Sigma_excl + 1e-6 * torch.eye(Sigma_excl.size(0), device=Sigma_excl.device)

        eigvals, eigvecs = torch.linalg.eigh(Sigma_excl)
        lambda_min = eigvals[0].clamp(min=1e-12)
        mask = eigvals <= (a * lambda_min)

        if not torch.any(mask):
            print("没有找到有效的零空间基向量，跳过替换操作。")
            return None

        # # R
        # total = eigvals.sum().item()
        # selected_sum = eigvals[mask].sum().item()
        # R = selected_sum / total if total > 0 else 0.0

        # print(R)

        U2 = eigvecs[:, mask].to(cov.dtype)
        return U2

    except Exception:
        return None


# def project_to_null_space(feat, U2):
#     """
#     将特征投影到近似零空间（剔除历史特征子空间分量）
#     Args:
#         feat: [1, D] 输入特征（待更新的新特征）
#         U2: D×K 零空间基矩阵
#     Returns:
#         proj_feat: [1, D] 投影后的特征（仅保留零空间方向信息）
#     """
#     with torch.no_grad():
#         if U2.size(1) == 0:  # 无有效零空间（理论罕见），返回原始特征
#             print("没有找到零空间，跳过投影操作。")
#             return feat
#         # 投影公式：U2 @ U2^T @ feat^T → 转置回[1, D]
#         proj_feat = (feat @ U2) @ U2.T
#         return proj_feat

# def update_covariance(cov_matrix, old_feat=None, new_feat=None):
#     """
#     增量更新协方差矩阵（支持新增/替换场景）
#     Args:
#         cov_matrix: 当前协方差矩阵 (D×D)
#         old_feat: [1, D] 被替换的旧特征（替换场景传入）
#         new_feat: [1, D] 新特征（新增/替换场景均传入）
#     Returns:
#         updated_cov: 更新后的协方差矩阵
#     """
#     with torch.no_grad():
#         updated_cov = cov_matrix.clone()
#         # 替换场景：先剔除旧特征的协方差贡献
#         if old_feat is not None:
#             updated_cov -= old_feat.T @ old_feat
#         # 新增/替换场景：添加新特征的协方差贡献
#         updated_cov += new_feat.T @ new_feat
#         return updated_cov

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

def replace_with_delta_projection(pos_cache, pos_cov, class_idx, new_feat, loss, pos_params, a=10):
    """
    正确的投影替换逻辑：
        Δ = new_feat - old_feat
        Δ_proj = U2 U2^T Δ
        f_corr = old_feat + Δ_proj
    """
    cache_list = pos_cache[class_idx]
    old_feat = cache_list[-1][0]
    k = len(cache_list)
    cov = pos_cov[class_idx]

    Δ = new_feat - old_feat   # [1, D]

    # 构造排除 old_feat 的协方差 Σ_excl
    U2 = compute_nullspace_excluding_old(cov, old_feat, k, a=a)

    # 若零空间不可用 → 保守回退策略：不替换
    if U2 is None:
        print("无法计算零空间，跳过替换操作。")
        return False

    # Δ 投影到零空间
    Δ_proj = (Δ.float() @ U2.float()) @ U2.float().T
    Δ_proj = Δ_proj.to(Δ.dtype)

    # print(Δ)
    # print(Δ_proj)
    # 得到修正后的特征
    f_corr = old_feat + Δ_proj
    f_corr = F.normalize(f_corr, dim=1)

    # 用替换公式更新协方差
    old32 = old_feat.float()
    corr32 = f_corr.float()
    cov32 = cov.float()

    # Σ_new = Σ_old + (f_new f_new^T - f_old f_old^T)/old_k
    cov_new = cov32 + (corr32.t() @ corr32 - old32.t() @ old32) / k
    pos_cov[class_idx] = cov_new.to(cov.dtype)

    # 更新缓存
    update_cache(pos_cache, class_idx, [f_corr, loss], pos_params['shot_capacity'])

    return True


def update_positive_cache(pos_cache, pos_cov, image_features, pred, loss, pos_params, pos_null_a, D):
    """
    带零空间约束的正缓存更新函数
    Args:
        pos_cache: dict, 正缓存字典（key: 类别索引, value: 特征-熵值列表）
        pos_cov: dict, 类别级协方差矩阵字典（key: 类别索引, value: D×D 协方差矩阵）
        image_features: torch.Tensor [1, D], 原始图像特征（归一化后）
        pred: int, 样本的预测类别索引（scalar）
        loss: torch.Tensor, 样本的熵值（越小置信度越高）
        pos_params: dict, 正缓存参数（包含shot_capacity）
        pos_null_a: int/float, 零空间奇异值筛选阈值
        D: int, 特征维度（CLIP输出维度）
    """
    with torch.no_grad():
        new_feat = image_features.clone()  # 原始新特征 [1, D]
        class_idx = pred  # 当前样本的预测类别索引
        cache_list = pos_cache.get(class_idx, [])   # 当前类的缓存列表

        # 初始化协方差矩阵
        if class_idx not in pos_cov:
            pos_cov[class_idx] = torch.zeros((D, D), 
                                             device=new_feat.device, 
                                             dtype=new_feat.dtype)

        cov = pos_cov[class_idx]   # 当前类协方差矩阵

        # ----------------------------------------------------------------------
        # Case 1: 缓存没满 → 不做零空间投影，直接加入原始 new_feat
        # ----------------------------------------------------------------------
        if len(cache_list) < pos_params['shot_capacity']:
            # 更新协方差: Σ += f_new f_new^T # Σ_new = (old_k*Σ_old + f_new f_new^T) / new_k
            pos_cov[class_idx] = ( len(cache_list) * cov + new_feat.T @ new_feat )/ (len(cache_list) + 1)
        
            # 加入缓存（存原始特征）
            update_cache(pos_cache, class_idx, [new_feat, loss], pos_params['shot_capacity'])
            
            return 
        # ----------------------------------------------------------------------
        # Case 2: 缓存已满 + 新样本熵更大 → 忽略，不更新
        # ----------------------------------------------------------------------
        last_loss = cache_list[-1][1]
        if loss >= last_loss:
            return  # 不更新缓存 & 不更新协方差（确保二者同步）
        # ----------------------------------------------------------------------
        # Case 3: 缓存已满 + 新样本熵更小 →  执行 Δ 投影替换
        # ----------------------------------------------------------------------
        # U2 = compute_approx_null_space(cov, a=pos_null_a)  # 近似零空间基
        # new_feat_proj = project_to_null_space(new_feat, U2)  # 投影后的特征
        # # 更新协方差：Σ = Σ - f_old f_old^T + f_proj f_proj^T
        # old_feat = cache_list[-1][0]    # [1, D]
        # pos_cov[class_idx] = cov - old_feat.T @ old_feat + new_feat_proj.T @ new_feat_proj

        # update_cache(pos_cache, class_idx, [new_feat_proj, loss], pos_params['shot_capacity'])
        # Case 3: 缓存已满 + 新样本熵更小 → 正确的 Δ 投影替换
        success = replace_with_delta_projection(
            pos_cache, pos_cov, class_idx,
            new_feat, loss,
            pos_params, a=pos_null_a
        )
    
        # 若零空间不可用 → 不替换（保持稳定）
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
        
        pos_cov = {}  # key: 类别索引, value: D×D 协方差矩阵 (torch.cuda.HalfTensor)
        D = clip_weights.size(0)  # 特征维度（CLIP输出维度，RN50=1024，ViT-B/16=512）

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
                    pos_cov=pos_cov,
                    image_features=image_features,
                    pred=pred,
                    loss=loss,
                    pos_params=pos_params,
                    pos_null_a=10,
                    D=D
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
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess) ## 测试图像的datacloader, 类别名称，模板
        clip_weights = clip_classifier(classnames, template, clip_model) # [D, C] 特征维度 x class类别数
        
        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()