# from einops import rearrange, repeat
# from einops import rearrange, repeat
# import torch
# import sys
# sys.path.append("../local_distance_indicator")
# from models.pointops.functions import pointops
# from args.pu1k_args import parse_pu1k_args
# from args.pugan_args import parse_pugan_args
# from models.P2PNet_Attention import P2PNet
# try:
#     from args.pu1k_args import parse_pu1k_args_multiscale
#     from args.pugan_args import parse_pugan_args_multiscale
#     from models.P2PNet_MultiScale import P2PNet_MultiScale
#     MULTISCALE_AVAILABLE = True
# except ImportError:
#     MULTISCALE_AVAILABLE = False
#     print("Warning: Multi-scale modules not found, using original model only")
# import numpy as np
# from sklearn.neighbors import NearestNeighbors

# def detect_model_type_from_checkpoint(ckpt_path):
#     """
#     从checkpoint检测模型类型
    
#     为什么需要这个函数：
#     - 自动识别checkpoint是单尺度还是多尺度
#     - 避免用户手动指定错误导致加载失败
    
#     Args:
#         ckpt_path: checkpoint路径
    
#     Returns:
#         str: 'multiscale', 'original', or 'unknown'
#     """
#     try:
#         checkpoint = torch.load(ckpt_path, map_location='cpu')
        
#         # 处理不同格式的checkpoint
#         if isinstance(checkpoint, dict):
#             # 有些checkpoint保存为 {'model': state_dict, 'optimizer': ...}
#             # 有些直接保存 state_dict
#             state_dict = checkpoint.get('model', checkpoint)
#         else:
#             state_dict = checkpoint
        
#         keys = list(state_dict.keys())
        
#         # 多尺度模型的特征键
#         multiscale_indicators = [
#             'feature_extractor.scales',           # 尺度列表
#             'feature_extractor.scale_extractors',  # 多尺度提取器
#             'feature_extractor.fusion',            # 融合模块
#             'scale_extractors.0',                  # 第一个尺度提取器
#             'fusion.fusion_layers'                 # 融合层
#         ]
        
#         # 检查是否包含多尺度特征
#         for indicator in multiscale_indicators:
#             if any(indicator in key for key in keys):
#                 return 'multiscale'
        
#         # 如果没有多尺度特征，则为原始模型
#         return 'original'
        
#     except Exception as e:
#         print(f"Warning: Could not detect model type from checkpoint: {e}")
#         return 'unknown'


# def load_checkpoint_safe(model, ckpt_path, device='cuda', strict=False):
#     """
#     安全地加载checkpoint，处理键不匹配问题
    
#     为什么需要这个函数：
#     1. 多尺度模型和单尺度模型的state_dict键不完全相同
#     2. 多尺度模型有额外的参数（scale_extractors, fusion）
#     3. 直接用 load_state_dict(strict=True) 会报错
    
#     解决方案：
#     - 使用 strict=False 模式
#     - 只加载匹配的键
#     - 详细报告加载情况
    
#     Args:
#         model: 目标模型
#         ckpt_path: checkpoint路径
#         device: 设备
#         strict: 是否严格匹配所有键
    
#     Returns:
#         bool: 是否成功加载
#     """
#     try:
#         print(f"\n{'='*70}")
#         print(f"Loading checkpoint: {ckpt_path}")
#         print(f"{'='*70}")
        
#         # 加载checkpoint
#         checkpoint = torch.load(ckpt_path, map_location=device)
        
#         # 获取state_dict
#         if isinstance(checkpoint, dict) and 'model' in checkpoint:
#             ckpt_state = checkpoint['model']
#         else:
#             ckpt_state = checkpoint
        
#         # 获取模型当前的state_dict
#         model_state = model.state_dict()
        
#         # 分析键匹配情况
#         model_keys = set(model_state.keys())
#         ckpt_keys = set(ckpt_state.keys())
        
#         matched_keys = model_keys & ckpt_keys  # 交集：匹配的键
#         missing_keys = model_keys - ckpt_keys  # 模型有，checkpoint没有
#         unexpected_keys = ckpt_keys - model_keys  # checkpoint有，模型没有
        
#         # 打印统计信息
#         print(f"\nCheckpoint Loading Statistics:")
#         print(f"  Model parameters: {len(model_keys)}")
#         print(f"  Checkpoint parameters: {len(ckpt_keys)}")
#         print(f"  Matched: {len(matched_keys)}")
#         print(f"  Missing (will use random init): {len(missing_keys)}")
#         print(f"  Unexpected (will ignore): {len(unexpected_keys)}")
        
#         # 显示部分缺失和多余的键（方便调试）
#         if missing_keys and len(missing_keys) <= 10:
#             print(f"\n  Missing keys:")
#             for key in sorted(missing_keys):
#                 print(f"    - {key}")
#         elif missing_keys:
#             print(f"\n  Missing keys (first 5):")
#             for key in sorted(missing_keys)[:5]:
#                 print(f"    - {key}")
#             print(f"    ... and {len(missing_keys)-5} more")
        
#         if unexpected_keys and len(unexpected_keys) <= 10:
#             print(f"\n  Unexpected keys:")
#             for key in sorted(unexpected_keys):
#                 print(f"    - {key}")
#         elif unexpected_keys:
#             print(f"\n  Unexpected keys (first 5):")
#             for key in sorted(unexpected_keys)[:5]:
#                 print(f"    - {key}")
#             print(f"    ... and {len(unexpected_keys)-5} more")
        
#         # 加载checkpoint
#         if strict:
#             # 严格模式：所有键必须完全匹配
#             model.load_state_dict(ckpt_state, strict=True)
#             print(f"\n✓ Loaded successfully (strict mode)")
#         else:
#             # 非严格模式：只加载匹配的键
#             # 这是关键！允许部分键不匹配
#             filtered_state = {k: v for k, v in ckpt_state.items() if k in model_keys}
#             model.load_state_dict(filtered_state, strict=False)
            
#             if missing_keys:
#                 print(f"\n⚠ Warning: {len(missing_keys)} parameters not loaded")
#                 print(f"  These will use random initialization")
            
#             print(f"\n✓ Loaded successfully (non-strict mode)")
#             print(f"  {len(filtered_state)}/{len(model_keys)} parameters loaded from checkpoint")
        
#         print(f"{'='*70}\n")
#         return True
        
#     except Exception as e:
#         print(f"\n✗ Error loading checkpoint:")
#         print(f"  {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False


# def get_local_model(dataset, use_multiscale=True):
#     """
#     Get local model (either original or multi-scale version)
    
#     Args:
#         dataset: 'pu1k' or 'pugan'
#         use_multiscale: whether to use multi-scale model
    
#     Returns:
#         model: P2PNet or P2PNet_MultiScale
#     """
#     if use_multiscale and not MULTISCALE_AVAILABLE:
#         print("Warning: Multi-scale requested but not available, using original model")
#         use_multiscale = False
    
#     # 获取模型参数
#     if dataset == 'pu1k':
#         if use_multiscale:
#             model_args = parse_pu1k_args_multiscale()
#         else:
#             model_args = parse_pu1k_args()
#     else:  # pugan
#         if use_multiscale:
#             model_args = parse_pugan_args_multiscale()
#         else:
#             model_args = parse_pugan_args()
    
#     # 创建模型
#     if use_multiscale:
#         print(f"Creating multi-scale model for {dataset}")
#         if hasattr(model_args, 'scales'):
#             print(f"  Scales: {model_args.scales}")
#         if hasattr(model_args, 'fusion_type'):
#             print(f"  Fusion: {model_args.fusion_type}")
#         return P2PNet_MultiScale(model_args)
#     else:
#         print(f"Creating original model for {dataset}")
#         return P2PNet(model_args)


    
# def index_points(pts, idx):
#     """
#     Input:
#         pts: input points data, [B, C, N]
#         idx: sample index data, [B, S, [K]]
#     Return:
#         new_points:, indexed points data, [B, C, S, [K]]
#     """
#     batch_size = idx.shape[0]
#     sample_num = idx.shape[1]
#     fdim = pts.shape[1]
#     reshape = False
#     if len(idx.shape) == 3:
#         reshape = True
#         idx = idx.reshape(batch_size, -1)
#     # (b, c, (s k))
#     res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
#     if reshape:
#         res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

#     return res


# def FPS(pts, fps_pts_num):
#     # input: (b, 3, n)

#     # (b, n, 3)
#     pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
#     # (b, fps_pts_num)
#     sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
#     # (b, 3, fps_pts_num)
#     sample_pts = index_points(pts, sample_idx)

#     return sample_pts

# # generate patch for test
# def extract_knn_patch(k, pts, center_pts):
#     # input : (b, 3, n)

#     # (n, 3)
#     pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
#     pts_np = pts_trans.detach().cpu().numpy()
#     # (m, 3)
#     center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
#     center_pts_np = center_pts_trans.detach().cpu().numpy()
#     knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
#     knn_search.fit(pts_np)
#     # (m, k)
#     knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
#     # (m, k, 3)
#     patches = np.take(pts_np, knn_idx, axis=0)
#     patches = torch.from_numpy(patches).float().cuda()
#     # (m, 3, k)
#     patches = rearrange(patches, 'm k c -> m c k').contiguous()

#     return patches

# def get_knn_pts(k, pts, center_pts, return_idx=False):
#     # input: (b, 3, n)

#     # (b, n, 3)
#     pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
#     # (b, m, 3)
#     center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
#     # (b, m, k)
#     knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
#     # (b, 3, m, k)
#     knn_pts = index_points(pts, knn_idx)

#     if return_idx == False:
#         return knn_pts
#     else:
#         return knn_pts, knn_idx


# def midpoint_interpolated_up_rate(up_rate, sparse_pts):
#     # sparse_pts: (b, 3, n)

#     pts_num = sparse_pts.shape[-1]
#     up_pts_num = int(pts_num * up_rate)
#     k = int(2 * up_rate)
#     # (b, 3, n, k)
#     knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
#     # (b, 3, n, k)
#     repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
#     # (b, 3, n, k)
#     mid_pts = (knn_pts + repeat_pts) / 2.0
#     # (b, 3, (n k))
#     mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
#     # note that interpolated_pts already contain sparse_pts
#     interpolated_pts = mid_pts
#     # fps: (b, 3, up_pts_num)
#     interpolated_pts = FPS(interpolated_pts, up_pts_num)

#     return interpolated_pts

# def normalize_point_cloud(input, centroid=None, furthest_distance=None):
#     # input: (b, 3, n) tensor

#     if centroid is None:
#         # (b, 3, 1)
#         centroid = torch.mean(input, dim=-1, keepdim=True)
#     # (b, 3, n)
#     input = input - centroid
#     if furthest_distance is None:
#         # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
#         furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
#     input = input / furthest_distance

#     return input, centroid, furthest_distance


#对LDI进行调用
from einops import rearrange, repeat
import torch
import sys
sys.path.append("../local_distance_indicator")
from models.pointops.functions import pointops
from args.pu1k_args import parse_pu1k_args
from args.pugan_args import parse_pugan_args
from models.P2PNet_Attention import P2PNet
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_local_model(dataset, use_normal_estimation=True):  # ✨ 修改
    """
    创建局部距离指示器模型
    Args:
        dataset: 'pu1k' or 'pugan'
        use_normal_estimation: 是否使用法线估计模块 (默认True)
    Returns:
        P2PNet model
    """
    if dataset == 'pu1k':
        model_args = parse_pu1k_args()
    else:
        model_args = parse_pugan_args()
    
    # ✨ 设置法线估计标志
    model_args.use_normal_estimation = use_normal_estimation
    
    print(f"Creating local model for {dataset}")
    print(f"  - Use normal estimation: {use_normal_estimation}")
    
    return P2PNet(model_args)
        
def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts

# generate patch for test
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)

    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolated_up_rate(up_rate, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance