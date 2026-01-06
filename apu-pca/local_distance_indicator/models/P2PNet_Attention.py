# import torch
# import torch.nn as nn
# from einops import repeat
# from models.FeatureExtractor import FeatureExtractor
# from models.P2PRegressor import P2PRegressor
# from models.AttentionLayer import WeightLayer
# from models.utils import get_knn_pts, index_points


# class P2PNet(nn.Module):
#     def __init__(self, args):
#         super(P2PNet, self).__init__()

#         self.args = args
#         self.feature_extractor = FeatureExtractor(args)
#         self.p2p_regressor = P2PRegressor(args)
#         self.weight = nn.ModuleList([])
#         self.global_prior = False
#         for i in range(args.block_num+1):
#             self.weight.append(WeightLayer(args))

#     def set_global_mode(self):
#         self.global_prior = True

#     def extract_feature(self, original_pts):
#         # input: (b, 3, n)

#         # global_feats: (b, c), local_feats: list (b, c, n)
#         global_feats, local_feats = self.feature_extractor(original_pts)
#         return global_feats, local_feats


#     def interpolate_feature(self, original_pts, query_pts, local_feat):
#         k = 3
#         # interpolation: (b, 3, n, k), (b, n, k)
#         knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
#         # dist
#         repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=k)
#         # (b, n, k)
#         dist = torch.norm(knn_pts - repeat_query_pts, p=2, dim=1)
#         dist_recip = 1.0 / (dist + 1e-8)
#         norm = torch.sum(dist_recip, dim=2, keepdim=True)
#         # (b, n, k)
#         weight = dist_recip / norm
#         # (b, c, n, k)
#         knn_feat = index_points(local_feat, knn_idx)
#         # (b, c, n, k)
#         interpolated_feat = knn_feat * weight.unsqueeze(1)
#         # (b, c, n)
#         interpolated_feat = torch.sum(interpolated_feat, dim=-1)
#         return interpolated_feat


#     def regress_distance(self, original_pts, query_pts, global_feats, local_feats, return_feat=False):
#         # pts: (b, 3, n) global_feats: (b, c), local_feats: list (b, c, n)

#         # (b, c, n)
#         global_feats = repeat(global_feats, 'b c -> b c n', n=query_pts.shape[-1])
#         # interpolated local feats
#         interpolated_local_feats = []
#         for i in range(self.args.block_num+1):
#             interpolated_feat = self.weight[i](original_pts, query_pts, local_feats[i])
#             interpolated_local_feats.append(interpolated_feat)
#         # (b, c*(block_num+1), n)
#         agg_local_feats = torch.cat(interpolated_local_feats, dim=1)
#         # (b, 3+c*(block_num+2), m)
#         agg_feats = torch.cat((query_pts, agg_local_feats, global_feats), dim=1)
#         # (b, 1, n)
#         p2p = self.p2p_regressor(agg_feats)
#         if not return_feat:
#             return p2p
#         else:
#             return p2p, torch.cat((agg_local_feats, global_feats), dim=1)


#     def forward(self, original_pts, query_pts, return_feat=False):
#         # input: (b, 3, n)

#         # global_feats: (b, c), local_feats: list (b, c, n)
#         if self.global_prior:
#             with torch.no_grad():
#                 global_feats, local_feats = self.extract_feature(original_pts)
#         else:
#             global_feats, local_feats = self.extract_feature(original_pts)
#         # (b, 1, n)
#         return self.regress_distance(original_pts, query_pts, global_feats, local_feats, return_feat)
        
    
import torch
import torch.nn as nn
from einops import repeat, rearrange
from models.FeatureExtractor import FeatureExtractor
from models.P2PRegressor import P2PRegressor
from models.AttentionLayer import WeightLayer
from models.utils import get_knn_pts, index_points
from models.NormalEstimator import SharedPFFEstimator  # 新增
from models.GeometricNormals import PCANormalEstimator


# class P2PNet(nn.Module):
#     def __init__(self, args):
#         super(P2PNet, self).__init__()

#         self.args = args
#         self.feature_extractor = FeatureExtractor(args)
#         self.p2p_regressor = P2PRegressor(args)
#         self.weight = nn.ModuleList([])
#         self.global_prior = False
        
#         for i in range(args.block_num+1):
#             self.weight.append(WeightLayer(args))
        
#         # ✨ 新增: 法线估计器
#         self.use_normal = args.use_normal_estimation  # 从 args 中读取
#         if self.use_normal:
#             self.normal_estimator = SharedPFFEstimator(args)

#     def set_global_mode(self):
#         self.global_prior = True

#     def extract_feature(self, original_pts):
#         """提取全局和局部特征"""
#         global_feats, local_feats = self.feature_extractor(original_pts)
#         return global_feats, local_feats

#     def estimate_normals(self, original_pts, query_pts, local_feats):
#         """
#         ✨ 新增方法: 估计 query_pts 的法线
#         Args:
#             original_pts: (B, 3, N) - 原始点云
#             query_pts: (B, 3, M) - 查询点
#             local_feats: list of (B, C, N) - 局部特征
#         Returns:
#             normals: (B, 3, M) - 查询点的法线
#         """
#         B, _, M = query_pts.shape
#         k = 48  # 最大邻域大小 (与 self.scales 一致)
        
#         # 1. 为每个查询点提取 KNN patch
#         knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
#         # knn_pts: (B, 3, M, K)
#         # knn_idx: (B, M, K)
        
#         # 2. 获取对应的特征 (使用最后一层局部特征)
#         last_feat = local_feats[-1]  # (B, C, N)
#         knn_feats = index_points(last_feat, knn_idx)  # (B, C, M, K)
        
#         # 3. 批量处理: reshape 成 (B*M, ...)
#         knn_pts_flat = rearrange(knn_pts, 'b c m k -> (b m) c k')
#         knn_feats_flat = rearrange(knn_feats, 'b c m k -> (b m) c k')
#         query_pts_flat = rearrange(query_pts, 'b c m -> (b m) c')
        
#         # 4. 调用法线估计器
#         normals_flat = self.normal_estimator(
#             knn_pts_flat,      # (B*M, 3, K)
#             query_pts_flat,    # (B*M, 3)
#             knn_feats_flat     # (B*M, C, K)
#         )  # (B*M, 3)
        
#         # 5. Reshape 回来
#         normals = rearrange(normals_flat, '(b m) c -> b c m', b=B, m=M)
        
#         return normals

#     def regress_distance(self, original_pts, query_pts, global_feats, 
#                         local_feats, query_normals=None, return_feat=False):
#         """
#         回归距离 (可选: 融合法线信息)
#         Args:
#             original_pts: (B, 3, N)
#             query_pts: (B, 3, M)
#             global_feats: (B, C)
#             local_feats: list of (B, C, N)
#             query_normals: (B, 3, M) - 可选的法线信息
#             return_feat: bool
#         Returns:
#             p2p: (B, 1, M) - 距离预测
#             (optional) features
#         """
#         # 全局特征扩展
#         global_feats = repeat(global_feats, 'b c -> b c m', m=query_pts.shape[-1])
        
#         # 插值局部特征
#         interpolated_local_feats = []
#         for i in range(self.args.block_num+1):
#             interpolated_feat = self.weight[i](original_pts, query_pts, local_feats[i])
#             interpolated_local_feats.append(interpolated_feat)
        
#         # 聚合局部特征
#         agg_local_feats = torch.cat(interpolated_local_feats, dim=1)
        
#         # ✨ 如果有法线信息，拼接进去
#         if query_normals is not None:
#             agg_feats = torch.cat([query_pts, query_normals, agg_local_feats, global_feats], dim=1)
#         else:
#             agg_feats = torch.cat([query_pts, agg_local_feats, global_feats], dim=1)
        
#         # 回归
#         p2p = self.p2p_regressor(agg_feats)
        
#         if not return_feat:
#             return p2p
#         else:
#             return p2p, torch.cat([agg_local_feats, global_feats], dim=1)

#     def forward(self, original_pts, query_pts, return_feat=False, return_normals=False):
#         """
#         ✨ 修改后的前向传播
#         Args:
#             original_pts: (B, 3, N)
#             query_pts: (B, 3, M)
#             return_feat: bool
#             return_normals: bool - 是否返回法线
#         Returns:
#             p2p: (B, 1, M)
#             (optional) normals: (B, 3, M)
#         """
#         # 1. 提取特征
#         if self.global_prior:
#             with torch.no_grad():
#                 global_feats, local_feats = self.extract_feature(original_pts)
#         else:
#             global_feats, local_feats = self.extract_feature(original_pts)
        
#         # 2. ✨ 估计法线 (如果启用)
#         query_normals = None
#         if self.use_normal:
#             query_normals = self.estimate_normals(original_pts, query_pts, local_feats)
        
#         # 3. 回归距离
#         if return_feat:
#             p2p, feats = self.regress_distance(
#                 original_pts, query_pts, global_feats, local_feats, 
#                 query_normals, return_feat=True
#             )
#             if return_normals and query_normals is not None:
#                 return p2p, feats, query_normals
#             return p2p, feats
#         else:
#             p2p = self.regress_distance(
#                 original_pts, query_pts, global_feats, local_feats, query_normals
#             )
#             if return_normals and query_normals is not None:
#                 return p2p, query_normals
#             return p2p

#使用PCA进行法线估计运算
class P2PNet(nn.Module):
    def __init__(self, args):
        super(P2PNet, self).__init__()

        self.args = args
        self.feature_extractor = FeatureExtractor(args)
        self.p2p_regressor = P2PRegressor(args)
        self.weight = nn.ModuleList([])
        self.global_prior = False
        
        for i in range(args.block_num+1):
            self.weight.append(WeightLayer(args))
        
        # ✨ 法线估计器选择
        self.use_normal = args.use_normal_estimation
        if self.use_normal:
            # 选择法线估计方法
            normal_method = getattr(args, 'normal_estimation_method', 'pca')  # 默认PCA
            
            if normal_method == 'pca':
                print("Using PCA-based normal estimation")
                k_normal = getattr(args, 'k_normal_estimation', 20)  # PCA邻域大小
                self.normal_estimator = PCANormalEstimator(k_neighbors=k_normal)
            elif normal_method == 'learned':
                print("Using learned normal estimation")
                self.normal_estimator = SharedPFFEstimator(args)
            else:
                raise ValueError(f"Unknown normal estimation method: {normal_method}")

    def set_global_mode(self):
        self.global_prior = True

    def extract_feature(self, original_pts):
        global_feats, local_feats = self.feature_extractor(original_pts)
        return global_feats, local_feats

    def estimate_normals(self, original_pts, query_pts, local_feats=None):
        """
        估计查询点的法线
        Args:
            original_pts: (B, 3, N)
            query_pts: (B, 3, M)
            local_feats: list of (B, C, N) - 仅learned方法需要
        Returns:
            normals: (B, 3, M)
        """
        if isinstance(self.normal_estimator, PCANormalEstimator):
            # ✨ PCA方法：直接基于几何
            normals = self.normal_estimator.estimate_query_normals(original_pts, query_pts)
        else:
            # 学习方法：需要特征
            B, _, M = query_pts.shape
            k = 48
            
            knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
            last_feat = local_feats[-1]
            knn_feats = index_points(last_feat, knn_idx)
            
            knn_pts_flat = rearrange(knn_pts, 'b c m k -> (b m) c k')
            knn_feats_flat = rearrange(knn_feats, 'b c m k -> (b m) c k')
            query_pts_flat = rearrange(query_pts, 'b c m -> (b m) c')
            
            normals_flat = self.normal_estimator(knn_pts_flat, query_pts_flat, knn_feats_flat)
            normals = rearrange(normals_flat, '(b m) c -> b c m', b=B, m=M)
        
        return normals

    def regress_distance(self, original_pts, query_pts, global_feats, 
                        local_feats, query_normals=None, return_feat=False):
        # 全局特征扩展
        global_feats = repeat(global_feats, 'b c -> b c m', m=query_pts.shape[-1])
        
        # 插值局部特征
        interpolated_local_feats = []
        for i in range(self.args.block_num+1):
            interpolated_feat = self.weight[i](original_pts, query_pts, local_feats[i])
            interpolated_local_feats.append(interpolated_feat)
        
        agg_local_feats = torch.cat(interpolated_local_feats, dim=1)
        
        # 拼接法线
        if query_normals is not None:
            agg_feats = torch.cat([query_pts, query_normals, agg_local_feats, global_feats], dim=1)
        else:
            agg_feats = torch.cat([query_pts, agg_local_feats, global_feats], dim=1)
        
        p2p = self.p2p_regressor(agg_feats)
        
        if not return_feat:
            return p2p
        else:
            return p2p, torch.cat([agg_local_feats, global_feats], dim=1)

    def forward(self, original_pts, query_pts, return_feat=False, return_normals=False):
        # 提取特征
        if self.global_prior:
            with torch.no_grad():
                global_feats, local_feats = self.extract_feature(original_pts)
        else:
            global_feats, local_feats = self.extract_feature(original_pts)
        
        # 估计法线
        query_normals = None
        if self.use_normal:
            query_normals = self.estimate_normals(original_pts, query_pts, local_feats)
        
        # 回归距离
        if return_feat:
            p2p, feats = self.regress_distance(
                original_pts, query_pts, global_feats, local_feats, 
                query_normals, return_feat=True
            )
            if return_normals and query_normals is not None:
                return p2p, feats, query_normals
            return p2p, feats
        else:
            p2p = self.regress_distance(
                original_pts, query_pts, global_feats, local_feats, query_normals
            )
            if return_normals and query_normals is not None:
                return p2p, query_normals
            return p2p