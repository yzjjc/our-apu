import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def weighted_pooling(features, rel_xyz):
    """
    基于距离的加权池化
    Args:
        features: (B, C, K) - 特征
        rel_xyz: (B, 3, K) - 相对坐标
    Returns:
        pooled: (B, C) - 池化后的特征
    """
    # 计算距离权重 (距离越近权重越大)
    dist = torch.norm(rel_xyz, dim=1, keepdim=True)  # (B, 1, K)
    weights = 1.0 / (dist + 1e-6)  # (B, 1, K)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # 归一化
    
    # 加权求和
    pooled = (features * weights).sum(dim=-1)  # (B, C)
    return pooled


class FeatureAwarePatchMLP(nn.Module):
    """
    融合几何特征和共享特征的 Patch MLP
    """
    def __init__(self, in_channel, out_dim):
        super(FeatureAwarePatchMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rel_xyz, shared_feat):
        """
        Args:
            rel_xyz: (B, 3, K) - 相对坐标
            shared_feat: (B, C, K) - 从 FeatureExtractor 获得的特征
        Returns:
            out: (B, out_dim, K)
        """
        # 计算几何特征
        dist = torch.norm(rel_xyz, dim=1, keepdim=True)  # (B, 1, K)
        
        # 拼接: [相对坐标, 距离, 共享特征]
        inp = torch.cat([rel_xyz, dist, shared_feat], dim=1)  # (B, 4+C, K)
        
        return self.mlp(inp)


class SharedPFFEstimator(nn.Module):
    """
    共享特征的多尺度法线估计器
    集成到 LDI 框架中，复用 FeatureExtractor 的特征
    """
    def __init__(self, args):
        super(SharedPFFEstimator, self).__init__()
        
        self.feat_dim = args.feat_dim  # 来自 FeatureExtractor
        self.hidden_dim = 64
        self.scales = [16, 32, 48]  # 多尺度邻域大小
        
        # 输入通道 = 4(几何) + feat_dim(共享特征)
        in_channel = 4 + self.feat_dim
        
        # 三个尺度的 MLP
        self.scale_mlps = nn.ModuleList([
            FeatureAwarePatchMLP(in_channel, self.hidden_dim),
            FeatureAwarePatchMLP(in_channel, self.hidden_dim),
            FeatureAwarePatchMLP(in_channel, self.hidden_dim),
        ])

        # 跨尺度融合层
        self.fusion1 = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 法线回归头
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # 输出 (nx, ny, nz)
        )

    def forward(self, patch_xyz, center_xyz, patch_feats):
        """
        Args:
            patch_xyz: (B, 3, K_max) - Patch 中邻居的坐标
            center_xyz: (B, 3) - 中心点坐标
            patch_feats: (B, C, K_max) - 来自 FeatureExtractor 的特征
        Returns:
            normals: (B, 3) - 归一化的法线向量
        """
        B = patch_xyz.shape[0]
        
        # 1. 计算相对坐标
        rel_xyz_all = patch_xyz - center_xyz.unsqueeze(-1)  # (B, 3, K_max)
        
        # 2. 多尺度处理
        feats_list = []
        for i, scale_k in enumerate(self.scales):
            # 切片: 取最近的 scale_k 个点
            curr_rel_xyz = rel_xyz_all[:, :, :scale_k]  # (B, 3, scale_k)
            curr_feat = patch_feats[:, :, :scale_k]     # (B, C, scale_k)
            
            # 通过对应尺度的 MLP
            f = self.scale_mlps[i](curr_rel_xyz, curr_feat)  # (B, 64, scale_k)
            
            # 加权池化
            f_pool = weighted_pooling(f, curr_rel_xyz)  # (B, 64)
            feats_list.append(f_pool)

        # 3. 跨尺度融合
        base = feats_list[0]  # (B, 64)
        
        # 融合中等尺度
        fused1 = torch.cat([base, feats_list[1]], dim=1)  # (B, 128)
        base = base + self.fusion1(fused1)  # (B, 64)
        
        # 融合大尺度
        fused2 = torch.cat([base, feats_list[2]], dim=1)  # (B, 128)
        base = base + self.fusion2(fused2)  # (B, 64)

        # 4. 预测法线并归一化
        normals = self.out_layer(base)  # (B, 3)
        normals = F.normalize(normals, p=2, dim=1)
        
        return normals