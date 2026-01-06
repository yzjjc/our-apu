"""
ANS (Adaptive Noise Suppression) Module
用于多尺度特征提取中的自适应噪声抑制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvPosEnc(nn.Module):
    """
    卷积位置编码
    使用深度可分离卷积捕获局部空间关系
    """
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv1d(
            dim, dim,
            kernel_size=k,
            padding=k // 2,
            groups=dim  # 深度可分离卷积
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, N) 特征
        Returns:
            out: (B, C, N) 添加位置编码后的特征
        """
        # 卷积位置编码
        pos_encoding = self.proj(x)
        # 残差连接
        out = x + pos_encoding
        return out


class CrossAttention(nn.Module):
    """
    跨注意力模块
    实现Query特征 attend to Key/Value特征
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 1/√d_k
        
        # Query投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Key/Value投影
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, kv_feat, query_feat):
        """
        Args:
            kv_feat: (B, C, N) Key/Value特征（参考特征）
            query_feat: (B, C, M) Query特征（含噪特征）
        Returns:
            out: (B, C, M) 注意力融合后的特征
        """
        B, C, N = kv_feat.shape
        _, _, M = query_feat.shape
        
        # 转换维度: (B, C, N) -> (B, N, C)
        kv_feat = kv_feat.transpose(1, 2)
        query_feat = query_feat.transpose(1, 2)
        
        # 生成K, V: (B, N, C) -> (B, N, 2, num_heads, C//num_heads)
        kv = self.kv(kv_feat).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]
        
        # 生成Q: (B, M, C) -> (B, num_heads, M, head_dim)
        q = self.q(query_feat).reshape(B, M, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        
        # 计算注意力权重: (B, num_heads, M, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权聚合: (B, num_heads, M, head_dim)
        out = attn @ v
        
        # 合并多头: (B, M, C)
        out = out.transpose(1, 2).reshape(B, M, C)
        
        # 输出投影
        out = self.proj(out)
        
        # 转换回: (B, M, C) -> (B, C, M)
        out = out.transpose(1, 2)
        
        return out


class ANSModule(nn.Module):
    
    def __init__(self, dim, num_heads=4, qkv_bias=False, use_bidirectional=True):
        super().__init__()
        self.use_bidirectional = use_bidirectional
        
        # 卷积位置编码
        self.pos_encoder = ConvPosEnc(dim=dim, k=3)
        
        # 双向跨注意力
        self.cross_attn_n2r = CrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias
        )
        
        if use_bidirectional:
            self.cross_attn_r2n = CrossAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias
            )
        
        # 两个独立的FFN处理两个流
        self.mlp_noisy = nn.Sequential(
            nn.Conv1d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv1d(dim * 4, dim, 1)
        )
        
        if use_bidirectional:
            self.mlp_ref = nn.Sequential(
                nn.Conv1d(dim, dim * 4, 1),
                nn.GELU(),
                nn.Conv1d(dim * 4, dim, 1)
            )
        
        # Layer Norm
        self.norm1_noisy = nn.BatchNorm1d(dim)
        self.norm2_noisy = nn.BatchNorm1d(dim)
        
        if use_bidirectional:
            self.norm1_ref = nn.BatchNorm1d(dim)
            self.norm2_ref = nn.BatchNorm1d(dim)
        
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, noisy_feat, ref_feat):
        """
        Args:
            noisy_feat: (B, C, N) 含噪特征
            ref_feat: (B, C, M) 参考特征
        Returns:
            noisy_out: (B, C, N) 去噪后的含噪特征
            ref_out: (B, C, M) 增强后的参考特征
        """
        B, C, N = noisy_feat.shape
        _, _, M = ref_feat.shape
        
        # ========== 步骤1：位置编码（对原始特征） ========== 
        noisy_feat_pos = self.pos_encoder(noisy_feat)
        ref_feat_pos = self.pos_encoder(ref_feat)
        
        # ========== 步骤2：第一次跨注意力（含噪 → 参考） ========== 
        # 含噪特征作为Query，参考特征作为K/V
        noisy_attn_1 = self.cross_attn_n2r(
            kv_feat=ref_feat_pos,      # K/V: 参考特征
            query_feat=noisy_feat_pos   # Q: 含噪特征
        )
        # 残差连接 + 归一化
        noisy_feat_1 = noisy_feat + self.gamma * noisy_attn_1
        noisy_feat_1 = self.norm1_noisy(noisy_feat_1)
        
        if self.use_bidirectional:
            # ========== 步骤3：第二次跨注意力（参考 → 含噪） ========== 
            # ⚠️ 关键修正：对更新后的特征重新做位置编码
            noisy_feat_1_pos = self.pos_encoder(noisy_feat_1)
            
            # 参考特征作为Query，更新后的含噪特征作为K/V
            ref_attn_1 = self.cross_attn_r2n(
                kv_feat=noisy_feat_1_pos,  # K/V: 更新后的含噪特征
                query_feat=ref_feat_pos     # Q: 参考特征
            )
            # 残差连接 + 归一化
            ref_feat_1 = ref_feat + self.gamma * ref_attn_1
            ref_feat_1 = self.norm1_ref(ref_feat_1)
            
            # ========== 步骤4：FFN精炼（两个流都要处理） ========== 
            # 含噪流
            noisy_feat_ffn = self.mlp_noisy(noisy_feat_1)
            noisy_feat_out = noisy_feat_1 + noisy_feat_ffn
            noisy_feat_out = self.norm2_noisy(noisy_feat_out)
            
            # 参考流
            ref_feat_ffn = self.mlp_ref(ref_feat_1)
            ref_feat_out = ref_feat_1 + ref_feat_ffn
            ref_feat_out = self.norm2_ref(ref_feat_out)
            
            return noisy_feat_out, ref_feat_out  
        
        else:
            # 单向模式：只处理含噪流
            noisy_feat_ffn = self.mlp_noisy(noisy_feat_1)
            noisy_feat_out = noisy_feat_1 + noisy_feat_ffn
            noisy_feat_out = self.norm2_noisy(noisy_feat_out)
            
            return noisy_feat_out, ref_feat  # 参考特征不变