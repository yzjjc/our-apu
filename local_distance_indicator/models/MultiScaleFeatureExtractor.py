#多尺度特征提取

import torch
import torch.nn as nn
from models.utils import get_knn_pts, index_points
from einops import rearrange
from models.pointops.functions import pointops


class Point3DConv(nn.Module):
    """3D point convolution layer"""
    def __init__(self, args, growth_rate=None):
        super(Point3DConv, self).__init__()
        
        growth_rate = growth_rate or args.growth_rate
        self.k = args.k
        
        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, growth_rate, 1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv_feats = nn.Sequential(
            nn.Conv2d(args.bn_size * growth_rate, growth_rate, 1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(growth_rate, growth_rate, 1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, pts, knn_idx=None):
        if knn_idx is None:
            knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
        else:
            knn_pts = index_points(pts, knn_idx)
        
        knn_delta = knn_pts - pts[..., None]
        knn_delta = self.conv_delta(knn_delta)
        
        knn_feats = index_points(feats, knn_idx)
        knn_feats = self.conv_feats(knn_feats)
        
        new_feats = knn_delta * knn_feats
        new_feats = self.post_conv(new_feats)
        new_feats = new_feats.sum(dim=-1)
        
        return new_feats


class DenseLayer(nn.Module):
    """Dense layer with 3D point convolution"""
    def __init__(self, args, input_dim, growth_rate=None):
        super(DenseLayer, self).__init__()
        
        growth_rate = growth_rate or args.growth_rate
        
        self.conv_bottle = nn.Sequential(
            nn.Conv1d(input_dim, args.bn_size * growth_rate, 1),
            nn.BatchNorm1d(args.bn_size * growth_rate),
            nn.ReLU(inplace=True)
        )
        self.point_conv = Point3DConv(args, growth_rate)

    def forward(self, feats, pts, knn_idx=None):
        new_feats = self.conv_bottle(feats)
        new_feats = self.point_conv(new_feats, pts, knn_idx)
        return torch.cat((feats, new_feats), dim=1)


class DenseUnit(nn.Module):
    """Dense unit with multiple dense layers"""
    def __init__(self, args, growth_rate=None):
        super(DenseUnit, self).__init__()
        
        growth_rate = growth_rate or args.growth_rate
        
        self.dense_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.dense_layers.append(
                DenseLayer(args, args.feat_dim + i * growth_rate, growth_rate)
            )

    def forward(self, feats, pts, knn_idx=None):
        for dense_layer in self.dense_layers:
            feats = dense_layer(feats, pts, knn_idx)
        return feats


class Transition(nn.Module):
    """Transition layer to reduce feature dimensions"""
    def __init__(self, args, growth_rate=None):
        super(Transition, self).__init__()
        
        growth_rate = growth_rate or args.growth_rate
        input_dim = args.feat_dim + args.layer_num * growth_rate
        
        self.trans = nn.Sequential(
            nn.Conv1d(input_dim, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        return self.trans(feats)


class ScaleFusionModule(nn.Module):
    """Multi-scale feature fusion module"""
    def __init__(self, feat_dim, num_scales, fusion_type='hierarchical'):
        super(ScaleFusionModule, self).__init__()
        
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        
        if fusion_type == 'concat':
            # Simple concatenation + MLP
            self.fusion = nn.Sequential(
                nn.Conv1d(feat_dim * num_scales, feat_dim * 2, 1),
                nn.BatchNorm1d(feat_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(feat_dim * 2, feat_dim, 1),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True)
            )
        
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(feat_dim, feat_dim // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(feat_dim // 4, 1, 1)
                ) for _ in range(num_scales)
            ])
            
        elif fusion_type == 'hierarchical':
            # Hierarchical fusion (scale by scale)
            self.fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(feat_dim * 2, feat_dim, 1),
                    nn.BatchNorm1d(feat_dim),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_scales - 1)
            ])

    def forward(self, scale_features):
        """
        Args:
            scale_features: List of features [(b, c, n), ...] for each scale
        Returns:
            fused_features: (b, c, n)
        """
        if self.fusion_type == 'concat':
            # Concatenate all scales
            concat_feat = torch.cat(scale_features, dim=1)
            return self.fusion(concat_feat)
        
        elif self.fusion_type == 'attention':
            # Compute attention weights
            attention_weights = []
            for i, feat in enumerate(scale_features):
                attn = self.attention[i](feat)
                attention_weights.append(attn)
            
            # Softmax over scales
            attention_weights = torch.stack(attention_weights, dim=1)  # (b, s, 1, n)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Weighted sum
            scale_features = torch.stack(scale_features, dim=1)  # (b, s, c, n)
            fused = (scale_features * attention_weights).sum(dim=1)
            return fused
        
        elif self.fusion_type == 'hierarchical':
            # Hierarchical fusion: fine to coarse
            fused = scale_features[0]  # Start with finest scale
            for i in range(1, len(scale_features)):
                concat = torch.cat([fused, scale_features[i]], dim=1)
                fused = self.fusion_layers[i-1](concat)
            return fused


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor with hierarchical fusion"""
    def __init__(self, args):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.scales = getattr(args, 'scales', [8, 16, 32])  # Multi-scale k values
        self.num_scales = len(self.scales)
        self.fusion_type = getattr(args, 'fusion_type', 'hierarchical')
        
        # Initial convolution (shared across scales)
        self.conv_init = nn.Sequential(
            nn.Conv1d(3, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale dense blocks
        self.scale_extractors = nn.ModuleList()
        for scale_k in self.scales:
            # Create a copy of args with different k
            scale_args = type('obj', (object,), {
                'k': scale_k,
                'feat_dim': args.feat_dim,
                'block_num': args.block_num,
                'layer_num': args.layer_num,
                'bn_size': args.bn_size,
                'growth_rate': args.growth_rate
            })()
            
            # Dense blocks for this scale
            scale_blocks = nn.ModuleList()
            for _ in range(args.block_num):
                scale_blocks.append(nn.ModuleList([
                    DenseUnit(scale_args),
                    Transition(scale_args)
                ]))
            self.scale_extractors.append(scale_blocks)
        
        # Multi-scale fusion module
        self.fusion = ScaleFusionModule(
            args.feat_dim, 
            self.num_scales, 
            self.fusion_type
        )

    def forward(self, pts):
        """
        Args:
            pts: (b, 3, n) input point cloud
        Returns:
            global_feats: (b, c) global features
            local_feats: list of (b, c, n) local features at each stage
        """
        # Compute KNN indices for each scale
        pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
        knn_indices = []
        for scale_k in self.scales:
            knn_idx = pointops.knnquery_heap(scale_k, pts_trans, pts_trans).long()
            knn_indices.append(knn_idx)
        
        # Initial features (shared)
        init_feats = self.conv_init(pts)
        
        # Extract features at each scale
        scale_features_all = []  # Store features at each block for all scales
        
        for scale_idx, (scale_blocks, knn_idx) in enumerate(zip(self.scale_extractors, knn_indices)):
            scale_feats = init_feats
            scale_local_feats = [scale_feats]
            
            for dense_block, trans in scale_blocks:
                scale_feats = dense_block(scale_feats, pts, knn_idx)
                scale_feats = trans(scale_feats)
                scale_local_feats.append(scale_feats)
            
            scale_features_all.append(scale_local_feats)
        
        # Fuse multi-scale features at each block level
        fused_local_feats = []
        for block_idx in range(len(scale_features_all[0])):
            # Get features from all scales at this block
            block_features = [scale_features_all[s][block_idx] 
                            for s in range(self.num_scales)]
            # Fuse them
            fused_feat = self.fusion(block_features)
            fused_local_feats.append(fused_feat)
        
        # Global features from the last fused features
        global_feats = fused_local_feats[-1].max(dim=-1)[0]
        
        return global_feats, fused_local_feats