import torch
import torch.nn as nn
from einops import repeat
from models.MultiScaleFeatureExtractor import MultiScaleFeatureExtractor
from models.P2PRegressor import P2PRegressor
from models.AttentionLayer import WeightLayer
from models.utils import get_knn_pts, index_points


class P2PNet_MultiScale(nn.Module):
    """Multi-scale P2PNet for distance prediction"""
    def __init__(self, args):
        super(P2PNet_MultiScale, self).__init__()

        self.args = args
        self.scales = getattr(args, 'scales', [8, 16, 32])
        
        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(args)
        
        # P2P regressor
        self.p2p_regressor = P2PRegressor(args)
        
        # Weight layers for each block (attention-based feature interpolation)
        self.weight = nn.ModuleList([])
        self.global_prior = False
        for i in range(args.block_num + 1):
            self.weight.append(WeightLayer(args))

    def set_global_mode(self):
        """Set to global mode for training global field"""
        self.global_prior = True

    def extract_feature(self, original_pts):
        """
        Extract multi-scale features
        Args:
            original_pts: (b, 3, n) input points
        Returns:
            global_feats: (b, c) global features
            local_feats: list of (b, c, n) local features
        """
        global_feats, local_feats = self.feature_extractor(original_pts)
        return global_feats, local_feats

    def interpolate_feature(self, original_pts, query_pts, local_feat):
        """
        Interpolate features for query points using k-NN
        Args:
            original_pts: (b, 3, n) original points
            query_pts: (b, 3, m) query points
            local_feat: (b, c, n) local features
        Returns:
            interpolated_feat: (b, c, m) interpolated features
        """
        k = 3
        # k-NN search
        knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
        
        # Distance-based weights
        repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=k)
        dist = torch.norm(knn_pts - repeat_query_pts, p=2, dim=1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        # Weighted interpolation
        knn_feat = index_points(local_feat, knn_idx)
        interpolated_feat = knn_feat * weight.unsqueeze(1)
        interpolated_feat = torch.sum(interpolated_feat, dim=-1)
        
        return interpolated_feat

    def regress_distance(self, original_pts, query_pts, global_feats, 
                        local_feats, return_feat=False):
        """
        Regress point-to-point distance for query points
        Args:
            original_pts: (b, 3, n) original points
            query_pts: (b, 3, m) query points
            global_feats: (b, c) global features
            local_feats: list of (b, c, n) local features
            return_feat: whether to return aggregated features
        Returns:
            p2p: (b, 1, m) predicted distances
            agg_feats (optional): aggregated features
        """
        # Expand global features
        global_feats = repeat(global_feats, 'b c -> b c n', n=query_pts.shape[-1])
        
        # Interpolate local features using attention-based weights
        interpolated_local_feats = []
        for i in range(self.args.block_num + 1):
            interpolated_feat = self.weight[i](original_pts, query_pts, local_feats[i])
            interpolated_local_feats.append(interpolated_feat)
        
        # Aggregate local features
        agg_local_feats = torch.cat(interpolated_local_feats, dim=1)
        
        # Concatenate with query coordinates and global features
        agg_feats = torch.cat((query_pts, agg_local_feats, global_feats), dim=1)
        
        # Predict distance
        p2p = self.p2p_regressor(agg_feats)
        
        if not return_feat:
            return p2p
        else:
            return p2p, torch.cat((agg_local_feats, global_feats), dim=1)

    def forward(self, original_pts, query_pts, return_feat=False):
        """
        Forward pass
        Args:
            original_pts: (b, 3, n) original points
            query_pts: (b, 3, m) query points
            return_feat: whether to return features
        Returns:
            p2p: (b, 1, m) predicted distances
        """
        # Extract features with gradient control for global mode
        if self.global_prior:
            with torch.no_grad():
                global_feats, local_feats = self.extract_feature(original_pts)
        else:
            global_feats, local_feats = self.extract_feature(original_pts)
        
        # Regress distance
        return self.regress_distance(
            original_pts, query_pts, global_feats, local_feats, return_feat
        )