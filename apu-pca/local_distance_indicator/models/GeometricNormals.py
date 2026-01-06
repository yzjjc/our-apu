import torch
import torch.nn.functional as F
from einops import rearrange

class PCANormalEstimator:
    """
    使用 PCA 估计点云法线（工业级健壮版本）
    特性：全向量化、分块计算防OOM、数值稳定、梯度截断
    """
    def __init__(self, k_neighbors=20):
        self.k = k_neighbors
    
    @torch.no_grad()
    def estimate_query_normals(self, original_pts, query_pts):
        """
        为查询点估计法线（基于原始点云的邻域）
        Args:
            original_pts: (B, 3, N) - 原始点云
            query_pts: (B, 3, M) - 查询点
        Returns:
            query_normals: (B, 3, M) - 查询点的法线
        """
        B, C, N = original_pts.shape
        _, _, M = query_pts.shape
        K = self.k
        device = original_pts.device
        
        # 1. 维度调整
        orig = original_pts.transpose(1, 2).contiguous()  # (B, N, 3)
        query = query_pts.transpose(1, 2).contiguous()    # (B, M, 3)
        
        # 2. 分块计算 KNN（避免显存爆炸的关键步骤）
        # 将 M 个查询点分成小块，每块大小为 512
        chunk_size = 512 
        all_normals = []
        
        for i in range(0, M, chunk_size):
            end_i = min(i + chunk_size, M)
            query_chunk = query[:, i:end_i, :] # (B, chunk, 3)
            
            # --- A. 计算 KNN ---
            # cdist: (B, chunk, 3) vs (B, N, 3) -> (B, chunk, N)
            dist_chunk = torch.cdist(query_chunk, orig)
            _, knn_idx_chunk = torch.topk(
                dist_chunk, k=min(K, N), dim=2, largest=False
            ) # (B, chunk, K)
            
            # --- B. 提取邻域点 (高级索引) ---
            # 构造 Batch 索引: (B, 1, 1) -> (B, chunk, K)
            current_chunk_size = knn_idx_chunk.shape[1]
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, current_chunk_size, K)
            
            # 提取: orig[batch, knn_idx] -> (B, chunk, K, 3)
            neighbors = orig[batch_idx, knn_idx_chunk, :]
            
            # --- C. PCA 计算 (协方差矩阵) ---
            # 中心化
            mean = neighbors.mean(dim=2, keepdim=True) # (B, chunk, 1, 3)
            centered = neighbors - mean                # (B, chunk, K, 3)
            
            # 协方差矩阵: (B, chunk, 3, K) @ (B, chunk, K, 3) -> (B, chunk, 3, 3)
            # 使用 float32 保证精度，如果原来是 float16 可能会不稳定
            cov = torch.matmul(centered.transpose(2, 3), centered)
            
            # --- D. 特征值分解 (增加鲁棒性) ---
            try:
                # 尝试 GPU 分解
                _, eigenvectors = torch.linalg.eigh(cov) 
            except RuntimeError:
                # 如果矩阵奇异导致分解失败，尝试加正则项
                try:
                    eps = 1e-6
                    eye = torch.eye(3, device=device).view(1, 1, 3, 3)
                    _, eigenvectors = torch.linalg.eigh(cov + eps * eye)
                except RuntimeError:
                    # 如果依然失败，回退到 CPU 计算 (最后的保底手段)
                    cov_cpu = cov.cpu()
                    _, eigenvectors_cpu = torch.linalg.eigh(cov_cpu)
                    eigenvectors = eigenvectors_cpu.to(device)

            # 最小特征值对应的特征向量即为法线 (eigh 返回特征值从小到大排列，所以取列 0)
            normals_chunk = eigenvectors[..., 0] # (B, chunk, 3)
            
            # --- E. 方向一致性 ---
            # 策略：指向局部邻域质心的外侧 (query - patch_centroid)
            patch_centroid = mean.squeeze(2) # (B, chunk, 3)
            view_dir = query_chunk - patch_centroid 
            
            # 批量点积
            dot = (normals_chunk * view_dir).sum(dim=-1) # (B, chunk)
            
            # 翻转: sign为-1则变号，为1则不变 (0的情况设为1)
            sign = torch.sign(dot)
            sign[sign == 0] = 1
            normals_chunk = normals_chunk * sign.unsqueeze(-1)
            
            # 归一化
            normals_chunk = F.normalize(normals_chunk, p=2, dim=-1)
            
            all_normals.append(normals_chunk)
        
        # 3. 合并结果
        query_normals = torch.cat(all_normals, dim=1) # (B, M, 3)
        
        return query_normals.transpose(1, 2).contiguous() # (B, 3, M)

    @torch.no_grad()
    def estimate_batch_normals(self, points_batch):
        """
        批量估计点云自身的法线 (输入即输出)
        Args:
            points_batch: (B, N, 3)
        Returns:
            normals: (B, N, 3)
        """
        # 复用 estimate_query_normals，将 query 设为 points 自身即可
        # 注意输入需要转置适应 estimate_query_normals 的接口 (B, 3, N)
        points_transposed = points_batch.transpose(1, 2)
        normals_transposed = self.estimate_query_normals(points_transposed, points_transposed)
        return normals_transposed.transpose(1, 2)


class AdaptivePCANormalEstimator(PCANormalEstimator):
    """
    自适应 PCA 法线估计（根据点云密度调整邻域大小）
    """
    def __init__(self, k_neighbors=20):
        super().__init__(k_neighbors)
    
    def estimate_adaptive_k(self, points):
        """
        根据点云密度自适应选择 k
        Args:
            points: (N, 3)
        Returns:
            adaptive_k: int
        """
        N = points.shape[0]
        device = points.device
        
        # 采样一些点估计平均密度
        sample_size = min(100, N)
        # 使用 torch 生成随机索引
        sample_idx = torch.randperm(N, device=device)[:sample_size]
        sample_pts = points[sample_idx]
        
        # 计算到最近邻的平均距离
        dist_matrix = torch.cdist(sample_pts, points)
        # 取第 10 个最近邻的距离作为参考
        k_target = min(10, N)
        kth_dist = torch.topk(dist_matrix, k=k_target, largest=False, dim=1)[0][:, -1]
        avg_dist = kth_dist.mean()
        
        # 根据密度调整 k (阈值需根据数据尺度调整，这里是示例值)
        if avg_dist < 0.01:
            return min(30, N)
        elif avg_dist < 0.05:
            return min(20, N)
        else:
            return min(15, N)