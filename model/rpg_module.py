"""
RPG-HoGRN: Relation Path Guided Module (Refactored)

主要改进：
1. P0: 软阈值替换硬截断
2. P1: 简化 AdaptiveFusion
3. P2: 分路径聚合 + 注意力加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class PathGuidedAggregator(nn.Module):
    """
    Path-Guided Aggregator: 分路径聚合 + 注意力加权
    """

    def __init__(self, embed_dim, num_relations, frequent_paths, rel_to_paths,
                 top_k_paths=5, sparse_threshold=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.frequent_paths = frequent_paths
        self.rel_to_paths = rel_to_paths
        self.top_k_paths = top_k_paths
        self.sparse_threshold = sparse_threshold

        # 软阈值的温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(2.0))

        # 路径注意力网络
        self.path_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0

        print(f"[RPG-Aggregator] embed_dim={embed_dim}, top_k={top_k_paths}, sparse_threshold={sparse_threshold}")
        print(f"[RPG-Aggregator] Loaded {len(frequent_paths)} paths, {len(rel_to_paths)} relations indexed")

        # 分路径存储（不合并）
        self.path_matrices_list = None  # [(path_tuple, sparse_matrix, norm_vector), ...]
        self.rel_matrices = None
        self.num_ent = None

    def build_relation_matrices(self, edge_index, edge_type, num_ent, device):
        """Build sparse adjacency matrix for each relation."""
        print(f"[RPG-Aggregator] Building relation matrices...")
        self.rel_matrices = {}

        for r in range(self.num_relations * 2):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            r_edge = edge_index[:, mask]
            row, col = r_edge[0], r_edge[1]
            val = torch.ones(row.size(0), device=device)
            indices = torch.stack([row, col], dim=0)
            self.rel_matrices[r] = torch.sparse_coo_tensor(
                indices, val, (num_ent, num_ent), device=device
            ).coalesce()

        print(f"[RPG-Aggregator] Built {len(self.rel_matrices)} relation matrices")

    def build_path_matrices(self, num_ent, device):
        """Build separate path matrix for each frequent path (不合并)."""
        print(f"[RPG-Aggregator] Building path matrices (分路径存储)...")

        self.path_matrices_list = []
        valid_count = 0
        skip_count = 0

        # 按频率排序，取 top_k
        sorted_paths = sorted(self.frequent_paths.items(), key=lambda x: -x[1])

        for path, freq in sorted_paths[:self.top_k_paths * 10]:  # 多取一些，防止无效路径
            if len(self.path_matrices_list) >= self.top_k_paths:
                break

            # 检查路径的第一个关系是否存在
            if path[0] not in self.rel_matrices:
                skip_count += 1
                continue

            # 计算路径矩阵: A_r1 @ A_r2 @ ...
            path_mat = self.rel_matrices[path[0]]
            valid = True

            for r in path[1:]:
                if r not in self.rel_matrices:
                    valid = False
                    break
                path_mat = torch.sparse.mm(path_mat, self.rel_matrices[r])

            if not valid:
                skip_count += 1
                continue

            # 提取并归一化
            path_mat = path_mat.coalesce()
            indices = path_mat.indices()

            if indices.size(1) == 0:
                skip_count += 1
                continue

            row, col = indices[0], indices[1]
            val = torch.ones(row.size(0), device=device)

            # 计算每个节点的出边数（用于归一化）
            node_edge_count = scatter_add(val, row, dim=0, dim_size=num_ent)
            node_edge_count = node_edge_count.clamp(min=1)

            # 重建归一化后的稀疏矩阵
            norm_val = val / node_edge_count[row]
            norm_path_mat = torch.sparse_coo_tensor(
                indices, norm_val, (num_ent, num_ent), device=device
            ).coalesce()

            self.path_matrices_list.append((path, norm_path_mat))
            valid_count += 1

        print(f"[RPG-Aggregator] Built {valid_count} path matrices, skipped {skip_count}")

        if valid_count > 0:
            print(f"[RPG-Aggregator] Top paths:")
            for i, (path, _) in enumerate(self.path_matrices_list[:5]):
                print(f"  Path {i}: {path}")

    def forward(self, entity_embeds, node_degrees, edge_index, edge_type):
        """
        分路径聚合 + 注意力加权 + 软阈值
        """
        N, dim = entity_embeds.shape
        device = entity_embeds.device
        self.num_ent = N

        # Build matrices on first call
        if self.path_matrices_list is None:
            self.build_relation_matrices(edge_index, edge_type, N, device)
            self.build_path_matrices(N, device)

        # 如果没有有效路径，返回零
        if not self.path_matrices_list:
            return torch.zeros_like(entity_embeds)

        # ===== 1. 分路径计算 remote features =====
        path_features = []
        for path, path_mat in self.path_matrices_list:
            # 稀疏矩阵乘法：聚合远程特征
            h_path = torch.sparse.mm(path_mat, entity_embeds)  # [N, dim]
            path_features.append(h_path)

        # 堆叠 [N, num_paths, dim]
        path_features = torch.stack(path_features, dim=1)
        num_paths = path_features.size(1)

        # ===== 2. 注意力加权 =====
        # 计算每条路径的注意力分数
        attn_scores = self.path_attention(path_features)  # [N, num_paths, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, num_paths, 1]

        # 加权求和
        remote_features = (attn_weights * path_features).sum(dim=1)  # [N, dim]

        # 注意：软阈值移到 AdaptiveFusion 中统一处理，避免重复应用

        # Log statistics
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            avg_attn = attn_weights.mean(dim=0).squeeze()  # [num_paths]
            remote_norm = remote_features.norm(dim=1).mean().item()
            print(f"[RPG-Aggregator] Step {self.forward_count}: "
                  f"remote_norm={remote_norm:.3f}, "
                  f"attn_dist={avg_attn.tolist()[:3]}")

        return remote_features


class AdaptiveFusion(nn.Module):
    """
    简化的 Adaptive Fusion: 用软阈值直接作为融合权重
    """

    def __init__(self, embed_dim, sparse_threshold=5, dropout=0.1):
        super().__init__()
        self.sparse_threshold = sparse_threshold
        self.dropout = nn.Dropout(dropout)

        # 可学习的融合参数
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.scale = nn.Parameter(torch.tensor(1.0))  # 控制融合强度（从0.5提高到1.0）

        # 可选：特征变换（让 remote 和 local 在同一空间）
        self.transform = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.transform.weight)  # 初始化为单位矩阵

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0
        self.alpha_sum = 0.0

        print(f"[RPG-Fusion] embed_dim={embed_dim}, sparse_threshold={sparse_threshold}, dropout={dropout}")

    def forward(self, h_local, h_remote, node_degrees):
        """
        Args:
            h_local: [N, dim] local GCN features
            h_remote: [N, dim] remote path features
            node_degrees: [N] node degrees

        Returns:
            h_fused: [N, dim] fused features
            alpha: [N] fusion weights
        """
        # ===== 软阈值作为融合权重 =====
        # 度数低 -> alpha 大，度数高 -> alpha 小
        alpha = torch.sigmoid(
            (self.sparse_threshold - node_degrees.float()) / self.temperature
        )
        alpha = alpha * self.scale  # 缩放到合理范围
        alpha = alpha.unsqueeze(1)  # [N, 1]

        # 特征变换 + dropout
        h_remote = self.transform(h_remote)
        h_remote = self.dropout(h_remote)

        # 残差融合
        h_fused = h_local + alpha * h_remote

        # Log statistics
        self.forward_count += 1
        self.alpha_sum += alpha.mean().item()

        if self.forward_count % self.log_interval == 0:
            avg_alpha = self.alpha_sum / self.forward_count
            curr_alpha = alpha.mean().item()
            print(f"[RPG-Fusion] Step {self.forward_count}: "
                  f"curr_alpha={curr_alpha:.4f}, avg_alpha={avg_alpha:.4f}, "
                  f"scale={self.scale.item():.4f}, temp={self.temperature.item():.4f}")

        return h_fused, alpha.squeeze(1)


# ===== 保留旧的类名以兼容，但标记为废弃 =====
class PathEncoder(nn.Module):
    """
    [DEPRECATED] Encode relation paths into vectors using LSTM + Attention.
    保留以兼容旧代码，新代码请使用 PathGuidedAggregator 的内置注意力。
    """

    def __init__(self, rel_dim, hidden_dim, num_relations):
        super().__init__()
        self.rel_dim = rel_dim
        self.hidden_dim = hidden_dim

        self.rel_embed = nn.Embedding(num_relations * 2, rel_dim)
        self.lstm = nn.LSTM(
            input_size=rel_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output_proj = nn.Linear(hidden_dim * 2, rel_dim)

    def forward(self, path_relations):
        rel_embeds = self.rel_embed(path_relations)
        outputs, _ = self.lstm(rel_embeds)
        attn_scores = self.attention(outputs)
        attn_weights = F.softmax(attn_scores, dim=1)
        path_repr = (attn_weights * outputs).sum(dim=1)
        path_embed = self.output_proj(path_repr)
        return path_embed
