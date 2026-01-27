"""
RPG-HoGRN: Relation Path Guided Module (Refactored)

主要改进：
1. P0: 软阈值替换硬截断
2. P1: 简化 AdaptiveFusion
3. P2: 分路径聚合 + 注意力加权
4. P3: Query-Aware Path Attention (查询感知的路径注意力)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

logger = logging.getLogger(__name__)


class PathGuidedAggregator(nn.Module):
    """
    Path-Guided Aggregator: 分路径聚合 + 查询感知注意力加权

    P3改进：路径注意力权重同时考虑：
    1. 路径聚合的特征质量（原有）
    2. 查询关系与路径的语义相关性（新增）
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

        # 路径注意力网络（基于特征质量）
        self.path_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

        # ===== P3: Query-Aware Path Attention =====
        # 关系嵌入（用于编码路径）
        self.rel_embed = nn.Embedding(num_relations * 2, embed_dim)

        # 路径编码器：将路径中的关系序列编码为向量
        self.path_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
            bidirectional=False
        )

        # 查询-路径相关性计算
        self.query_path_bilinear = nn.Bilinear(embed_dim, embed_dim, 1)

        # 融合权重：平衡特征注意力和查询相关性
        self.query_weight = nn.Parameter(torch.tensor(0.5))

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0

        logger.info(f"[RPG-Aggregator] embed_dim={embed_dim}, top_k={top_k_paths}, sparse_threshold={sparse_threshold}")
        logger.info(f"[RPG-Aggregator] Loaded {len(frequent_paths)} paths, {len(rel_to_paths)} relations indexed")
        logger.info(f"[RPG-Aggregator] P3 Query-Aware Attention enabled")

        # 分路径存储（不合并）
        self.path_matrices_list = None  # [(path_tuple, sparse_matrix, norm_vector), ...]
        self.path_relations = None  # 路径的关系ID列表，用于动态计算嵌入
        self.rel_matrices = None
        self.num_ent = None

    def build_relation_matrices(self, edge_index, edge_type, num_ent, device):
        """Build sparse adjacency matrix for each relation."""
        logger.info(f"[RPG-Aggregator] Building relation matrices...")
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

        logger.info(f"[RPG-Aggregator] Built {len(self.rel_matrices)} relation matrices")

    def build_path_matrices(self, num_ent, device):
        """Build separate path matrix for each frequent path (不合并)."""
        logger.info(f"[RPG-Aggregator] Building path matrices (分路径存储)...")

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

        logger.info(f"[RPG-Aggregator] Built {valid_count} path matrices, skipped {skip_count}")

        if valid_count > 0:
            logger.info(f"[RPG-Aggregator] Top paths:")
            for i, (path, _) in enumerate(self.path_matrices_list[:5]):
                logger.info(f"  Path {i}: {path}")

        # ===== P3: 预计算路径嵌入 =====
        self._build_path_embeddings(device)

    def _build_path_embeddings(self, device):
        """预计算每条路径的嵌入向量（每次forward时重新计算，以支持梯度更新）"""
        if not self.path_matrices_list:
            self.path_embeddings = None
            return

        # 存储路径的关系ID，用于每次forward时重新计算嵌入
        self.path_relations = []
        for path, _ in self.path_matrices_list:
            path_tensor = torch.tensor(list(path), dtype=torch.long, device=device)
            self.path_relations.append(path_tensor)

        logger.info(f"[RPG-Aggregator] Prepared {len(self.path_relations)} path relations for embedding")

    def _compute_path_embeddings(self):
        """每次forward时计算路径嵌入（支持梯度反传）"""
        if not self.path_relations:
            return None

        path_embeds = []
        for path_tensor in self.path_relations:
            # 获取关系嵌入
            rel_embeds = self.rel_embed(path_tensor)  # [path_len, embed_dim]
            # 通过GRU编码
            rel_embeds = rel_embeds.unsqueeze(0)  # [1, path_len, embed_dim]
            _, h_n = self.path_encoder(rel_embeds)  # h_n: [1, 1, embed_dim]
            path_embed = h_n.squeeze(0).squeeze(0)  # [embed_dim]
            path_embeds.append(path_embed)

        return torch.stack(path_embeds, dim=0)  # [num_paths, embed_dim]

    def forward(self, entity_embeds, node_degrees, edge_index, edge_type, rel_embed=None):
        """
        分路径聚合 + 查询感知注意力加权

        Args:
            entity_embeds: [N, dim] 实体嵌入
            node_degrees: [N] 节点度数
            edge_index: [2, E] 边索引
            edge_type: [E] 边类型
            rel_embed: [num_rel*2, dim] 关系嵌入（用于查询感知注意力）
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

        # ===== 2. 特征质量注意力（原有）=====
        feature_attn = self.path_attention(path_features)  # [N, num_paths, 1]

        # ===== 3. P3: 查询感知注意力（新增）=====
        if rel_embed is not None and hasattr(self, 'path_relations') and self.path_relations:
            # 每次forward重新计算路径嵌入（支持梯度）
            path_embeddings = self._compute_path_embeddings()  # [num_paths, dim]

            # 计算每条路径与所有关系的相关性
            # rel_embed: [num_rel*2, dim]
            num_rels = rel_embed.size(0)

            # 使用双线性层计算查询-路径相关性
            # 扩展维度以进行批量计算
            path_emb_exp = path_embeddings.unsqueeze(0).expand(num_rels, -1, -1)  # [num_rels, num_paths, dim]
            rel_emb_exp = rel_embed.unsqueeze(1).expand(-1, num_paths, -1)  # [num_rels, num_paths, dim]

            # 计算相关性分数
            query_path_scores = self.query_path_bilinear(
                rel_emb_exp.reshape(-1, dim),
                path_emb_exp.reshape(-1, dim)
            ).reshape(num_rels, num_paths)  # [num_rels, num_paths]

            # 对所有节点广播（每个节点根据其相关关系获得不同权重）
            # 这里简化处理：使用平均相关性作为路径的"全局重要性"
            query_attn = query_path_scores.mean(dim=0)  # [num_paths]
            query_attn = query_attn.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1)  # [N, num_paths, 1]

            # 融合两种注意力
            w = torch.sigmoid(self.query_weight)
            combined_attn = w * feature_attn + (1 - w) * query_attn
            attn_weights = F.softmax(combined_attn, dim=1)  # [N, num_paths, 1]
        else:
            # 如果没有关系嵌入，退回到仅使用特征注意力
            attn_weights = F.softmax(feature_attn, dim=1)

        # 加权求和
        remote_features = (attn_weights * path_features).sum(dim=1)  # [N, dim]

        # Log statistics
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            avg_attn = attn_weights.mean(dim=0).squeeze()  # [num_paths]
            remote_norm = remote_features.norm(dim=1).mean().item()
            w_val = torch.sigmoid(self.query_weight).item() if rel_embed is not None else 1.0
            logger.info(f"[RPG-Aggregator] Step {self.forward_count}: "
                  f"remote_norm={remote_norm:.3f}, "
                  f"query_weight={w_val:.3f}, "
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

        logger.info(f"[RPG-Fusion] embed_dim={embed_dim}, sparse_threshold={sparse_threshold}, dropout={dropout}")

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
            logger.info(f"[RPG-Fusion] Step {self.forward_count}: "
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
