"""
RPG-HoGRN: Path Score Enhancement Module

核心思想：在原始预测得分上加上路径可达性得分
- 方式 D：查询感知路径打分
- 不同查询关系偏好不同路径
- path_score = Σ relevance(path, query_rel) × reachable(head, tail, path)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PathScoreEnhancer(nn.Module):
    """
    路径得分增强模块：在原始预测得分上加上路径可达性得分

    方式 D：查询感知路径打分
    - 不同查询关系偏好不同路径
    - path_score = Σ relevance(path, query_rel) × reachable(head, tail, path)
    """

    def __init__(self, embed_dim, num_relations, frequent_paths, top_k_paths=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.frequent_paths = frequent_paths
        self.top_k_paths = top_k_paths

        # 路径编码器
        self.rel_embed = nn.Embedding(num_relations * 2, embed_dim)
        self.path_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
            bidirectional=False
        )

        # 查询-路径相关性计算
        self.query_path_scorer = nn.Bilinear(embed_dim, embed_dim, 1)

        # 可学习的缩放因子
        self.beta = nn.Parameter(torch.tensor(0.1))

        # 路径矩阵和关系序列
        self.path_matrices = None  # list of sparse [N, N]
        self.path_matrices_dense = None  # list of dense [N, N] (缓存)
        self.path_relations = None  # list of tensors
        self.num_ent = None

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0

        logger.info(f"[PathScoreEnhancer] embed_dim={embed_dim}, top_k_paths={top_k_paths}")
        logger.info(f"[PathScoreEnhancer] Loaded {len(frequent_paths)} frequent paths")

    def build_matrices(self, edge_index, edge_type, num_ent, device):
        """构建路径矩阵（首次调用时执行）"""
        logger.info(f"[PathScoreEnhancer] Building path matrices...")
        self.num_ent = num_ent

        # 1. 构建每个关系的邻接矩阵
        rel_matrices = {}
        for r in range(self.num_relations * 2):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            r_edge = edge_index[:, mask]
            row, col = r_edge[0], r_edge[1]
            val = torch.ones(row.size(0), device=device)
            indices = torch.stack([row, col], dim=0)
            rel_matrices[r] = torch.sparse_coo_tensor(
                indices, val, (num_ent, num_ent), device=device
            ).coalesce()

        logger.info(f"[PathScoreEnhancer] Built {len(rel_matrices)} relation matrices")

        # 2. 构建路径矩阵
        self.path_matrices = []
        self.path_relations = []

        sorted_paths = sorted(self.frequent_paths.items(), key=lambda x: -x[1])
        valid_count = 0

        for path, freq in sorted_paths[:self.top_k_paths * 10]:
            if len(self.path_matrices) >= self.top_k_paths:
                break

            # 检查路径有效性
            if path[0] not in rel_matrices:
                continue

            # 计算路径矩阵: A_r1 @ A_r2 @ ...
            path_mat = rel_matrices[path[0]]
            valid = True

            for r in path[1:]:
                if r not in rel_matrices:
                    valid = False
                    break
                path_mat = torch.sparse.mm(path_mat, rel_matrices[r])

            if not valid:
                continue

            path_mat = path_mat.coalesce()
            if path_mat.indices().size(1) == 0:
                continue

            # 二值化（只关心可达性，不关心路径数）
            indices = path_mat.indices()
            val = torch.ones(indices.size(1), device=device)
            binary_mat = torch.sparse_coo_tensor(
                indices, val, (num_ent, num_ent), device=device
            ).coalesce()

            self.path_matrices.append(binary_mat)
            path_tensor = torch.tensor(list(path), dtype=torch.long, device=device)
            self.path_relations.append(path_tensor)
            valid_count += 1

        logger.info(f"[PathScoreEnhancer] Built {valid_count} path matrices")
        if valid_count > 0:
            for i, path_rel in enumerate(self.path_relations[:5]):
                logger.info(f"  Path {i}: {path_rel.tolist()}")

    def _compute_path_embeddings(self):
        """计算所有路径的嵌入向量"""
        if not self.path_relations:
            return None

        path_embeds = []
        for path_tensor in self.path_relations:
            rel_embeds = self.rel_embed(path_tensor)  # [path_len, dim]
            rel_embeds = rel_embeds.unsqueeze(0)  # [1, path_len, dim]
            _, h_n = self.path_encoder(rel_embeds)  # [1, 1, dim]
            path_embeds.append(h_n.squeeze(0).squeeze(0))

        return torch.stack(path_embeds, dim=0)  # [num_paths, dim]

    def forward(self, original_score, head_idx, rel_embed, edge_index, edge_type):
        """
        在原始得分上加上路径得分

        Args:
            original_score: [batch, N] 原始预测得分（sigmoid之前）
            head_idx: [batch] 头实体索引
            rel_embed: [batch, dim] 查询关系嵌入
            edge_index: [2, E] 边索引（用于首次构建矩阵）
            edge_type: [E] 边类型

        Returns:
            final_score: [batch, N] 增强后的得分
        """
        batch_size = head_idx.size(0)
        device = head_idx.device

        # 首次调用时构建矩阵
        if self.path_matrices is None:
            num_ent = original_score.size(1)
            self.build_matrices(edge_index, edge_type, num_ent, device)

        # 如果没有有效路径，直接返回原始得分
        if not self.path_matrices:
            return original_score

        num_paths = len(self.path_matrices)

        # 1. 计算路径嵌入
        path_embeds = self._compute_path_embeddings()  # [num_paths, dim]

        # 2. 计算查询-路径相关性
        # rel_embed: [batch, dim], path_embeds: [num_paths, dim]
        # 扩展维度进行批量计算
        rel_exp = rel_embed.unsqueeze(1).expand(-1, num_paths, -1)  # [batch, num_paths, dim]
        path_exp = path_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_paths, dim]

        relevance = self.query_path_scorer(
            rel_exp.reshape(-1, self.embed_dim),
            path_exp.reshape(-1, self.embed_dim)
        ).reshape(batch_size, num_paths)  # [batch, num_paths]

        # softmax 归一化
        path_weights = F.softmax(relevance, dim=1)  # [batch, num_paths]

        # 3. 获取每个头实体的路径可达性得分
        # 首次调用时缓存 dense 矩阵
        if self.path_matrices_dense is None:
            self.path_matrices_dense = [pm.to_dense() for pm in self.path_matrices]
            logger.info(f"[PathScoreEnhancer] Cached {len(self.path_matrices_dense)} dense path matrices")

        # 对每条路径，获取从 head 可达的实体
        path_scores_list = []
        for path_mat_dense in self.path_matrices_dense:
            head_reachable = path_mat_dense[head_idx]  # [batch, N]
            path_scores_list.append(head_reachable)

        # [batch, num_paths, N]
        path_scores = torch.stack(path_scores_list, dim=1)

        # 4. 加权求和
        # path_weights: [batch, num_paths] -> [batch, num_paths, 1]
        # path_scores: [batch, num_paths, N]
        weighted_path_score = (path_weights.unsqueeze(-1) * path_scores).sum(dim=1)  # [batch, N]

        # 5. 融合
        beta = torch.sigmoid(self.beta)  # 限制在 0-1
        final_score = original_score + beta * weighted_path_score

        # Log statistics
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            avg_weight = path_weights.mean(dim=0)  # [num_paths]
            avg_path_score = weighted_path_score.mean().item()
            beta_val = beta.item()
            logger.info(f"[PathScoreEnhancer] Step {self.forward_count}: "
                       f"beta={beta_val:.4f}, "
                       f"avg_path_score={avg_path_score:.4f}, "
                       f"path_weights={avg_weight.tolist()[:3]}")

        return final_score
