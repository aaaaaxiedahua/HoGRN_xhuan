"""
RPG-HoGRN: Reverse Path Reasoning Module (逆向路径推理)

核心思想：学习每个关系的"答案模式"，推理时检查候选实体是否符合该模式
- 离线统计：对每个关系 r，统计其答案实体的入边/出边模式频率
- 在线推理：对所有候选实体计算模式匹配得分，加到原始得分上
- 不依赖 head→tail 的路径连通性，只看 tail 自身的局部结构
"""

import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReversePathReasoner(nn.Module):
    """
    逆向路径推理模块

    离线阶段：
      1. 收集每个关系的答案实体
      2. 统计答案实体的入边/出边模式频率
      3. 构建实体-模式矩阵 M[N, P] (连续值, log(1+count))
      4. 构建关系-模式权重矩阵 W[R, P] (频率作为初始值)

    在线阶段：
      score = M @ W[query_rel]  →  [batch, N]
      final = original_score + gamma * normalize(score)
    """

    def __init__(self, num_relations, num_ent):
        super().__init__()
        self.num_relations = num_relations
        self.num_ent = num_ent

        # 模式总数 = 2 * num_relations * 2 (入边/出边 × 含逆关系)
        # 入边模式: (in, r) for r in 0..2*num_rel-1
        # 出边模式: (out, r) for r in 0..2*num_rel-1
        self.num_patterns = 2 * num_relations * 2  # in/out × all relations (含逆)

        # 可学习的关系-模式权重 (初始化后会用频率覆盖)
        # W[r, p]: 关系 r 对模式 p 的权重
        self.pattern_weight = nn.Parameter(torch.zeros(num_relations * 2, self.num_patterns))

        # 可学习的融合系数
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # 实体-模式矩阵 (buffer, 不参与梯度)
        self.register_buffer('entity_pattern_matrix', torch.zeros(num_ent, self.num_patterns))

        # 是否已构建
        self.built = False

        # Log
        self.forward_count = 0
        self.log_interval = 100

        logger.info(f"[ReversePathReasoner] num_relations={num_relations}, "
                    f"num_ent={num_ent}, num_patterns={self.num_patterns}")

    def _pattern_index(self, direction, rel_id):
        """将 (direction, rel_id) 映射为模式索引"""
        # direction: 0=in, 1=out
        # rel_id: 0 ~ 2*num_rel-1
        return direction * (self.num_relations * 2) + rel_id

    def build(self, triples, edge_index, edge_type, device):
        """
        离线构建实体-模式矩阵和关系-模式权重

        Args:
            triples: list of (h, r, t) 训练三元组
            edge_index: [2, E] 边索引
            edge_type: [E] 边类型
            device: torch device
        """
        logger.info("[ReversePathReasoner] Building entity-pattern matrix and relation-pattern weights...")

        num_rel_total = self.num_relations * 2  # 含逆关系

        # ===== Step 1: 构建入边/出边索引 =====
        # 使用 edge_index 和 edge_type (包含逆关系)
        edge_src = edge_index[0].cpu().numpy()
        edge_dst = edge_index[1].cpu().numpy()
        edge_tp = edge_type.cpu().numpy()

        # in_edges[entity] = [(rel, src), ...]
        in_edges = defaultdict(list)
        # out_edges[entity] = [(rel, dst), ...]
        out_edges = defaultdict(list)

        for i in range(len(edge_tp)):
            src, dst, r = int(edge_src[i]), int(edge_dst[i]), int(edge_tp[i])
            out_edges[src].append((r, dst))
            in_edges[dst].append((r, src))

        logger.info(f"[ReversePathReasoner] Built adjacency: "
                    f"{len(out_edges)} entities with out-edges, "
                    f"{len(in_edges)} entities with in-edges")

        # ===== Step 2: 构建实体-模式矩阵 M[N, P] =====
        # M[e][p] = log(1 + count(entity e has pattern p))
        entity_pattern = torch.zeros(self.num_ent, self.num_patterns)

        # 入边模式计数
        for entity, edges in in_edges.items():
            rel_count = defaultdict(int)
            for r, _ in edges:
                rel_count[r] += 1
            for r, count in rel_count.items():
                p_idx = self._pattern_index(0, r)  # in direction
                entity_pattern[entity, p_idx] = torch.log1p(torch.tensor(float(count)))

        # 出边模式计数
        for entity, edges in out_edges.items():
            rel_count = defaultdict(int)
            for r, _ in edges:
                rel_count[r] += 1
            for r, count in rel_count.items():
                p_idx = self._pattern_index(1, r)  # out direction
                entity_pattern[entity, p_idx] = torch.log1p(torch.tensor(float(count)))

        self.entity_pattern_matrix.copy_(entity_pattern.to(device))

        # 统计非零模式
        nonzero_count = (entity_pattern > 0).sum().item()
        total_count = entity_pattern.numel()
        logger.info(f"[ReversePathReasoner] Entity-pattern matrix: "
                    f"shape={list(entity_pattern.shape)}, "
                    f"nonzero={nonzero_count}/{total_count} "
                    f"({100*nonzero_count/total_count:.2f}%)")

        # ===== Step 3: 收集每个关系的答案实体，统计模式频率 =====
        answer_sets = defaultdict(set)

        # 使用 edge_index (已含逆关系)
        for i in range(len(edge_tp)):
            r = int(edge_tp[i])
            t = int(edge_dst[i])
            answer_sets[r].add(t)

        logger.info(f"[ReversePathReasoner] Collected answer sets for {len(answer_sets)} relations")

        # 计算频率: freq[r][p] = 拥有模式p的答案实体数 / 答案实体总数
        freq_matrix = torch.zeros(num_rel_total, self.num_patterns)

        for r, answers in answer_sets.items():
            if len(answers) == 0:
                continue
            n_answers = len(answers)

            for p_idx in range(self.num_patterns):
                count = 0
                for ans in answers:
                    if entity_pattern[ans, p_idx] > 0:
                        count += 1
                freq_matrix[r, p_idx] = count / n_answers

        # 用频率初始化可学习权重
        with torch.no_grad():
            self.pattern_weight.copy_(freq_matrix.to(device))

        # 统计每个关系有多少非零模式
        rel_nonzero = (freq_matrix > 0).sum(dim=1)
        avg_patterns = rel_nonzero.float().mean().item()
        logger.info(f"[ReversePathReasoner] Avg patterns per relation: {avg_patterns:.1f}")

        # 输出前5个关系的Top模式
        for r in range(min(5, num_rel_total)):
            weights = freq_matrix[r]
            top_vals, top_idxs = torch.topk(weights, min(3, (weights > 0).sum().item()))
            if top_vals.numel() > 0:
                patterns_str = []
                for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
                    direction = "in" if idx < num_rel_total else "out"
                    rel_id = idx % num_rel_total
                    patterns_str.append(f"({direction},r{rel_id}):{val:.2f}")
                logger.info(f"  Rel {r} top patterns: {', '.join(patterns_str)}")

        self.built = True
        logger.info("[ReversePathReasoner] Build complete!")

    def forward(self, original_score, query_rel, edge_index, edge_type):
        """
        在原始得分上加上逆向路径推理得分

        Args:
            original_score: [batch, N] 原始预测得分（sigmoid之前）
            query_rel: [batch] 查询关系索引
            edge_index: [2, E] 边索引（用于首次构建）
            edge_type: [E] 边类型

        Returns:
            final_score: [batch, N] 增强后的得分
        """
        device = original_score.device

        # 首次调用时构建
        if not self.built:
            # 从 edge_index/edge_type 构建, triples 不需要
            self.build(triples=None, edge_index=edge_index, edge_type=edge_type, device=device)

        # 1. 获取查询关系的模式权重 (ReLU保证非负)
        rel_weights = F.relu(self.pattern_weight[query_rel])  # [batch, num_patterns]

        # 2. 计算模式匹配得分
        # entity_pattern_matrix: [N, num_patterns] (非负, log1p值)
        # rel_weights: [batch, num_patterns] (非负, ReLU后)
        # score = rel_weights @ M.T → [batch, N] (非负)
        pattern_score = torch.mm(rel_weights, self.entity_pattern_matrix.t())  # [batch, N]

        # 3. 归一化 (per-sample, min-max to [0,1])
        score_min = pattern_score.min(dim=1, keepdim=True)[0]
        score_max = pattern_score.max(dim=1, keepdim=True)[0]
        pattern_score_norm = (pattern_score - score_min) / (score_max - score_min + 1e-8)

        # 4. 融合
        gamma = torch.sigmoid(self.gamma)
        final_score = original_score + gamma * pattern_score_norm

        # Log statistics
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            avg_pattern_score = pattern_score_norm.mean().item()
            gamma_val = gamma.item()
            weight_min = self.pattern_weight.min().item()
            weight_max = self.pattern_weight.max().item()
            logger.info(f"[ReversePathReasoner] Step {self.forward_count}: "
                        f"gamma={gamma_val:.4f}, "
                        f"avg_norm_score={avg_pattern_score:.4f}, "
                        f"weight_range=[{weight_min:.3f}, {weight_max:.3f}]")

        return final_score
