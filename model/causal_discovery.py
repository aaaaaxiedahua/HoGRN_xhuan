"""
因果发现模块 (Causal Discovery Module)

核心思想：
- 每条边一个直接可学习的 logit 参数
- 梯度从主损失通过 GCN → edge_weight → Gumbel → logit 直接回传
- 无需 MLP 中间层，梯度路径最短
- 关系级别先验为每条边提供初始偏置
"""

import torch
import torch.nn as nn


class CausalDiscovery(nn.Module):
    """
    因果发现模块（直接可学习 logits 版）

    每条边拥有独立的可学习 logit 参数，
    加上关系级别先验偏置。
    """

    def __init__(self, num_edges, num_rels=None):
        """
        Args:
            num_edges: 边的数量
            num_rels: 关系数量（用于关系级别的因果先验）
        """
        super().__init__()
        self.num_edges = num_edges

        # 直接可学习的 per-edge logits，初始化为 0 → sigmoid(0) = 0.5
        self.edge_logits = nn.Parameter(torch.zeros(num_edges))

        # 关系级别因果先验（可学习）
        if num_rels is not None:
            self.rel_causal_prior = nn.Parameter(torch.zeros(num_rels * 2))

    def forward(self, edge_type=None):
        """
        计算边的因果强度

        Args:
            edge_type: 边类型 [num_edges]（可选，用于关系先验）

        Returns:
            causal_score: 因果强度 [num_edges, 1]，范围 [0, 1]
            edge_logits: 原始 logits [num_edges, 1]
        """
        logits = self.edge_logits

        # 添加关系级别先验
        if edge_type is not None and hasattr(self, 'rel_causal_prior'):
            rel_prior = self.rel_causal_prior[edge_type]
            logits = logits + rel_prior

        causal_score = torch.sigmoid(logits)

        return causal_score.unsqueeze(-1), logits.unsqueeze(-1)

    def get_stats(self, causal_score):
        """获取因果分数统计信息"""
        with torch.no_grad():
            score_flat = causal_score.squeeze()
            stats = {
                'causal_mean': score_flat.mean().item(),
                'causal_std': score_flat.std().item(),
                'causal_min': score_flat.min().item(),
                'causal_max': score_flat.max().item(),
                'high_causal_ratio': (score_flat > 0.5).float().mean().item(),
                'very_high_ratio': (score_flat > 0.8).float().mean().item(),
                'very_low_ratio': (score_flat < 0.2).float().mean().item(),
            }
        return stats
