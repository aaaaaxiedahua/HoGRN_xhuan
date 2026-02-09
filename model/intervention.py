"""
干预模块 (Intervention Module)

核心思想：
- 模拟因果推断中的 do 操作
- 通过干预实验计算边的因果效应
- 干预效应大的边是真正的因果边
"""

import torch
import torch.nn as nn
import numpy as np


class InterventionModule(nn.Module):
    """
    干预模块

    实现两种干预策略：
    1. 边删除干预：移除某些边，观察预测变化
    2. 边扰动干预：用噪声替换边的消息
    """

    def __init__(self, num_interventions=3, intervention_ratio=0.1):
        """
        Args:
            num_interventions: 每次训练的干预采样次数
            intervention_ratio: 每次干预移除的边比例
        """
        super().__init__()
        self.num_interventions = num_interventions
        self.intervention_ratio = intervention_ratio

    def sample_intervention_mask(self, num_edges, causal_scores=None, strategy='random'):
        """
        采样干预掩码

        Args:
            num_edges: 边数量
            causal_scores: 因果分数 [num_edges, 1]（可选）
            strategy: 采样策略
                - 'random': 随机采样
                - 'low_causal': 优先干预低因果分数的边
                - 'high_causal': 优先干预高因果分数的边
                - 'mixed': 混合策略

        Returns:
            mask: 干预掩码 [num_edges]，True 表示保留，False 表示移除
        """
        device = causal_scores.device if causal_scores is not None else 'cpu'
        num_intervene = max(1, int(num_edges * self.intervention_ratio))

        if strategy == 'random' or causal_scores is None:
            # 随机选择要干预的边
            indices = torch.randperm(num_edges, device=device)[:num_intervene]

        elif strategy == 'low_causal':
            # 优先干预低因果分数的边（验证它们确实不重要）
            scores = causal_scores.squeeze()
            _, indices = torch.topk(scores, num_intervene, largest=False)

        elif strategy == 'high_causal':
            # 干预高因果分数的边（验证它们确实重要）
            scores = causal_scores.squeeze()
            _, indices = torch.topk(scores, num_intervene, largest=True)

        elif strategy == 'mixed':
            # 混合：一半低分数，一半高分数
            scores = causal_scores.squeeze()
            half = num_intervene // 2
            _, low_indices = torch.topk(scores, half, largest=False)
            _, high_indices = torch.topk(scores, num_intervene - half, largest=True)
            indices = torch.cat([low_indices, high_indices])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 创建掩码
        mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        mask[indices] = False

        return mask

    def apply_intervention(self, edge_index, edge_type, mask):
        """
        应用干预（边删除）

        Args:
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            mask: 干预掩码 [num_edges]

        Returns:
            intervened_edge_index: 干预后的边索引
            intervened_edge_type: 干预后的边类型
        """
        intervened_edge_index = edge_index[:, mask]
        intervened_edge_type = edge_type[mask]
        return intervened_edge_index, intervened_edge_type

    def compute_intervention_effect(self, original_scores, intervened_scores):
        """
        计算干预效应

        干预效应 = |original_score - intervened_score|
        效应大说明被干预的边对预测有因果影响

        Args:
            original_scores: 原始预测分数 [batch, num_ent]
            intervened_scores: 干预后预测分数 [batch, num_ent]

        Returns:
            effect: 干预效应 [batch]
        """
        # L1 距离作为效应度量
        effect = (original_scores - intervened_scores).abs().mean(dim=-1)
        return effect

    def forward(self, edge_index, edge_type, causal_scores, strategy='mixed'):
        """
        执行一次干预采样

        Args:
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            causal_scores: 因果分数 [num_edges, 1]
            strategy: 干预策略

        Returns:
            interventions: 干预列表，每个元素包含：
                - mask: 干预掩码
                - edge_index: 干预后边索引
                - edge_type: 干预后边类型
                - intervened_indices: 被干预的边索引
        """
        num_edges = edge_index.size(1)
        interventions = []

        # 使用不同策略采样多次干预
        strategies = ['low_causal', 'high_causal', 'random']

        for i in range(self.num_interventions):
            # 轮换策略
            current_strategy = strategies[i % len(strategies)]

            mask = self.sample_intervention_mask(
                num_edges, causal_scores, strategy=current_strategy
            )

            int_edge_index, int_edge_type = self.apply_intervention(
                edge_index, edge_type, mask
            )

            interventions.append({
                'mask': mask,
                'edge_index': int_edge_index,
                'edge_type': int_edge_type,
                'intervened_indices': (~mask).nonzero(as_tuple=True)[0],
                'strategy': current_strategy
            })

        return interventions


class SoftIntervention(nn.Module):
    """
    软干预模块

    不是完全移除边，而是用因果分数作为权重
    这样可以实现可微分的干预
    """

    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: 温度参数，控制软干预的"硬度"
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, causal_scores, hard=False):
        """
        计算软干预权重

        Args:
            causal_scores: 因果分数 [num_edges, 1]
            hard: 是否使用硬干预（Gumbel-Softmax）

        Returns:
            weights: 干预权重 [num_edges, 1]
        """
        if hard:
            # 硬干预：使用 straight-through estimator
            hard_weights = (causal_scores > 0.5).float()
            # 直通估计：前向用硬值，反向用软值的梯度
            weights = hard_weights - causal_scores.detach() + causal_scores
        else:
            # 软干预：直接用因果分数作为权重
            weights = causal_scores

        return weights

    def counterfactual_weight(self, causal_scores):
        """
        计算反事实权重

        反事实：如果因果边不存在会怎样？
        权重 = 1 - causal_score（因果边被削弱）

        Args:
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            cf_weights: 反事实权重 [num_edges, 1]
        """
        return 1.0 - causal_scores
