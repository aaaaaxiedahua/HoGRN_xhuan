"""
因果发现模块 (Causal Discovery Module)

核心思想：
- 评估每条边的因果强度 (causal strength)
- 基于边特征和局部结构信息
- 因果强度高的边对预测有真正的因果影响，而非仅仅是统计相关
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDiscovery(nn.Module):
    """
    因果发现模块

    为每条边计算因果强度分数，用于：
    1. 作为消息传递的权重
    2. 指导干预实验的边选择
    """

    def __init__(self, dim, hidden_dim=None, num_rels=None):
        """
        Args:
            dim: 输入嵌入维度
            hidden_dim: 隐藏层维度
            num_rels: 关系数量（用于关系级别的因果先验）
        """
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.num_rels = num_rels

        # 简化的边级别因果强度评估器
        # 使用 Tanh 而非 ReLU，避免信号被截断
        self.edge_scorer = nn.Sequential(
            nn.Linear(dim * 3 + 2, hidden_dim),  # +2 for degree features
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # Tanh 保持信号流动，输出在 [-1, 1]
            nn.Linear(hidden_dim, 1)
        )

        # 关系级别因果先验（可学习）
        if num_rels is not None:
            self.rel_causal_prior = nn.Parameter(torch.zeros(num_rels * 2))

        self._init_weights()

    def _init_weights(self):
        """初始化权重，使初始因果分数接近 0.5"""
        for i, m in enumerate(self.edge_scorer):
            if isinstance(m, nn.Linear):
                if i == len(self.edge_scorer) - 1:
                    # 最后一层：小权重 + 零偏置，确保输出接近 0
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)
                else:
                    # 中间层：正常初始化
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

    def forward(self, h_src, h_dst, r_emb, edge_type=None, src_deg=None, dst_deg=None):
        """
        计算边的因果强度

        Args:
            h_src: 源节点嵌入 [num_edges, dim]
            h_dst: 目标节点嵌入 [num_edges, dim]
            r_emb: 关系嵌入 [num_edges, dim]
            edge_type: 边类型 [num_edges]（可选，用于关系先验）
            src_deg: 源节点度数 [num_edges]（可选）
            dst_deg: 目标节点度数 [num_edges]（可选）

        Returns:
            causal_score: 因果强度 [num_edges, 1]，范围 [0, 1]
        """
        num_edges = h_src.size(0)
        device = h_src.device

        # 度数特征（归一化）
        if src_deg is not None and dst_deg is not None:
            deg_feat = torch.stack([
                1.0 / (1.0 + src_deg.float()),
                1.0 / (1.0 + dst_deg.float())
            ], dim=-1)  # [num_edges, 2]
        else:
            deg_feat = torch.zeros(num_edges, 2, device=device)

        # 拼接特征
        edge_feat = torch.cat([h_src, r_emb, h_dst, deg_feat], dim=-1)

        # 计算边级别因果分数
        edge_logits = self.edge_scorer(edge_feat)

        # 添加关系级别先验（如果有）
        if edge_type is not None and hasattr(self, 'rel_causal_prior'):
            rel_prior = self.rel_causal_prior[edge_type].unsqueeze(-1)
            edge_logits = edge_logits + rel_prior

        # 返回 logits 和 scores
        # scores 用 sigmoid 映射到 [0,1]，供 edge_weight 使用
        # logits 直接传给 CausalLoss 做均衡损失（绕过 sigmoid 饱和区）
        causal_score = torch.sigmoid(edge_logits)

        return causal_score, edge_logits

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
