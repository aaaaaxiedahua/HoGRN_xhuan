"""
边选择器模块 (Edge Selector)

核心思想：
- 为每条边计算重要性概率 prob ∈ [0, 1]
- prob 直接作为消息传递的权重
- 通过分化损失让 prob 趋向 0 或 1，实现软过滤
"""

import torch
import torch.nn as nn


class EdgeSelector(nn.Module):
    """
    边重要性评估器

    输入: 头实体嵌入、关系嵌入、尾实体嵌入
    输出: 边重要性概率 (用作消息传递的权重)
    """

    def __init__(self, dim, hidden_dim=None):
        """
        Args:
            dim: 输入嵌入维度
            hidden_dim: 隐藏层维度，默认等于 dim
        """
        super().__init__()
        hidden_dim = hidden_dim or dim

        # 边重要性评分网络
        # 输入: [h_src || r_edge || h_dst]，维度为 3*dim
        self.scorer = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 额外的关系级别重要性（可学习的先验）
        # 某些关系可能整体更重要或更不重要
        self.use_rel_prior = True

        # 初始化：让初始输出接近 0.5
        self._init_weights()

    def _init_weights(self):
        """初始化权重，使初始 prob 接近 0.5"""
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                # 使用较小的初始化，让初始输出接近 0（sigmoid(0) = 0.5）
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_src, h_dst, r_emb):
        """
        计算边重要性概率

        Args:
            h_src: 源节点(头实体)嵌入 [num_edges, dim]
            h_dst: 目标节点(尾实体)嵌入 [num_edges, dim]
            r_emb: 边(关系)嵌入 [num_edges, dim]

        Returns:
            prob: 边重要性概率 [num_edges, 1]
        """
        # 拼接边特征
        edge_feat = torch.cat([h_src, r_emb, h_dst], dim=-1)

        # 计算重要性分数
        logits = self.scorer(edge_feat)

        # Sigmoid 映射到 [0, 1]
        prob = torch.sigmoid(logits)

        return prob

    def forward_with_structure(self, h_src, h_dst, r_emb, src_deg, dst_deg):
        """
        结合结构信息计算边重要性

        低度节点的边可能更重要（稀疏节点需要保留更多信息）

        Args:
            h_src, h_dst, r_emb: 同 forward
            src_deg: 源节点度数 [num_edges]
            dst_deg: 目标节点度数 [num_edges]

        Returns:
            prob: 边重要性概率 [num_edges, 1]
        """
        # 基础概率
        prob = self.forward(h_src, h_dst, r_emb)

        # 度数调整：低度节点的边权重稍微提升
        # 归一化度数到 [0, 1]
        deg_factor = 1.0 / (1.0 + 0.1 * (src_deg + dst_deg).unsqueeze(-1).float())

        # 加权组合
        prob = prob * (1.0 + 0.2 * deg_factor)
        prob = torch.clamp(prob, 0.0, 1.0)

        return prob

    def polarization_loss(self, prob):
        """
        分化损失：鼓励 prob 趋向 0 或 1，同时保持稀疏性

        包含两部分：
        1. 熵损失：让分布确定（趋向 0 或 1）
        2. 稀疏损失：鼓励更多边被过滤（趋向 0）

        Args:
            prob: 边重要性概率 [num_edges, 1]

        Returns:
            loss: 标量损失值
        """
        eps = 1e-8

        # 1. 熵损失：鼓励极端化
        entropy = -(prob * torch.log(prob + eps) +
                    (1 - prob) * torch.log(1 - prob + eps))
        entropy_loss = entropy.mean()

        # 2. 稀疏损失：鼓励 prob 趋向 0（过滤更多边）
        # 但不能太强，否则所有边都被过滤
        sparse_loss = prob.mean()

        # 组合：熵损失 + 0.5 * 稀疏损失
        # 稀疏损失权重较小，只起到"打破对称"的作用
        return entropy_loss + 0.5 * sparse_loss

    def get_stats(self, prob):
        """
        获取边选择的统计信息（用于日志记录）

        Args:
            prob: 边重要性概率

        Returns:
            dict: 统计信息
        """
        with torch.no_grad():
            prob_flat = prob.squeeze()
            stats = {
                'prob_mean': prob_flat.mean().item(),
                'prob_std': prob_flat.std().item(),
                'prob_min': prob_flat.min().item(),
                'prob_max': prob_flat.max().item(),
                'high_prob_ratio': (prob_flat > 0.5).float().mean().item(),  # prob>0.5 的比例
                'very_high_ratio': (prob_flat > 0.9).float().mean().item(),  # prob>0.9 的比例
                'very_low_ratio': (prob_flat < 0.1).float().mean().item(),   # prob<0.1 的比例
            }
        return stats
