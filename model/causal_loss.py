"""
因果稀疏损失模块 (Causal Sparsity Loss)

核心思想：
- 稀疏惩罚：鼓励模型只保留少数真正因果重要的边
- 信息瓶颈原理：迫使模型从最少的边中提取最多的预测信息
- 主预测损失通过 Gumbel-Softmax 直接训练边重要性

两种力量的平衡：
- 主预测损失（通过 Gumbel-Softmax 反传）：推高重要边的分数
- 稀疏惩罚：推低所有边的分数
- 对抗结果：只有真正因果重要的边保持高分数
"""

import torch
import torch.nn as nn


class CausalSparsityLoss(nn.Module):
    """
    因果稀疏损失（目标稀疏率版）

    设计原则：
    - 目标稀疏率惩罚：当 mean(scores) > target 时才产生惩罚
    - 低于 target 时惩罚为 0，避免分数崩溃到 0
    - 主预测损失推高重要边的分数（通过 Gumbel-Softmax）
    - 分数在 target 附近稳定，不同边根据主损失梯度分化出高低
    """

    def __init__(self, lambda_sparse=0.0001, warmup_epochs=30, target_sparsity=0.5):
        """
        Args:
            lambda_sparse: 稀疏惩罚权重
            warmup_epochs: warmup 轮数（稀疏惩罚逐渐加强）
            target_sparsity: 目标保留率，mean(scores) 低于此值时不惩罚
        """
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.warmup_epochs = warmup_epochs
        self.target_sparsity = target_sparsity
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前 epoch"""
        self.current_epoch = epoch

    def get_warmup_factor(self):
        """获取 warmup 系数"""
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        return self.current_epoch / self.warmup_epochs

    def forward(self, causal_scores, logits=None):
        """
        计算稀疏损失

        Args:
            causal_scores: sigmoid(logits) [num_edges, 1]，因果分数
            logits: 原始 logits [num_edges, 1]（用于统计）

        Returns:
            loss: 稀疏损失
            loss_dict: 统计信息
        """
        warmup = self.get_warmup_factor()
        scores = causal_scores.squeeze()

        # 目标稀疏率惩罚：只在 mean > target 时产生惩罚
        excess = (scores.mean() - self.target_sparsity).clamp(min=0)
        sparse_loss = excess ** 2

        total = warmup * self.lambda_sparse * sparse_loss

        # 统计信息
        logits_flat = logits.squeeze() if logits is not None else scores

        loss_dict = {
            'causal_total': total.item(),
            'sparse_loss': sparse_loss.item(),
            'warmup': warmup,
            # 因果分数统计
            'cs_mean': scores.mean().item(),
            'cs_std': scores.std().item(),
            'cs_min': scores.min().item(),
            'cs_max': scores.max().item(),
            'cs_median': scores.median().item(),
            # logits 统计
            'logit_mean': logits_flat.mean().item(),
            'logit_std': logits_flat.std().item(),
            # 分布统计
            'cs_below_0.3': (scores < 0.3).float().mean().item(),
            'cs_above_0.7': (scores > 0.7).float().mean().item(),
            'cs_mid_range': ((scores >= 0.3) & (scores <= 0.7)).float().mean().item(),
        }

        return total, loss_dict
