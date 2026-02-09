"""
因果损失模块 (Causal Loss Module) - 简化版

核心思想：
- 因果分数直接作为边权重，让预测损失的梯度自然指导学习
- 反事实预测用 (1-因果分数) 作为权重
- 原始预测应该比反事实预测好（因为保留了因果边）
- 分离损失鼓励因果分数趋向 0 或 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLoss(nn.Module):
    """
    简化版因果损失

    核心设计：
    1. 预测损失的梯度自然流向因果分数，指导哪些边重要
    2. 对比损失：原始预测应该比反事实好
    3. 分离损失：鼓励因果分数极化
    """

    def __init__(self, alpha=0.1, beta=0.1, gamma=0.01, warmup_epochs=10):
        """
        Args:
            alpha: 对比损失权重（原始 vs 反事实）
            beta: 未使用，保留兼容性
            gamma: 分离损失权重
            warmup_epochs: warmup 轮数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前 epoch"""
        self.current_epoch = epoch

    def get_warmup_factor(self):
        """获取 warmup 系数"""
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        return self.current_epoch / self.warmup_epochs

    def separation_loss(self, causal_scores):
        """
        分离损失：鼓励因果分数趋向 0 或 1

        使用二元熵，当分数接近 0.5 时熵最大，接近 0 或 1 时熵最小

        Args:
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            loss: 分离损失（熵）
        """
        eps = 1e-8
        scores = causal_scores.squeeze()

        # 二元熵: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -(scores * torch.log(scores + eps) +
                    (1 - scores) * torch.log(1 - scores + eps))

        return entropy.mean()

    def contrastive_loss(self, original_loss, counterfactual_loss):
        """
        对比损失：原始预测应该比反事实好

        如果原始损失 > 反事实损失，说明因果分数学错了
        （因为因果边被保留的原始预测应该更好）

        Args:
            original_loss: 原始预测损失（标量）
            counterfactual_loss: 反事实预测损失（标量）

        Returns:
            loss: 对比损失
        """
        # margin ranking loss: 原始损失应该比反事实损失小
        # loss = max(0, original_loss - counterfactual_loss + margin)
        margin = 0.0  # 可以设置一个正的 margin
        loss = F.relu(original_loss - counterfactual_loss + margin)
        return loss

    def forward(self, causal_scores, original_pred, interventions_data):
        """
        计算因果损失

        Args:
            causal_scores: 因果分数 [num_edges, 1]
            original_pred: 原始预测 [batch, num_ent]（未使用）
            interventions_data: 干预数据列表

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        warmup = self.get_warmup_factor()
        device = causal_scores.device

        # 分离损失
        sep_loss = self.separation_loss(causal_scores)

        # 对比损失（如果有反事实数据）
        contrast_loss = torch.tensor(0.0, device=device)

        # 总损失：只用分离损失
        # 对比损失和对齐损失的效果由预测损失的梯度自然实现
        total_loss = warmup * self.gamma * sep_loss

        loss_dict = {
            'causal_total': total_loss.item(),
            'alignment_loss': 0.0,  # 不再使用
            'consistency_loss': 0.0,  # 不再使用
            'separation_loss': sep_loss.item(),
            'warmup': warmup
        }

        return total_loss, loss_dict


class ContrastiveCausalLoss(nn.Module):
    """
    对比因果损失

    更直接的设计：
    1. 原始预测损失
    2. 反事实预测损失
    3. 约束：原始损失 < 反事实损失
    """

    def __init__(self, gamma=0.01, margin=0.1, warmup_epochs=10):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_warmup_factor(self):
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        return self.current_epoch / self.warmup_epochs

    def forward(self, causal_scores, original_loss, counterfactual_loss):
        """
        Args:
            causal_scores: 因果分数 [num_edges, 1]
            original_loss: 原始预测损失（标量）
            counterfactual_loss: 反事实预测损失（标量）
        """
        warmup = self.get_warmup_factor()
        device = causal_scores.device

        # 分离损失
        eps = 1e-8
        scores = causal_scores.squeeze()
        entropy = -(scores * torch.log(scores + eps) +
                    (1 - scores) * torch.log(1 - scores + eps))
        sep_loss = entropy.mean()

        # 对比损失
        contrast_loss = F.relu(original_loss - counterfactual_loss + self.margin)

        # 总损失
        total_loss = warmup * (self.gamma * sep_loss + 0.1 * contrast_loss)

        loss_dict = {
            'causal_total': total_loss.item(),
            'separation_loss': sep_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'warmup': warmup
        }

        return total_loss, loss_dict
