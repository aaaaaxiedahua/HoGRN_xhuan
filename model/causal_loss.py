"""
因果损失模块 (Causal Loss Module) - 对比学习版

核心思想：
- 残差连接保证最小信息流：edge_weight = 0.3 + 0.7 * causal_score
- 对比损失：原始预测应该比反事实预测好
- 分离损失（辅助）：轻微鼓励因果分数极化
- 两种力量平衡：对比损失推高重要边分数，分离损失推向极端
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLoss(nn.Module):
    """
    对比因果损失

    核心设计：
    1. 对比损失：原始损失应该小于反事实损失（重要边被保留时预测更好）
    2. 分离损失（辅助）：轻微鼓励极化，避免所有分数都在 0.5
    3. Warmup：让模型先学会基本预测，再学因果结构
    """

    def __init__(self, alpha=0.5, beta=0.1, gamma=0.0001, margin=0.1, warmup_epochs=10):
        """
        Args:
            alpha: 对比损失权重（核心损失）
            beta: 未使用，保留兼容性
            gamma: 分离损失权重（应该很小，只起辅助作用）
            margin: 对比损失的 margin（原始损失应该比反事实损失小 margin）
            warmup_epochs: warmup 轮数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前 epoch"""
        self.current_epoch = epoch

    def get_warmup_factor(self):
        """获取 warmup 系数"""
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        # 使用更平滑的 warmup 曲线
        progress = self.current_epoch / self.warmup_epochs
        return progress * progress  # 平方曲线，开始慢后面快

    def separation_loss(self, causal_scores):
        """
        分离损失：轻微鼓励因果分数趋向 0 或 1

        使用二元熵，但权重应该很小（gamma=0.0001）
        主要靠对比损失来区分边的重要性

        Args:
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            loss: 分离损失（熵）
        """
        eps = 1e-8
        scores = causal_scores.squeeze().clamp(eps, 1 - eps)

        # 二元熵: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -(scores * torch.log(scores) +
                    (1 - scores) * torch.log(1 - scores))

        return entropy.mean()

    def contrastive_loss(self, original_loss, counterfactual_loss):
        """
        对比损失：原始预测应该比反事实好

        核心原理：
        - 原始预测使用权重 = 0.3 + 0.7 * causal_score（高因果边权重高）
        - 反事实预测使用权重 = 1.0 - 0.7 * causal_score（高因果边权重低）
        - 如果因果分数正确，原始预测应该更好（损失更小）
        - 如果原始损失 > 反事实损失，说明因果分数学反了

        Margin Ranking Loss:
        loss = max(0, original_loss - counterfactual_loss + margin)

        当原始损失比反事实损失大时产生正损失，推动模型调整因果分数

        Args:
            original_loss: 原始预测损失（标量）
            counterfactual_loss: 反事实预测损失（标量）

        Returns:
            loss: 对比损失
        """
        loss = F.relu(original_loss - counterfactual_loss + self.margin)
        return loss

    def forward(self, causal_scores, original_loss, counterfactual_loss):
        """
        计算因果损失

        新接口：直接接收原始损失和反事实损失（标量）

        Args:
            causal_scores: 因果分数 [num_edges, 1]
            original_loss: 原始预测损失（标量）
            counterfactual_loss: 反事实预测损失（标量）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        warmup = self.get_warmup_factor()
        device = causal_scores.device

        # 分离损失（辅助）
        sep_loss = self.separation_loss(causal_scores)

        # 对比损失（核心）
        contrast_loss = self.contrastive_loss(original_loss, counterfactual_loss)

        # 总损失
        # 对比损失是核心，分离损失只是辅助
        total_loss = warmup * (self.alpha * contrast_loss + self.gamma * sep_loss)

        loss_dict = {
            'causal_total': total_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'separation_loss': sep_loss.item(),
            'original_loss': original_loss.item() if isinstance(original_loss, torch.Tensor) else original_loss,
            'cf_loss': counterfactual_loss.item() if isinstance(counterfactual_loss, torch.Tensor) else counterfactual_loss,
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
