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
        分离损失 + 均衡损失

        两部分：
        1. 熵损失：鼓励分数趋向 0 或 1（极化）
        2. 均衡损失：鼓励分数均值接近 0.5（防止单边坍塌）

        为什么需要均衡损失？
        - 如果只有熵损失，所有分数可能都坍塌到 0（或都到 1）
        - 均衡损失确保有些边分数高，有些边分数低
        - 这样模型才能学习区分重要边和不重要边

        Args:
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            loss: 分离损失 + 均衡损失
        """
        eps = 1e-8
        scores = causal_scores.squeeze().clamp(eps, 1 - eps)

        # 1. 熵损失：鼓励极化
        entropy = -(scores * torch.log(scores) +
                    (1 - scores) * torch.log(1 - scores))
        entropy_loss = entropy.mean()

        # 2. 均衡损失：均值应该接近 0.5
        # 这防止所有分数都往 0 或都往 1 坍塌
        mean_score = scores.mean()
        balance_loss = (mean_score - 0.5) ** 2

        # 组合：熵损失 + 均衡损失（均衡损失权重更大）
        # 均衡损失最大值是 0.25（当均值为 0 或 1 时）
        # 熵损失最大值是 log(2) ≈ 0.693（当分数为 0.5 时）
        total = entropy_loss + 10.0 * balance_loss

        return total

    def contrastive_loss(self, original_loss, counterfactual_loss):
        """
        对比损失：原始预测应该比反事实好

        核心原理：
        - 原始预测使用权重 = 0.3 + 0.7 * causal_score（高因果边权重高）
        - 反事实预测使用权重 = 1.0 - 0.7 * causal_score（高因果边权重低）
        - 如果因果分数正确，原始预测应该更好（损失更小）

        使用 Softplus 替代 ReLU：
        - ReLU 的问题：当 orig < cf 时梯度为 0，因果分数无法学习
        - Softplus 始终有非零梯度，即使 orig < cf 也能传递信号
        - loss = log(1 + exp(orig - cf + margin))
        - 当 orig << cf 时，loss ≈ 0（但仍有小梯度）
        - 当 orig >> cf 时，loss ≈ orig - cf + margin

        这样即使模型正确（orig < cf），也有轻微梯度继续优化因果分数

        Args:
            original_loss: 原始预测损失（标量）
            counterfactual_loss: 反事实预测损失（标量）

        Returns:
            loss: 对比损失
        """
        # Softplus: 始终有梯度，避免因果分数学习停滞
        diff = original_loss - counterfactual_loss + self.margin
        loss = F.softplus(diff)
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

        # 详细统计
        scores = causal_scores.squeeze()

        loss_dict = {
            'causal_total': total_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'separation_loss': sep_loss.item(),
            'original_loss': original_loss.item() if isinstance(original_loss, torch.Tensor) else original_loss,
            'cf_loss': counterfactual_loss.item() if isinstance(counterfactual_loss, torch.Tensor) else counterfactual_loss,
            'loss_diff': (original_loss - counterfactual_loss).item() if isinstance(original_loss, torch.Tensor) else 0,
            'warmup': warmup,
            # 因果分数统计
            'cs_mean': scores.mean().item(),
            'cs_std': scores.std().item(),
            'cs_min': scores.min().item(),
            'cs_max': scores.max().item(),
            'cs_median': scores.median().item(),
            # 分布统计
            'cs_q25': scores.quantile(0.25).item(),
            'cs_q75': scores.quantile(0.75).item(),
            'cs_below_0.3': (scores < 0.3).float().mean().item(),
            'cs_above_0.7': (scores > 0.7).float().mean().item(),
            'cs_mid_range': ((scores >= 0.3) & (scores <= 0.7)).float().mean().item(),
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
