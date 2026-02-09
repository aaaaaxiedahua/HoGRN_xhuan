"""
因果损失模块 (Causal Loss Module)

核心思想：
- 通过干预效应差异指导因果分数学习
- 高因果分数的边被干预后，预测应该变化大
- 低因果分数的边被干预后，预测应该变化小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLoss(nn.Module):
    """
    因果损失函数

    包含三个部分：
    1. 因果对齐损失：让因果分数与干预效应一致
    2. 干预一致性损失：非因果边的干预不应影响预测
    3. 因果分离损失：鼓励因果分数分化（趋向 0 或 1）
    """

    def __init__(self, alpha=0.1, beta=0.1, gamma=0.01, warmup_epochs=10):
        """
        Args:
            alpha: 因果对齐损失权重
            beta: 干预一致性损失权重
            gamma: 因果分离损失权重
            warmup_epochs: warmup 轮数
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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

    def causal_alignment_loss(self, causal_scores, intervention_effects, intervened_indices):
        """
        因果对齐损失

        核心思想：
        - 被干预边的因果分数应该与干预效应正相关
        - 高因果分数 + 大干预效应 → 正确
        - 低因果分数 + 小干预效应 → 正确
        - 高因果分数 + 小干预效应 → 惩罚
        - 低因果分数 + 大干预效应 → 惩罚

        Args:
            causal_scores: 因果分数 [num_edges, 1]
            intervention_effects: 干预效应 [batch]
            intervened_indices: 被干预的边索引

        Returns:
            loss: 对齐损失
        """
        if len(intervened_indices) == 0:
            return torch.tensor(0.0, device=causal_scores.device)

        # 获取被干预边的因果分数
        intervened_scores = causal_scores[intervened_indices].squeeze()  # [num_intervened]

        # 干预效应（标量，整个 batch 的平均效应）
        effect = intervention_effects.mean()

        # 平均因果分数
        avg_causal = intervened_scores.mean()

        # 对齐损失：因果分数高的边被干预应该产生大效应
        # 使用负相关作为损失（我们希望正相关）
        # 如果 avg_causal 高但 effect 低，说明因果分数不准确
        alignment_loss = -torch.log(effect + 1e-8) * avg_causal + torch.log(effect + 1e-8) * (1 - avg_causal)

        return alignment_loss

    def intervention_consistency_loss(self, original_pred, intervened_pred,
                                       causal_scores, intervened_indices):
        """
        干预一致性损失

        核心思想：
        - 干预低因果分数的边后，预测应该稳定
        - 干预高因果分数的边后，预测可以变化

        Args:
            original_pred: 原始预测 [batch, num_ent]
            intervened_pred: 干预后预测 [batch, num_ent]
            causal_scores: 因果分数 [num_edges, 1]
            intervened_indices: 被干预的边索引

        Returns:
            loss: 一致性损失
        """
        if len(intervened_indices) == 0:
            return torch.tensor(0.0, device=causal_scores.device)

        # 被干预边的平均因果分数
        intervened_causal = causal_scores[intervened_indices].mean()

        # 预测变化
        pred_change = F.mse_loss(original_pred, intervened_pred)

        # 一致性损失：
        # 如果干预的是低因果边（intervened_causal 低），pred_change 应该小
        # 如果干预的是高因果边（intervened_causal 高），pred_change 可以大
        # 损失 = (1 - causal_score) * pred_change
        # 低因果分数边被干预产生大变化 → 大损失
        consistency_loss = (1.0 - intervened_causal) * pred_change

        return consistency_loss

    def causal_separation_loss(self, causal_scores):
        """
        因果分离损失

        鼓励因果分数趋向 0 或 1（明确的因果/非因果判断）
        使用熵损失

        Args:
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            loss: 分离损失
        """
        eps = 1e-8
        scores = causal_scores.squeeze()

        # 二元熵
        entropy = -(scores * torch.log(scores + eps) +
                    (1 - scores) * torch.log(1 - scores + eps))

        return entropy.mean()

    def forward(self, causal_scores, original_pred, interventions_data):
        """
        计算总的因果损失

        Args:
            causal_scores: 因果分数 [num_edges, 1]
            original_pred: 原始预测 [batch, num_ent]
            interventions_data: 干预数据列表，每个元素包含：
                - intervened_pred: 干预后预测
                - intervened_indices: 被干预的边索引
                - strategy: 干预策略

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        warmup = self.get_warmup_factor()
        device = causal_scores.device

        # 初始化损失
        alignment_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        # 对每次干预计算损失
        num_interventions = len(interventions_data)
        for int_data in interventions_data:
            intervened_pred = int_data['intervened_pred']
            intervened_indices = int_data['intervened_indices']
            strategy = int_data.get('strategy', 'random')

            # 计算干预效应
            effect = (original_pred - intervened_pred).abs().mean(dim=-1)

            # 因果对齐损失
            align_loss = self.causal_alignment_loss(
                causal_scores, effect, intervened_indices
            )
            alignment_loss = alignment_loss + align_loss

            # 干预一致性损失
            consist_loss = self.intervention_consistency_loss(
                original_pred, intervened_pred, causal_scores, intervened_indices
            )
            consistency_loss = consistency_loss + consist_loss

        # 平均
        if num_interventions > 0:
            alignment_loss = alignment_loss / num_interventions
            consistency_loss = consistency_loss / num_interventions

        # 因果分离损失
        separation_loss = self.causal_separation_loss(causal_scores)

        # 总损失
        total_loss = warmup * (
            self.alpha * alignment_loss +
            self.beta * consistency_loss +
            self.gamma * separation_loss
        )

        loss_dict = {
            'causal_total': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'separation_loss': separation_loss.item(),
            'warmup': warmup
        }

        return total_loss, loss_dict


class SimpleCausalLoss(nn.Module):
    """
    简化版因果损失

    核心思想更直接：
    - 用因果分数作为边权重
    - 反事实：用 (1 - 因果分数) 作为权重
    - 损失：原始预测应该比反事实预测更好
    """

    def __init__(self, margin=0.1, warmup_epochs=10):
        """
        Args:
            margin: 边界值
            warmup_epochs: warmup 轮数
        """
        super().__init__()
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_warmup_factor(self):
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        return self.current_epoch / self.warmup_epochs

    def forward(self, original_loss, counterfactual_loss, causal_scores):
        """
        计算简化因果损失

        Args:
            original_loss: 原始预测损失
            counterfactual_loss: 反事实预测损失
            causal_scores: 因果分数 [num_edges, 1]

        Returns:
            causal_loss: 因果损失
            loss_dict: 损失详情
        """
        warmup = self.get_warmup_factor()

        # 因果分离损失（熵）
        eps = 1e-8
        scores = causal_scores.squeeze()
        entropy = -(scores * torch.log(scores + eps) +
                    (1 - scores) * torch.log(1 - scores + eps))
        separation_loss = entropy.mean()

        # 对比损失：原始预测应该比反事实好
        # 如果反事实损失 < 原始损失，说明因果分数没学好
        contrastive_loss = F.relu(original_loss - counterfactual_loss + self.margin)

        # 总损失
        total_loss = warmup * (0.1 * separation_loss + 0.1 * contrastive_loss)

        loss_dict = {
            'causal_total': total_loss.item(),
            'separation_loss': separation_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'warmup': warmup
        }

        return total_loss, loss_dict
