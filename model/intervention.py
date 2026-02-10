"""
边级别随机干预模块 (Edge-level Stochastic Intervention)

核心思想：
- 使用 Gumbel-Softmax / Concrete Relaxation 对每条边做可微的随机干预
- 训练时：每条边随机决定"保留"或"丢弃"，梯度通过采样反传
- 推理时：使用确定性权重（sigmoid）
- 主预测损失直接训练每条边的重要性，稀疏惩罚发现最小因果子图

参考文献：
- Concrete Relaxation: Maddison et al. (2017), Jang et al. (2017)
- L0 Regularization: Louizos et al. (2018)
"""

import torch
import torch.nn as nn


class GumbelIntervention(nn.Module):
    """
    Gumbel-Softmax 边干预模块

    使用 Concrete / Gumbel-Softmax 松弛实现可微的 Bernoulli 采样：
    - 训练时对每条边独立采样 keep/drop
    - 主预测损失直接反传到每条边的 logit
    - Temperature 退火：从软采样逐渐到硬采样

    edge_weight = base_weight + scale * z
    - z ∈ (0, 1) 来自 Gumbel-Softmax 采样
    - edge_weight ∈ [base_weight, base_weight + scale]
    """

    def __init__(self, base_weight=0.3, scale=0.7,
                 init_temperature=1.0, min_temperature=0.1):
        """
        Args:
            base_weight: 基础权重（残差连接），保证最小信息流
            scale: 缩放因子
            init_temperature: 初始温度（高温 = 软采样）
            min_temperature: 最低温度（低温 = 硬采样）
        """
        super().__init__()
        self.base_weight = base_weight
        self.scale = scale
        self.temperature = init_temperature
        self.min_temperature = min_temperature

    def set_temperature(self, temperature):
        """设置当前温度"""
        self.temperature = max(self.min_temperature, temperature)

    def forward(self, logits):
        """
        对每条边做可微的随机干预

        Args:
            logits: 因果 logits [num_edges, 1]

        Returns:
            edge_weight: 边权重 [num_edges, 1]
            z: 采样值 [num_edges, 1]，∈ (0, 1)
        """
        if self.training:
            # Concrete / Gumbel-Softmax relaxation for Bernoulli
            # z = sigmoid((logit + logistic_noise) / temperature)
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel_noise = torch.log(u) - torch.log(1 - u)  # Logistic noise
            z = torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            # 推理时：确定性权重
            z = torch.sigmoid(logits)

        edge_weight = self.base_weight + self.scale * z
        return edge_weight, z
