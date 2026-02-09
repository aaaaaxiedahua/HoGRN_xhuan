"""
信息瓶颈损失模块 (Information Bottleneck Loss)

核心思想：
- 学习压缩的表示，只保留预测必需的信息
- L = L_pred + β * L_KL + γ * L_polar
"""

import torch
import torch.nn as nn


class IBLoss(nn.Module):
    """
    信息瓶颈损失函数

    包含三个部分：
    1. KL 散度损失：压缩表示，最小化 I(X; Z)
    2. 分化损失：让边权重趋向 0 或 1
    """

    def __init__(self, beta=0.01, polar_weight=0.1):
        """
        Args:
            beta: KL 散度损失权重
            polar_weight: 分化损失权重
        """
        super().__init__()
        self.beta = beta
        self.polar_weight = polar_weight

    def kl_divergence(self, mu, logvar):
        """
        计算 KL 散度: KL(q(z|x) || p(z))

        假设:
        - q(z|x) = N(μ, σ²)  编码器输出的分布
        - p(z) = N(0, I)     先验分布

        KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

        Args:
            mu: 均值 [num_nodes, dim]
            logvar: 对数方差 [num_nodes, dim]

        Returns:
            kl_loss: 标量损失值
        """
        # KL 散度的解析解
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def polarization_loss(self, prob):
        """
        分化损失：让 prob 趋向 0 或 1

        Args:
            prob: 边重要性概率 [num_edges, 1]

        Returns:
            polar_loss: 标量损失值
        """
        eps = 1e-8
        entropy = -(prob * torch.log(prob + eps) +
                    (1 - prob) * torch.log(1 - prob + eps))
        return entropy.mean()

    def forward(self, mu, logvar, edge_prob):
        """
        计算总的 IB 损失

        Args:
            mu: 编码均值 [num_nodes, dim]
            logvar: 编码对数方差 [num_nodes, dim]
            edge_prob: 边重要性概率 [num_edges, 1]

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        # KL 散度损失
        kl_loss = self.kl_divergence(mu, logvar)

        # 分化损失
        polar_loss = self.polarization_loss(edge_prob)

        # 总损失
        total_loss = self.beta * kl_loss + self.polar_weight * polar_loss

        loss_dict = {
            'ib_total': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'polar_loss': polar_loss.item(),
            'beta': self.beta,
            'polar_weight': self.polar_weight
        }

        return total_loss, loss_dict


class StochasticEncoder(nn.Module):
    """
    随机编码器：输出分布参数而非确定性表示

    用于实现信息瓶颈的压缩效果
    """

    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
        """
        super().__init__()

        # 均值投影
        self.mu_layer = nn.Linear(in_dim, out_dim)

        # 对数方差投影
        self.logvar_layer = nn.Linear(in_dim, out_dim)

        # 方差下界（防止方差过小导致 KL 爆炸）
        self.min_logvar = -10.0

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        # logvar 初始化为较小值，使初始方差接近 1
        nn.init.zeros_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, h, training=True):
        """
        编码并采样

        Args:
            h: 输入特征 [num_nodes, in_dim]
            training: 是否训练模式

        Returns:
            z: 采样的表示 [num_nodes, out_dim]
            mu: 均值 [num_nodes, out_dim]
            logvar: 对数方差 [num_nodes, out_dim]
        """
        # 计算分布参数
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # 限制方差范围
        logvar = torch.clamp(logvar, min=self.min_logvar)

        if training:
            # 重参数化采样: z = μ + ε * σ
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 测试时直接用均值
            z = mu

        return z, mu, logvar
