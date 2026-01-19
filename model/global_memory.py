"""
GloMem-HoGRN: Global Memory Enhancement Module

This module implements the global memory mechanism for enhancing entity representations
in sparse knowledge graphs.

Core Components:
1. GlobalWriteModule: Aggregates information from all entities to update global memory
2. GlobalReadModule: Distributes global memory to entities via gated fusion
3. MultiHeadGlobalMemory: Multi-head version for capturing diverse global patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalWriteModule(nn.Module):
    """
    Global Write Module: Aggregates information from all entities using attention mechanism.

    Mathematical Formulation:
        e_i = LeakyReLU(a^T [g || h_i])
        α_i = exp(e_i) / Σ_j exp(e_j)
        g^new = Σ_i α_i * h_i

    Args:
        dim (int): Feature dimension
        attention_type (str): Type of attention mechanism
            - 'concat': Concatenation-based attention [g || h_i] (default)
            - 'dot': Dot-product attention g^T h_i
            - 'additive': Additive attention W1*g + W2*h_i
    """

    def __init__(self, dim, attention_type='concat'):
        super(GlobalWriteModule, self).__init__()
        self.dim = dim
        self.attention_type = attention_type

        if attention_type == 'concat':
            # Concatenation attention: a^T [g || h_i]
            self.attention_vec = nn.Parameter(torch.Tensor(2 * dim, 1))
            nn.init.xavier_uniform_(self.attention_vec)

        elif attention_type == 'additive':
            # Additive attention: v^T tanh(W1*g + W2*h_i)
            self.W_g = nn.Linear(dim, dim, bias=False)
            self.W_h = nn.Linear(dim, dim, bias=False)
            self.v = nn.Parameter(torch.Tensor(dim, 1))
            nn.init.xavier_uniform_(self.v)

        elif attention_type == 'dot':
            # Dot-product attention: no additional parameters needed
            pass

        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, global_memory, entity_embeds):
        """
        Forward pass of global write module.

        Args:
            global_memory: Tensor (1, d) - Current global memory vector
            entity_embeds: Tensor (N, d) - All entity embeddings

        Returns:
            g_new: Tensor (1, d) - Updated global memory
            attention_weights: Tensor (N,) - Contribution weight of each entity
        """
        N, d = entity_embeds.shape

        if self.attention_type == 'concat':
            # Expand global memory: (1, d) -> (N, d)
            g_expanded = global_memory.expand(N, d)

            # Concatenate: (N, 2d)
            concat = torch.cat([g_expanded, entity_embeds], dim=1)

            # Compute attention scores: (N, 1)
            e = self.leaky_relu(torch.matmul(concat, self.attention_vec))

        elif self.attention_type == 'dot':
            # Dot-product attention: g^T h_i
            e = torch.matmul(entity_embeds, global_memory.t())  # (N, 1)
            e = e / np.sqrt(d)  # Scaling for numerical stability

        elif self.attention_type == 'additive':
            # Additive attention
            g_expanded = global_memory.expand(N, d)
            e = self.leaky_relu(
                self.W_g(g_expanded) + self.W_h(entity_embeds)
            )  # (N, d)
            e = torch.matmul(e, self.v)  # (N, 1)

        # Softmax normalization
        alpha = F.softmax(e.squeeze(1), dim=0)  # (N,)

        # Weighted aggregation
        g_new = torch.matmul(alpha.unsqueeze(0), entity_embeds)  # (1, d)

        return g_new, alpha


class GlobalReadModule(nn.Module):
    """
    Global Read Module: Distributes global memory to entities via gated fusion.

    Mathematical Formulation:
        β_i = σ(W_gate [h_i || g])
        h_i^enhanced = (1-β_i) * h_i + β_i * g

    The gate value β_i is automatically learned:
        - For sparse nodes: β → 1 (rely on global knowledge)
        - For hub nodes: β → 0 (preserve individuality)

    Args:
        dim (int): Feature dimension
        gate_type (str): Type of gating mechanism
            - 'mlp': Multi-layer perceptron (default)
            - 'linear': Single linear layer
            - 'highway': Highway network style
        use_residual (bool): Whether to use residual connection
    """

    def __init__(self, dim, gate_type='mlp', use_residual=False, extra_input_dim=0):
        super(GlobalReadModule, self).__init__()
        self.dim = dim
        self.gate_type = gate_type
        self.use_residual = use_residual
        self.extra_input_dim = extra_input_dim
        input_dim = 2 * dim + extra_input_dim

        if gate_type == 'mlp':
            # Two-layer MLP gate network
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        elif gate_type == 'linear':
            # Single linear layer gate
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )

        elif gate_type == 'highway':
            # Highway network style
            self.transform_gate = nn.Linear(input_dim, dim)
            self.carry_gate = nn.Linear(input_dim, dim)

        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def forward(self, entity_embeds, global_memory, extra_features=None):
        """
        Forward pass of global read module.

        Args:
            entity_embeds: Tensor (N, d) - Original entity embeddings
            global_memory: Tensor (1, d) - Global memory vector

        Returns:
            enhanced_embeds: Tensor (N, d) - Enhanced entity embeddings
            gate_values: Tensor (N,) - Gate value for each entity
        """
        N, d = entity_embeds.shape

        # Expand global memory: (1, d) -> (N, d)
        g_expanded = global_memory.expand(N, d)

        # Concatenate features: (N, 2d [+ extra])
        concat = torch.cat([entity_embeds, g_expanded], dim=1)
        if extra_features is None and self.extra_input_dim > 0:
            extra_features = entity_embeds.new_zeros((N, self.extra_input_dim))
        if extra_features is not None:
            if extra_features.dim() == 1:
                extra_features = extra_features.unsqueeze(1)
            concat = torch.cat([concat, extra_features], dim=1)

        if self.gate_type in ['mlp', 'linear']:
            # Compute gate values: (N, 1)
            beta = self.gate_network(concat)

            # Adaptive fusion: (N, d)
            enhanced = (1 - beta) * entity_embeds + beta * g_expanded

        elif self.gate_type == 'highway':
            # Highway style fusion
            T = torch.sigmoid(self.transform_gate(concat))  # Transform gate
            C = torch.sigmoid(self.carry_gate(concat))      # Carry gate
            enhanced = T * g_expanded + C * entity_embeds
            beta = T.mean(dim=1, keepdim=True)  # Average as gate value

        # Optional: residual connection
        if self.use_residual:
            enhanced = enhanced + entity_embeds
            enhanced = enhanced / 2.0  # Normalize

        return enhanced, beta.squeeze(1)


class MultiHeadGlobalMemory(nn.Module):
    """
    Multi-Head Global Memory: Uses K global vectors to capture diverse global patterns.

    Similar to Multi-Head Attention, each head captures different semantic aspects.
    For example (K=4):
        - Head 1: Geographic relations (country-city)
        - Head 2: Temporal relations (birth-death)
        - Head 3: Social relations (friend-colleague)
        - Head 4: Attribute relations (color-size)

    Args:
        dim (int): Total feature dimension
        num_heads (int): Number of global memory heads
        attention_type (str): Attention type for write module
        gate_type (str): Gate type for read module
    """

    def __init__(self, dim, num_heads=4, attention_type='concat', gate_type='mlp', extra_input_dim=0):
        super(MultiHeadGlobalMemory, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # K global memory vectors
        self.global_memories = nn.Parameter(
            torch.Tensor(num_heads, self.head_dim)
        )
        nn.init.xavier_uniform_(self.global_memories)

        # Each head has independent Write/Read modules
        self.write_modules = nn.ModuleList([
            GlobalWriteModule(self.head_dim, attention_type)
            for _ in range(num_heads)
        ])

        self.read_modules = nn.ModuleList([
            GlobalReadModule(self.head_dim, gate_type, use_residual=False, extra_input_dim=extra_input_dim)
            for _ in range(num_heads)
        ])

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, entity_embeds, extra_features=None):
        """
        Forward pass of multi-head global memory.

        Args:
            entity_embeds: Tensor (N, d) - All entity embeddings

        Returns:
            enhanced_embeds: Tensor (N, d) - Enhanced entity embeddings
            beta_avg: Tensor (N,) - Average gate value across heads
        """
        N, d = entity_embeds.shape

        # Split into multiple heads: (N, d) -> (N, num_heads, head_dim)
        entity_embeds_split = entity_embeds.view(N, self.num_heads, self.head_dim)

        enhanced_heads = []
        all_betas = []

        for i in range(self.num_heads):
            # Entity embeddings for head i: (N, head_dim)
            h_i = entity_embeds_split[:, i, :]

            # Global memory for head i: (1, head_dim)
            g_i = self.global_memories[i:i+1, :]

            # Write-Read cycle
            g_new, _ = self.write_modules[i](g_i, h_i)
            h_enhanced, beta = self.read_modules[i](h_i, g_new, extra_features=extra_features)

            enhanced_heads.append(h_enhanced)
            all_betas.append(beta)

        # Concatenate: (N, d)
        enhanced = torch.cat(enhanced_heads, dim=1)

        # Output projection
        enhanced = self.output_proj(enhanced)

        # Average gate values across heads
        beta_avg = torch.stack(all_betas, dim=1).mean(dim=1)

        return enhanced, beta_avg
