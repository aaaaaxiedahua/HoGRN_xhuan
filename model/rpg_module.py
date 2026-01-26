"""
RPG-HoGRN: Relation Path Guided Module

This module implements path-guided feature propagation for sparse nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class PathEncoder(nn.Module):
    """
    Encode relation paths into vectors using LSTM + Attention.
    """

    def __init__(self, rel_dim, hidden_dim, num_relations):
        super().__init__()
        self.rel_dim = rel_dim
        self.hidden_dim = hidden_dim

        # Relation embeddings for path encoding
        self.rel_embed = nn.Embedding(num_relations * 2, rel_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=rel_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, rel_dim)

    def forward(self, path_relations):
        """
        Args:
            path_relations: [batch, path_len] relation ID sequence

        Returns:
            path_embed: [batch, rel_dim] path embedding
        """
        # Get relation embeddings
        rel_embeds = self.rel_embed(path_relations)  # [batch, path_len, rel_dim]

        # LSTM encoding
        outputs, _ = self.lstm(rel_embeds)  # [batch, path_len, hidden*2]

        # Attention weighting
        attn_scores = self.attention(outputs)  # [batch, path_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        path_repr = (attn_weights * outputs).sum(dim=1)  # [batch, hidden*2]

        # Project to relation space
        path_embed = self.output_proj(path_repr)  # [batch, rel_dim]

        return path_embed


class PathGuidedAggregator(nn.Module):
    """
    Path-Guided Aggregator: Collect remote node features along frequent paths.
    """

    def __init__(self, embed_dim, num_relations, frequent_paths, rel_to_paths,
                 top_k_paths=5, sparse_threshold=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.frequent_paths = frequent_paths
        self.rel_to_paths = rel_to_paths
        self.top_k_paths = top_k_paths
        self.sparse_threshold = sparse_threshold

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0
        self.total_sparse_nodes = 0
        self.total_enhanced_nodes = 0

        print(f"[RPG-Aggregator] embed_dim={embed_dim}, top_k={top_k_paths}, sparse_threshold={sparse_threshold}")
        print(f"[RPG-Aggregator] Loaded {len(frequent_paths)} paths, {len(rel_to_paths)} relations indexed")

        # Learnable path weights
        self.path_weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Adjacency dict will be built from edge_index
        self.adj_by_rel = None

    def build_adjacency(self, edge_index, edge_type, num_ent):
        """Build adjacency dict grouped by relation type."""
        print(f"[RPG-Aggregator] Building adjacency from {edge_index.shape[1]} edges...")
        self.adj_by_rel = {}
        edge_index_np = edge_index.cpu().numpy()
        edge_type_np = edge_type.cpu().numpy()

        for i in range(len(edge_type_np)):
            r = edge_type_np[i]
            h, t = edge_index_np[0, i], edge_index_np[1, i]
            if r not in self.adj_by_rel:
                self.adj_by_rel[r] = {}
            if h not in self.adj_by_rel[r]:
                self.adj_by_rel[r][h] = []
            self.adj_by_rel[r][h].append(t)

        print(f"[RPG-Aggregator] Adjacency built: {len(self.adj_by_rel)} relation types")

    def get_path_endpoints(self, start_nodes, path):
        """
        Find endpoints by following a relation path from start nodes.

        Args:
            start_nodes: set of starting node IDs
            path: tuple of relation IDs (r1, r2, ...)

        Returns:
            endpoints: set of endpoint node IDs
        """
        current = start_nodes
        for rel in path:
            if rel not in self.adj_by_rel:
                return set()
            next_nodes = set()
            for node in current:
                if node in self.adj_by_rel[rel]:
                    next_nodes.update(self.adj_by_rel[rel][node])
            current = next_nodes
            if not current:
                return set()
        return current

    def forward(self, entity_embeds, node_degrees, edge_index, edge_type):
        """
        Path-guided aggregation for sparse nodes.

        Args:
            entity_embeds: [N, dim] entity embeddings
            node_degrees: [N] node degree tensor
            edge_index: [2, E] edge index
            edge_type: [E] edge types

        Returns:
            remote_features: [N, dim] aggregated remote features
        """
        N, dim = entity_embeds.shape
        device = entity_embeds.device

        # Build adjacency if not done
        if self.adj_by_rel is None:
            self.build_adjacency(edge_index, edge_type, N)

        # Initialize output
        remote_features = torch.zeros_like(entity_embeds)
        remote_counts = torch.zeros(N, 1, device=device)

        # Find sparse nodes
        sparse_mask = (node_degrees <= self.sparse_threshold)
        sparse_nodes = torch.where(sparse_mask)[0].cpu().tolist()

        if not sparse_nodes or not self.rel_to_paths:
            return remote_features

        # Process each sparse node
        enhanced_count = 0
        for node_id in sparse_nodes:
            node_features = []

            # Get relations connected to this node
            for rel, paths_list in self.rel_to_paths.items():
                # Use top-k paths for this relation
                for path, count in paths_list[:self.top_k_paths]:
                    endpoints = self.get_path_endpoints({node_id}, path)

                    if endpoints:
                        # Aggregate endpoint features
                        endpoint_ids = list(endpoints)
                        endpoint_embeds = entity_embeds[endpoint_ids]
                        agg_embed = endpoint_embeds.mean(dim=0)

                        # Compute path weight
                        weight = self.path_weight_net(agg_embed)
                        node_features.append(weight * agg_embed)

            if node_features:
                # Average all path features
                remote_features[node_id] = torch.stack(node_features).mean(dim=0)
                remote_counts[node_id] = 1.0
                enhanced_count += 1

        # Log statistics periodically
        self.forward_count += 1
        self.total_sparse_nodes += len(sparse_nodes)
        self.total_enhanced_nodes += enhanced_count

        if self.forward_count % self.log_interval == 0:
            avg_sparse = self.total_sparse_nodes / self.forward_count
            avg_enhanced = self.total_enhanced_nodes / self.forward_count
            enhance_ratio = avg_enhanced / (avg_sparse + 1e-6) * 100
            print(f"[RPG-Aggregator] Step {self.forward_count}: "
                  f"avg_sparse={avg_sparse:.1f}, avg_enhanced={avg_enhanced:.1f}, "
                  f"enhance_ratio={enhance_ratio:.1f}%")

        return remote_features


class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion: Fuse local and remote features based on node sparsity.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Log statistics
        self.log_interval = 100
        self.forward_count = 0
        self.beta_sum = 0.0

        print(f"[RPG-Fusion] embed_dim={embed_dim}, dropout={dropout}")

    def forward(self, h_local, h_remote, node_degrees):
        """
        Args:
            h_local: [N, dim] local GCN features
            h_remote: [N, dim] remote path features
            node_degrees: [N] node degrees

        Returns:
            h_fused: [N, dim] fused features
            beta: [N] fusion weights
        """
        # Compute sparsity indicator
        sparsity = 1.0 / (node_degrees.float() + 1)
        sparsity = sparsity.unsqueeze(1)

        # Concatenate features
        gate_input = torch.cat([h_local, h_remote, sparsity], dim=1)

        # Compute fusion weight
        beta = self.gate(gate_input)

        # Fuse: sparse nodes rely more on remote features
        h_fused = (1 - beta) * h_local + beta * h_remote

        # Log statistics periodically
        self.forward_count += 1
        self.beta_sum += beta.mean().item()

        if self.forward_count % self.log_interval == 0:
            avg_beta = self.beta_sum / self.forward_count
            print(f"[RPG-Fusion] Step {self.forward_count}: avg_beta={avg_beta:.4f}")

        return h_fused, beta.squeeze(1)
