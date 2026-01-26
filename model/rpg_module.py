"""
RPG-HoGRN: Relation Path Guided Module

This module implements path-guided feature propagation for sparse nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul


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
        # Sparse matrices for GPU acceleration
        self.path_matrices = None
        self.sparse_mask = None
        self.num_ent = None

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

    def build_relation_matrices(self, edge_index, edge_type, num_ent, device):
        """Build sparse adjacency matrix for each relation."""
        print(f"[RPG-Aggregator] Building relation matrices...")
        self.rel_matrices = {}

        for r in range(self.num_relations * 2):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            r_edge = edge_index[:, mask]
            row, col = r_edge[0], r_edge[1]
            val = torch.ones(row.size(0), device=device)
            self.rel_matrices[r] = SparseTensor(
                row=row, col=col, value=val,
                sparse_sizes=(num_ent, num_ent)
            )

        print(f"[RPG-Aggregator] Built {len(self.rel_matrices)} relation matrices")

    def build_path_matrices(self, node_degrees, num_ent, device):
        """Build combined path matrix for sparse nodes (precompute once)."""
        print(f"[RPG-Aggregator] Building path matrices...")

        # Get sparse node mask
        self.sparse_mask = (node_degrees <= self.sparse_threshold)
        sparse_count = self.sparse_mask.sum().item()

        # Collect all path edges
        all_rows, all_cols = [], []
        path_count = 0

        for path, freq in self.frequent_paths.items():
            if path_count >= len(self.frequent_paths):
                break

            # Compute path matrix: A_r1 @ A_r2 @ ...
            if path[0] not in self.rel_matrices:
                continue

            path_mat = self.rel_matrices[path[0]]
            valid = True

            for r in path[1:]:
                if r not in self.rel_matrices:
                    valid = False
                    break
                path_mat = path_mat @ self.rel_matrices[r]

            if not valid:
                continue

            # Extract edges from path matrix
            row, col, _ = path_mat.coo()
            all_rows.append(row)
            all_cols.append(col)
            path_count += 1

        if all_rows:
            # Combine all path edges
            all_rows = torch.cat(all_rows)
            all_cols = torch.cat(all_cols)

            # Remove duplicates
            edge_hash = all_rows * num_ent + all_cols
            unique_hash, inverse = torch.unique(edge_hash, return_inverse=True)
            unique_rows = unique_hash // num_ent
            unique_cols = unique_hash % num_ent

            # Build combined sparse matrix
            val = torch.ones(unique_rows.size(0), device=device)
            self.path_matrices = SparseTensor(
                row=unique_rows, col=unique_cols, value=val,
                sparse_sizes=(num_ent, num_ent)
            )

            # Count edges per node for normalization
            self.node_edge_count = scatter_add(
                val, unique_rows, dim=0, dim_size=num_ent
            )
            self.node_edge_count = self.node_edge_count.clamp(min=1)

            print(f"[RPG-Aggregator] Path matrix: {unique_rows.size(0)} edges, "
                  f"{sparse_count} sparse nodes")
        else:
            print(f"[RPG-Aggregator] Warning: No valid path matrices built")

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
        Path-guided aggregation using sparse matrix multiplication (GPU).
        """
        N, dim = entity_embeds.shape
        device = entity_embeds.device
        self.num_ent = N

        # Build matrices on first call (precompute once)
        if self.path_matrices is None:
            self.build_relation_matrices(edge_index, edge_type, N, device)
            self.build_path_matrices(node_degrees, N, device)

        # If no valid paths, return zeros
        if self.path_matrices is None:
            return torch.zeros_like(entity_embeds)

        # GPU sparse matrix multiplication: aggregate remote features
        # remote_features[i] = sum of entity_embeds[j] for all j reachable from i
        remote_features = matmul(self.path_matrices, entity_embeds)

        # Normalize by edge count
        remote_features = remote_features / self.node_edge_count.unsqueeze(1)

        # Only keep features for sparse nodes
        sparse_mask = (node_degrees <= self.sparse_threshold)
        remote_features = remote_features * sparse_mask.float().unsqueeze(1)

        # Log statistics
        self.forward_count += 1
        if self.forward_count % self.log_interval == 0:
            sparse_count = sparse_mask.sum().item()
            enhanced = (remote_features.abs().sum(dim=1) > 0).sum().item()
            print(f"[RPG-Aggregator] Step {self.forward_count}: "
                  f"sparse={sparse_count}, enhanced={enhanced}")

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
