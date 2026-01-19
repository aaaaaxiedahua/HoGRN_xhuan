"""
GloMem-HoGRN Visualization and Analysis Script

This script provides tools for analyzing and visualizing the behavior of GloMem-HoGRN:
1. Gate value distribution analysis
2. Global memory semantics analysis
3. Case study for long-tail entities
4. Performance comparison

Usage:
    python analyze_glomem.py --checkpoint ./checkpoints/glomem_model --dataset FB15k-237
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
from collections import defaultdict
from run import Runner
from helper import *

# Set style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def compute_node_degrees(edge_index, num_nodes):
    """
    Compute the degree of each node in the graph.

    Args:
        edge_index: Tensor (2, E) - Edge indices
        num_nodes: int - Number of nodes

    Returns:
        degrees: Tensor (N,) - Degree of each node
    """
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        degrees[src] += 1
        degrees[dst] += 1
    return degrees


class GloMemAnalyzer:
    """
    Analyzer for GloMem-HoGRN model.
    """

    def __init__(self, model, runner):
        """
        Args:
            model: Trained GloMem-HoGRN model
            runner: Runner instance containing data
        """
        self.model = model
        self.runner = runner
        self.device = model.device

        # Compute node degrees
        self.node_degrees = compute_node_degrees(
            runner.edge_index,
            runner.p.num_ent
        ).cpu().numpy()

    def _get_global_memory_input(self):
        r = self.model.init_rel if self.model.p.score_func != 'transe' else torch.cat(
            [self.model.init_rel, -self.model.init_rel], dim=0
        )
        x, _ = self.model.conv1(self.model.init_embed, self.model.edge_index, self.model.edge_type, rel_embed=r)
        return x

    def _get_degree_features(self):
        if hasattr(self.model, 'node_deg'):
            return self.model.node_deg.unsqueeze(1)
        return None

    def analyze_gate_distribution(self, save_path='./analysis/gate_distribution.png'):
        """
        Analyze the distribution of gate values across different node degrees.

        Expected behavior:
        - Low-degree nodes: β → 1 (rely on global memory)
        - High-degree nodes: β → 0 (preserve individuality)
        """
        print("\n" + "="*60)
        print("Analyzing Gate Value Distribution")
        print("="*60)

        self.model.eval()

        with torch.no_grad():
            x_input = self._get_global_memory_input()
            deg_input = self._get_degree_features()
            # Get gate values for all entities
            if hasattr(self.model, 'global_memory_module'):
                # Multi-head version
                _, gate_values = self.model.global_memory_module(x_input, extra_features=deg_input)
            else:
                # Single-head version
                g_new, _ = self.model.global_write(
                    self.model.global_memory,
                    x_input
                )
                _, gate_values = self.model.global_read(
                    x_input,
                    g_new,
                    extra_features=deg_input
                )

            gate_values = gate_values.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Scatter plot: Gate value vs Node degree
        ax1 = axes[0, 0]
        ax1.scatter(self.node_degrees, gate_values, alpha=0.3, s=10)
        ax1.set_xlabel('Node Degree', fontsize=12)
        ax1.set_ylabel('Gate Value (β)', fontsize=12)
        ax1.set_title('Gate Value vs Node Degree', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(self.node_degrees, gate_values, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(self.node_degrees.min(), self.node_degrees.max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
        ax1.legend()

        # 2. Box plot: Gate values by degree bins
        ax2 = axes[0, 1]
        degree_bins = [0, 5, 10, 20, 50, 100, np.inf]
        bin_labels = ['0-5', '5-10', '10-20', '20-50', '50-100', '100+']
        binned_gates = []

        for i in range(len(degree_bins) - 1):
            mask = (self.node_degrees >= degree_bins[i]) & (self.node_degrees < degree_bins[i+1])
            if mask.sum() > 0:
                binned_gates.append(gate_values[mask])
            else:
                binned_gates.append([])

        ax2.boxplot([g for g in binned_gates if len(g) > 0],
                    labels=[bin_labels[i] for i, g in enumerate(binned_gates) if len(g) > 0])
        ax2.set_xlabel('Node Degree Range', fontsize=12)
        ax2.set_ylabel('Gate Value (β)', fontsize=12)
        ax2.set_title('Gate Value Distribution by Degree', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Histogram: Gate value distribution
        ax3 = axes[1, 0]
        ax3.hist(gate_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(gate_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gate_values.mean():.3f}')
        ax3.set_xlabel('Gate Value (β)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Overall Gate Value Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Compute statistics
        low_degree_mask = self.node_degrees < 5
        high_degree_mask = self.node_degrees > 50

        stats_data = [
            ['Metric', 'Value'],
            ['Overall Mean β', f'{gate_values.mean():.4f}'],
            ['Overall Std β', f'{gate_values.std():.4f}'],
            ['Low-degree (<5) Mean β', f'{gate_values[low_degree_mask].mean():.4f}'],
            ['High-degree (>50) Mean β', f'{gate_values[high_degree_mask].mean():.4f}'],
            ['Correlation (degree, β)', f'{np.corrcoef(self.node_degrees, gate_values)[0,1]:.4f}'],
        ]

        table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gate distribution analysis saved to {save_path}")

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Overall Mean β: {gate_values.mean():.4f}")
        print(f"  Low-degree (<5) Mean β: {gate_values[low_degree_mask].mean():.4f}")
        print(f"  High-degree (>50) Mean β: {gate_values[high_degree_mask].mean():.4f}")
        print(f"  Correlation (degree, β): {np.corrcoef(self.node_degrees, gate_values)[0,1]:.4f}")

        return gate_values

    def analyze_global_memory_semantics(self, top_k=20, save_path='./analysis/global_memory_semantics.txt'):
        """
        Analyze what semantics the global memory captures.
        Find top-k entities most similar to global memory.
        """
        print("\n" + "="*60)
        print("Analyzing Global Memory Semantics")
        print("="*60)

        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, 'global_memory'):
                g = self.model.global_memory  # (1, d)
            else:
                # Multi-head: use first head
                g = self.model.global_memory_module.global_memories[0:1, :]

            H = self._get_global_memory_input()  # (N, d)
            if hasattr(self.model, 'global_memory_module'):
                head_dim = self.model.global_memory_module.head_dim
                H = H[:, :head_dim]

            # Compute cosine similarity
            g_norm = torch.nn.functional.normalize(g, p=2, dim=1)
            H_norm = torch.nn.functional.normalize(H, p=2, dim=1)
            similarities = torch.mm(g_norm, H_norm.t()).squeeze()  # (N,)

            # Top-k most similar entities
            top_values, top_indices = torch.topk(similarities, top_k)
            top_indices = top_indices.cpu().numpy()
            top_values = top_values.cpu().numpy()

        # Save results
        with open(save_path, 'w') as f:
            f.write(f"Top-{top_k} entities most similar to global memory:\n")
            f.write("="*60 + "\n\n")

            for i, (idx, sim) in enumerate(zip(top_indices, top_values)):
                entity_name = self.runner.id2ent[idx]
                degree = self.node_degrees[idx]
                f.write(f"{i+1}. {entity_name}\n")
                f.write(f"   Similarity: {sim:.4f}, Degree: {degree}\n\n")

        print(f"✓ Global memory semantics saved to {save_path}")
        print(f"\nTop-5 entities:")
        for i in range(min(5, top_k)):
            idx = top_indices[i]
            print(f"  {i+1}. {self.runner.id2ent[idx]} (sim={top_values[i]:.4f}, degree={self.node_degrees[idx]})")

        return top_indices, top_values


def main():
    """
    Main function for running GloMem analysis.
    """
    parser = argparse.ArgumentParser(description='GloMem-HoGRN Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./analysis', help='Output directory for analysis results')
    parser.add_argument('--model', type=str, default='hogrn', help='Model name')
    parser.add_argument('--score_func', type=str, default='conve', help='Score function')
    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("GloMem-HoGRN Analysis Tool")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")

    # Load checkpoint to get training args
    print("\nLoading checkpoint...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    state = torch.load(args.checkpoint, map_location='cpu')
    saved_args = argparse.Namespace(**state['args'])

    # Override dataset if specified
    saved_args.dataset = args.dataset
    saved_args.gpu = '-1'  # Use CPU for analysis

    print(f"✓ Checkpoint loaded (Best MRR: {state['best_val']['mrr']:.4f})")

    # Create runner and load model
    print("\nInitializing model...")
    runner = Runner(saved_args)
    runner.load_model(args.checkpoint)
    model = runner.model
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  - Entities: {runner.p.num_ent}")
    print(f"  - Relations: {runner.p.num_rel}")
    print(f"  - GloMem enabled: {hasattr(runner.p, 'use_global_memory') and runner.p.use_global_memory}")

    # Create analyzer
    print("\nCreating analyzer...")
    analyzer = GloMemAnalyzer(model, runner)

    # Run analyses
    print("\n" + "="*60)
    analyzer.analyze_gate_distribution(save_path=f'{args.output_dir}/gate_distribution.png')
    analyzer.analyze_global_memory_semantics(save_path=f'{args.output_dir}/global_memory_semantics.txt')

    print("\n" + "="*60)
    print("Analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
