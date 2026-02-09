# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoGRN (High-order Graph Reasoning Network) is a PyTorch-based framework for Knowledge Graph Completion (KGC) using Graph Neural Networks. It implements link prediction on sparse knowledge graphs with explainable high-order reasoning.

**Paper:** "HoGRN: Explainable Sparse Knowledge Graph Completion via High-Order Graph Reasoning Network" (IEEE TKDE 2024)

## Commands

### Training
```bash
# Using config file (recommended)
python run.py -config_file exp_configs/nell23k_conve.json

# Direct command line
python run.py -data NELL23K -score_func conve -batch 256 -rel_reason -pre_reason

# Restore from checkpoint
python run.py -config_file exp_configs/nell23k_conve.json -restore

# Using shell scripts
sh sh/nell23k_conve.sh
```

### Analysis
```bash
python analyze_glomem.py --checkpoint checkpoints/<run_name> --dataset NELL23K --output_dir analysis/<run_name>
```

## Architecture

### Core Components

- **`run.py`**: Main entry point. Contains `Runner` class with data loading (`load_data()`), adjacency construction (`construct_adj()`), training loop (`fit()`), and evaluation (`evaluate()`).
- **`model/models.py`**: Model definitions (HoGRN_TransE, HoGRN_DistMult, HoGRN_ConvE) inheriting from `HoGRNBase`. Each model combines GCN message passing with a specific scoring function.
- **`model/hogrn_conv.py`**: `HoGRNConv` - relation-aware message-passing layer with mult/sub/corr composition operations and optional MixerDrop reasoning.
- **`model/mixer.py`**: `MixerDrop` blocks implementing inter-relation learning (relation masking) and intra-relation learning (channel dropout via MLP).
- **`model/rpg_module.py`**: `PrototypeEnhancer` - computes relation-specific answer prototypes and enhances scores via similarity.
- **`model/path_mining.py`**: `PathMiner` - standalone utility for discovering frequent 2-3 hop relation paths (not integrated into main training).
- **`data_loader.py`**: `TrainDataset` (label smoothing, 1-N training) and `TestDataset` (full entity ranking evaluation).

### Data Flow

1. Load triplets from `data/<dataset>/` (train.txt, valid.txt, test.txt)
2. Build entity/relation mappings; create inverse relations (rel + num_rel)
3. Construct edge_index/edge_type via `construct_adj()`
4. Forward: entities/relations through HoGRNConv layers with optional MixerDrop reasoning
5. Score predictions using TransE/DistMult/ConvE
6. Evaluate with MRR, MR, Hits@1-10 metrics

### Configuration System

JSON configs in `exp_configs/` support all command-line arguments. Key mappings:
- `batch` → `batch_size`
- `data` → `dataset`
- `epoch` → `max_epochs`
- `gcn_drop` → `dropout`

Keys starting with `_` are ignored (used for comments/metadata).

## Model Architecture

### HoGRNConv Layer
Relation-aware message passing without learnable weight matrices:
- **Composition operations**: `mult` (element-wise multiply), `sub` (subtraction), `corr` (circular correlation)
- **Message aggregation**: Bidirectional (in + out + loop) with tanh attention coefficients
- **Optional relation reasoning**: MixerDrop applied before or after aggregation (`-pre_reason` flag)

### MixerDrop Block
Two-stage relation representation learning:
1. **Inter-relation learning**: Transpose relations → MLP across relation dimension → transpose back. Uses relation masking dropout during training.
2. **Intra-relation learning**: MLP across embedding dimension with channel dropout.

### Scoring Functions
- **TransE**: `γ - ||h + r - t||₁` (translation-based)
- **DistMult**: `(h * r) · t + bias` (bilinear)
- **ConvE**: 2D convolution on reshaped (h, r) concatenation → FC → dot product with all entities

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-data` | Dataset: FB15K-237-10/20/50, WN18RR, NELL23K, WD-singer |
| `-score_func` | Scoring: transe, distmult, conve |
| `-opn` | Composition operation: mult, sub, corr |
| `-rel_reason` | Enable MixerDrop relation reasoning |
| `-pre_reason` | Apply reasoning before (vs after) aggregation |
| `-gcn_layer` | Number of GCN layers (1-4) |
| `-gcn_drop` | Dropout in GCN layer |
| `-hid_drop` | Dropout after GCN |
| `-rel_mask` | Relation masking rate in MixerDrop |
| `-chan_drop` | Channel dropout rate in MixerDrop |
| `-gamma` | Margin for TransE scoring |
| `-sim_decay` | Weight for relational contrastive loss (0 to disable) |

## Data Format

```
data/{DATASET}/
├── train.txt    # Training triplets (tab-separated)
├── valid.txt    # Validation triplets
└── test.txt     # Test triplets

Format: subject\trelation\tobject (FB15k-237, WN18RR)
        subject\tobject\trelation (NELL23K, WD-singer)
```

## Dependencies

- Python 3.6.8, PyTorch 1.6.0
- torch-sparse 0.4.3, torch-cluster 1.4.5, torch-scatter 2.0.6
- numpy 1.16.3, ordered-set 3.1
