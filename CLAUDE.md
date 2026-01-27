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

- **`run.py`**: Main entry point. Handles data loading (`load_data()`), adjacency construction (`construct_adj()`), training loop (`fit()`), and evaluation (`evaluate()`).
- **`model/models.py`**: Model definitions (HoGRN_TransE, HoGRN_DistMult, HoGRN_ConvE) inheriting from HoGRNBase with GCN layers and optional enhancement modules.
- **`model/hogrn_conv.py`**: HoGRNConv - relation-aware message-passing layer with mult/sub/corr operations.
- **`model/mixer.py`**: MixerDrop blocks for inter-relation (relation masking) and intra-relation (channel dropout) learning.
- **`model/rpg_module.py`**: RPG-HoGRN modules - PathGuidedAggregator and AdaptiveFusion for sparse node enhancement.
- **`model/path_mining.py`**: PathMiner for discovering frequent 2-3 hop relation paths.
- **`data_loader.py`**: TrainDataset (label smoothing, negative sampling) and TestDataset (full entity evaluation).

### Data Flow

1. Load triplets from `data/<dataset>/` (train.txt, valid.txt, test.txt)
2. Build entity/relation mappings and inverse relations
3. Construct adjacency via `construct_adj()`
4. Optional: Mine relation paths if RPG enabled (`model/path_mining.py`)
5. Forward: entities/relations through GCN layers with optional RPG enhancement
6. Score predictions using TransE/DistMult/ConvE
7. Evaluate with MRR, MR, Hits@K metrics

### Configuration System

JSON configs in `exp_configs/` support all command-line arguments. Keys map to argument names (e.g., `batch` -> `batch_size`). Keys starting with `_` are ignored (metadata/comments).

## Model Architecture

### Base HoGRN
Weight-free GCN message passing with relation reasoning module (Inter/Intra-relation learning):
- **HoGRNConv**: Relation-aware graph convolution with mult/sub/corr composition operations
- **MixerDrop**: Inter-relation learning (relation-wise dropout) + Intra-relation learning (channel-wise processing)
- **Scoring Functions**: TransE (translation), DistMult (bilinear), ConvE (2D convolution)

### RPG-HoGRN (Relation Path Guided Enhancement)
Addresses sparse (low-degree) nodes by propagating features along frequent relation paths:
- **PathMiner**: Discovers frequent 2-3 hop relation paths during preprocessing, saves to `data/{dataset}/paths/frequent_paths.pkl`
- **PathGuidedAggregator**: Builds sparse adjacency matrices per relation, computes path composition via sparse matrix multiplication
- **AdaptiveFusion**: Gate network fuses local (GCN) and remote (path-guided) features based on node sparsity (1/(degree+1))

**RPG Parameters:**
- `-use_rpg`: Enable RPG enhancement
- `-rpg_max_path_length`: Max path length (2-3)
- `-rpg_min_path_count`: Min path frequency threshold
- `-rpg_top_k_paths`: Top-k paths per relation
- `-rpg_sparse_threshold`: Degree threshold for sparse nodes

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-data` | Dataset: FB15K-237-10/20/50, WN18RR, NELL23K, WD-singer |
| `-score_func` | Scoring: transe, distmult, conve |
| `-rel_reason` | Enable relation reasoning (MixerDrop) |
| `-pre_reason` | Apply reasoning before aggregation |
| `-use_rpg` | Enable RPG-HoGRN path guidance |
| `-gcn_layer` | Number of GCN layers (1-4) |
| `-gcn_drop`, `-hid_drop`, `-chan_drop` | Dropout rates |
| `-gamma` | Margin for TransE scoring |

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
