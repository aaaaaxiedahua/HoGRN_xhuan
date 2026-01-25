# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoGRN (High-order Graph Reasoning Network) is a PyTorch-based framework for Knowledge Graph Completion (KGC) using Graph Neural Networks. It implements link prediction on sparse knowledge graphs with explainable high-order reasoning.

**Paper:** "HoGRN: Explainable Sparse Knowledge Graph Completion via High-Order Graph Reasoning Network" (IEEE TKDE 2024)

## Model Architecture

### Base HoGRN (原模型)
原始 HoGRN 通过 Weight-free GCN 进行消息传递，结合关系推理模块（Inter/Intra-relation learning）实现高阶推理。核心组件：
- **HoGRNConv**: 关系感知的图卷积层，支持 mult/sub/corr 组合操作
- **MixerDrop**: 关系推理模块，包含 inter-relation 和 intra-relation 学习
- **Scoring Functions**: TransE (平移距离), DistMult (双线性), ConvE (2D卷积)

### 创新模块 1: GloMem (Global Memory Enhancement)
解决长尾实体（稀疏节点）信息不足的问题。通过全局记忆向量实现"全图广播"。

**核心机制:**
- **Global Write**: 注意力机制聚合全图实体信息到全局向量 $g$
- **Global Read**: 门控机制将全局信息分发给各节点，稀疏节点获得更多全局信息

**参数:**
- `-use_global_memory`: 启用全局记忆
- `-global_attention_type`: 注意力类型 (concat/dot/additive)
- `-global_gate_type`: 门控类型 (mlp/linear/highway)
- `-global_use_residual`: 残差连接
- `-global_memory_heads`: 多头数量

**实现:** `model/global_memory.py` (GlobalWriteModule, GlobalReadModule, MultiHeadGlobalMemory)

### 创新模块 2: VC (Virtual Centroid Enhancement)
为低度节点引入虚拟质心，增强稀疏实体的表示能力。

**核心思想:** 对于度数低于阈值的节点，计算其所属关系的头/尾实体质心作为补充信息。

**参数:**
- `-use_virtual_centroid`: 启用虚拟质心
- `-vc_degree_threshold`: 度数阈值（低于此值的节点使用质心增强）

**实现:** `model/models.py` 中的 `_compute_virtual_centroid()` 方法

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

- **`run.py`**: Main entry point. Handles data loading, model initialization, training loop, and evaluation.
- **`model/models.py`**: Model definitions (HoGRN_TransE, HoGRN_DistMult, HoGRN_ConvE) with GCN layers, global memory, and virtual centroid modules.
- **`model/hogrn_conv.py`**: Custom message-passing layer with relation-aware aggregation and high-order reasoning.
- **`model/global_memory.py`**: GlobalWriteModule (attention-based aggregation) and GlobalReadModule (gated distribution).
- **`model/mixer.py`**: MixerDrop blocks for inter/intra-relation learning.
- **`data_loader.py`**: TrainDataset and TestDataset classes for triplet processing.

### Data Flow

1. Load triplets from `data/<dataset>/` (train.txt, valid.txt, test.txt)
2. Build entity/relation mappings and inverse relations
3. Construct adjacency via `construct_adj()`
4. Forward: entities/relations through GCN layers with optional global memory and virtual centroid
5. Score predictions using TransE/DistMult/ConvE
6. Evaluate with MRR, MR, Hits@K metrics

### Configuration System

JSON configs in `exp_configs/` support all command-line arguments. Keys map to argument names (e.g., `batch` -> `batch_size`). Keys starting with `_` are ignored (metadata).

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-data` | Dataset: FB15K-237-10/20/50, WN18RR, NELL23K, WD-singer |
| `-score_func` | Scoring: transe, distmult, conve |
| `-rel_reason` | Enable relation reasoning |
| `-pre_reason` | Apply reasoning before aggregation |
| `-use_global_memory` | Enable GloMem enhancement |
| `-use_virtual_centroid` | Enable VC enhancement |
| `-gcn_layer` | Number of GCN layers (1-4) |
| `-gcn_drop`, `-hid_drop`, `-chan_drop` | Dropout rates |

## Dependencies

- Python 3.6.8, PyTorch 1.6.0
- torch-sparse 0.4.3, torch-cluster 1.4.5, torch-scatter 2.0.6
- numpy 1.16.3, ordered-set 3.1
