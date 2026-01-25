# 方案D：关系路径引导的特征传播 (RPG-HoGRN)

## 一、问题背景

### 1.1 当前HoGRN的局限性

HoGRN使用GCN进行消息传递，但存在**感受野受限**的问题：

```
稀疏节点"小明"的困境：

    小明 ----出生地----> 北京 ----位于----> 中国
      ↑                    ↑                  ↑
   (度数=1)            (度数=100)         (度数=500)

问题：
- 1层GCN：小明只能看到北京
- 2层GCN：小明能看到中国，但信息被稀释
- 更多层：过平滑，所有节点趋同
```

### 1.2 核心洞察

知识图谱中存在大量**可复用的推理模式（关系路径）**：

| 路径模式 | 语义含义 | 出现频率 |
|----------|----------|----------|
| 出生地 → 位于 | 人的国籍推断 | 15234次 |
| 导演 → 类型 | 导演擅长的类型 | 8921次 |
| 工作于 → 位于 | 人的工作地点 | 12453次 |
| 毕业于 → 位于 | 人的求学地点 | 9876次 |

**关键思想**：利用这些高频路径，让稀疏节点"借道"获取远距离信息。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      RPG-HoGRN 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ 预处理阶段   │    │  训练阶段   │    │  推理阶段   │     │
│  │             │    │             │    │             │     │
│  │ 路径挖掘    │───>│ 路径编码器  │───>│ 路径引导    │     │
│  │ 路径筛选    │    │ GCN + 融合  │    │ 聚合预测    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、实现步骤

### 3.1 第一步：关系路径挖掘（预处理）

这一步在训练前完成，**不需要梯度**，只是统计分析。

```python
# path_mining.py

from collections import defaultdict
import pickle

class PathMiner:
    """
    关系路径挖掘器
    扫描训练集，统计所有长度为2-3的关系路径
    """

    def __init__(self, triples, num_relations):
        """
        Args:
            triples: List of (head, relation, tail)
            num_relations: 关系总数
        """
        self.triples = triples
        self.num_relations = num_relations

        # 构建邻接表：entity -> [(relation, neighbor), ...]
        self.adj_list = defaultdict(list)
        for h, r, t in triples:
            self.adj_list[h].append((r, t))
            # 添加逆关系
            self.adj_list[t].append((r + num_relations, h))

    def mine_paths(self, max_length=3, min_count=100):
        """
        挖掘高频关系路径

        Returns:
            path_count: Dict[(r1, r2, ...), count]
        """
        path_count = defaultdict(int)

        # 遍历所有三元组作为起点
        for h, r1, t1 in self.triples:
            # 长度为2的路径
            for r2, t2 in self.adj_list[t1]:
                path_count[(r1, r2)] += 1

                # 长度为3的路径
                if max_length >= 3:
                    for r3, t3 in self.adj_list[t2]:
                        path_count[(r1, r2, r3)] += 1

        # 筛选高频路径
        frequent_paths = {
            path: count
            for path, count in path_count.items()
            if count >= min_count
        }

        return frequent_paths

    def get_path_to_relations(self, frequent_paths):
        """
        构建：查询关系 -> 相关路径 的映射

        例如：查询"国籍"时，应该使用哪些路径？
        """
        # 这里简化处理：路径的第一个关系作为"触发关系"
        rel_to_paths = defaultdict(list)

        for path, count in frequent_paths.items():
            first_rel = path[0]
            rel_to_paths[first_rel].append((path, count))

        # 按频率排序
        for rel in rel_to_paths:
            rel_to_paths[rel].sort(key=lambda x: -x[1])

        return dict(rel_to_paths)


# 使用示例
if __name__ == "__main__":
    # 假设训练集
    triples = [
        (0, 0, 1),   # 小明 -出生地-> 北京
        (1, 1, 2),   # 北京 -位于-> 中国
        (3, 0, 4),   # 张三 -出生地-> 上海
        (4, 1, 2),   # 上海 -位于-> 中国
        # ... 更多三元组
    ]

    miner = PathMiner(triples, num_relations=10)
    frequent_paths = miner.mine_paths(min_count=2)

    print("高频路径：")
    for path, count in frequent_paths.items():
        print(f"  {path}: {count}次")

    # 输出示例：
    # (出生地, 位于): 15234次
    # (工作于, 位于): 8923次
```

---

### 3.2 第二步：路径编码器

将关系路径编码为向量表示。

```python
# path_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PathEncoder(nn.Module):
    """
    路径编码器：将关系序列编码为向量
    """

    def __init__(self, rel_dim, hidden_dim, num_relations):
        super().__init__()
        self.rel_dim = rel_dim
        self.hidden_dim = hidden_dim

        # 关系嵌入（与主模型共享或独立）
        self.rel_embed = nn.Embedding(num_relations * 2, rel_dim)

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=rel_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 注意力层
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim * 2, rel_dim)

    def forward(self, path_relations):
        """
        Args:
            path_relations: [batch, path_len] 关系ID序列

        Returns:
            path_embed: [batch, rel_dim] 路径嵌入
        """
        # 获取关系嵌入
        rel_embeds = self.rel_embed(path_relations)  # [batch, path_len, rel_dim]

        # LSTM编码
        outputs, _ = self.lstm(rel_embeds)  # [batch, path_len, hidden*2]

        # 注意力加权
        attn_scores = self.attention(outputs)  # [batch, path_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # 加权求和
        path_repr = (attn_weights * outputs).sum(dim=1)  # [batch, hidden*2]

        # 投影到关系空间
        path_embed = self.output_proj(path_repr)  # [batch, rel_dim]

        return path_embed
```

---

### 3.3 第三步：路径引导聚合模块

核心模块：沿高频路径收集远程信息。

```python
# path_aggregator.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add

class PathGuidedAggregator(nn.Module):
    """
    路径引导聚合器：沿高频路径收集远程节点信息
    """

    def __init__(self, embed_dim, frequent_paths, rel_to_paths, path_encoder):
        super().__init__()
        self.embed_dim = embed_dim
        self.frequent_paths = frequent_paths  # Dict: path -> count
        self.rel_to_paths = rel_to_paths      # Dict: rel -> [(path, count), ...]
        self.path_encoder = path_encoder

        # 路径重要性权重（可学习）
        self.path_weights = nn.ParameterDict()
        for path in frequent_paths:
            path_key = "_".join(map(str, path))
            self.path_weights[path_key] = nn.Parameter(torch.ones(1))

    def get_path_endpoints(self, node_ids, path, edge_index, edge_type):
        """
        从起始节点出发，沿路径找到终点节点

        Args:
            node_ids: [N] 起始节点
            path: (r1, r2, ...) 关系路径
            edge_index: [2, E] 边索引
            edge_type: [E] 边类型

        Returns:
            endpoints: Dict[start_node] -> List[end_nodes]
        """
        current_nodes = {n.item(): {n.item()} for n in node_ids}

        for rel in path:
            next_nodes = {n: set() for n in current_nodes}

            # 找到类型为rel的所有边
            rel_mask = (edge_type == rel)
            rel_edges = edge_index[:, rel_mask]

            for start in current_nodes:
                for mid in current_nodes[start]:
                    # 找mid通过rel连接的邻居
                    mask = (rel_edges[0] == mid)
                    neighbors = rel_edges[1, mask].tolist()
                    next_nodes[start].update(neighbors)

            current_nodes = {k: v for k, v in next_nodes.items() if v}

        return current_nodes

    def forward(self, entity_embeds, query_rels, edge_index, edge_type, sparse_mask):
        """
        路径引导聚合

        Args:
            entity_embeds: [N, dim] 实体嵌入
            query_rels: [batch] 查询关系
            edge_index: [2, E] 边索引
            edge_type: [E] 边类型
            sparse_mask: [N] 稀疏节点掩码

        Returns:
            remote_features: [N, dim] 远程聚合特征
        """
        N, dim = entity_embeds.shape
        remote_features = torch.zeros_like(entity_embeds)
        remote_counts = torch.zeros(N, 1, device=entity_embeds.device)

        # 只对稀疏节点进行路径聚合
        sparse_nodes = torch.where(sparse_mask)[0]

        for query_rel in query_rels.unique():
            # 获取该关系相关的路径
            if query_rel.item() not in self.rel_to_paths:
                continue

            paths = self.rel_to_paths[query_rel.item()]

            for path, count in paths[:5]:  # 只用top-5路径
                # 沿路径找终点
                endpoints = self.get_path_endpoints(
                    sparse_nodes, path, edge_index, edge_type
                )

                # 聚合终点特征
                path_key = "_".join(map(str, path))
                weight = torch.sigmoid(self.path_weights[path_key])

                for start, ends in endpoints.items():
                    if ends:
                        end_embeds = entity_embeds[list(ends)]
                        agg_embed = end_embeds.mean(dim=0)
                        remote_features[start] += weight * agg_embed
                        remote_counts[start] += 1

        # 归一化
        remote_features = remote_features / (remote_counts + 1e-8)

        return remote_features
```

---

### 3.4 第四步：自适应融合模块

将本地特征和远程特征融合。

```python
# adaptive_fusion.py

class AdaptiveFusion(nn.Module):
    """
    自适应融合：根据节点稀疏程度决定融合比例
    """

    def __init__(self, embed_dim):
        super().__init__()
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_local, h_remote, node_degrees):
        """
        Args:
            h_local: [N, dim] 本地GCN特征
            h_remote: [N, dim] 远程路径特征
            node_degrees: [N] 节点度数

        Returns:
            h_fused: [N, dim] 融合后特征
        """
        # 归一化度数作为稀疏度指标
        sparsity = 1.0 / (node_degrees.float() + 1)
        sparsity = sparsity.unsqueeze(1)  # [N, 1]

        # 拼接特征
        gate_input = torch.cat([h_local, h_remote, sparsity], dim=1)

        # 计算融合权重
        beta = self.gate(gate_input)  # [N, 1]

        # 融合：稀疏节点更依赖远程特征
        h_fused = (1 - beta) * h_local + beta * h_remote

        return h_fused, beta
```

---

### 3.5 第五步：与HoGRN集成

```python
# 修改 model/models.py

class RPG_HoGRNBase(HoGRNBase):
    """
    RPG-HoGRN: 关系路径引导的HoGRN
    """

    def __init__(self, edge_index, edge_type, num_rel, params=None,
                 frequent_paths=None, rel_to_paths=None):
        super().__init__(edge_index, edge_type, num_rel, params)

        # 路径编码器
        self.path_encoder = PathEncoder(
            rel_dim=params.embed_dim,
            hidden_dim=params.embed_dim,
            num_relations=num_rel
        )

        # 路径聚合器
        self.path_aggregator = PathGuidedAggregator(
            embed_dim=params.embed_dim,
            frequent_paths=frequent_paths,
            rel_to_paths=rel_to_paths,
            path_encoder=self.path_encoder
        )

        # 自适应融合
        self.fusion = AdaptiveFusion(params.embed_dim)

        # 稀疏阈值
        self.sparse_threshold = 5

    def forward_base(self, sub, rel, drop1, drop2):
        # 原始GCN前向传播
        edge_index, edge_type = self.edge_index, self.edge_type
        r = self.init_rel

        # 第一层GCN
        x, r = self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r)
        x = drop1(x)

        # 【核心】路径引导聚合
        sparse_mask = (self.node_deg_raw <= self.sparse_threshold)
        h_remote = self.path_aggregator(
            x, rel, edge_index, edge_type, sparse_mask
        )

        # 自适应融合
        x, fusion_weights = self.fusion(x, h_remote, self.node_deg_raw)

        # 后续GCN层
        if self.p.gcn_layer >= 2:
            x, r = self.conv2(x, edge_index, edge_type, rel_embed=r)
            x = drop2(x)

        # 获取查询实体嵌入
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x, 0
```

---

## 四、具体示例

### 4.1 示例场景

```
知识图谱片段：

实体：
- 小明 (ID=0, 度数=1, 稀疏节点)
- 北京 (ID=1, 度数=50)
- 中国 (ID=2, 度数=200)
- 张三 (ID=3, 度数=2, 稀疏节点)
- 上海 (ID=4, 度数=80)

关系：
- 出生地 (ID=0)
- 位于 (ID=1)
- 国籍 (ID=2)

三元组：
- (小明, 出生地, 北京)
- (北京, 位于, 中国)
- (张三, 出生地, 上海)
- (上海, 位于, 中国)
- ... 更多三元组

任务：预测 (小明, 国籍, ?)
```

### 4.2 传统GCN的困境

```
Step 1: 小明通过1层GCN聚合邻居
  小明的邻居：只有北京
  h_小明 = aggregate([h_北京])
  问题：北京的特征是"城市"，不直接包含国籍信息

Step 2: 如果用2层GCN
  第1层：h_小明 = f(h_北京)
  第2层：h_小明 = f(h_北京的邻居) = f(h_中国, h_其他城市, ...)
  问题：信息被大量稀释，中国只是众多邻居之一

结果：小明的嵌入质量差，预测国籍困难
```

### 4.3 RPG-HoGRN的解决方案

```
Step 1: 路径挖掘（预处理）
  发现高频路径：(出生地, 位于) 出现15234次
  语义：出生地的城市所在的国家 ≈ 国籍

Step 2: 查询关系分析
  查询关系：国籍
  相关路径：[(出生地, 位于), ...]

Step 3: 路径引导聚合
  从小明出发，沿路径 (出生地, 位于) 走：
  小明 --出生地--> 北京 --位于--> 中国

  收集终点特征：h_remote = h_中国

Step 4: 自适应融合
  小明度数=1，很稀疏
  beta = sigmoid(gate([h_小明, h_中国, 1.0])) ≈ 0.8

  h_小明_new = 0.2 * h_小明 + 0.8 * h_中国

Step 5: 预测
  h_小明_new 现在包含"中国"的特征
  与候选实体计算相似度，中国得分最高
  预测结果：(小明, 国籍, 中国) ✓
```

### 4.4 数据流图

```
                    ┌─────────────────────────────────────┐
                    │           输入层                     │
                    │  实体嵌入 init_embed [N, dim]        │
                    │  关系嵌入 init_rel [R, dim]          │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         GCN Layer 1                  │
                    │  x = conv1(init_embed, edges, rels)  │
                    │  输出: h_local [N, dim]              │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │                                      │
                    ▼                                      ▼
    ┌───────────────────────────┐        ┌───────────────────────────┐
    │    路径引导聚合 (稀疏节点)  │        │    直接传递 (稠密节点)     │
    │                           │        │                           │
    │  1. 查找相关路径           │        │    h_dense = h_local      │
    │  2. 沿路径收集终点特征     │        │                           │
    │  3. 加权聚合              │        │                           │
    │                           │        │                           │
    │  输出: h_remote           │        │                           │
    └─────────────┬─────────────┘        └─────────────┬─────────────┘
                  │                                    │
                  └─────────────┬──────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────────┐
                    │         自适应融合                   │
                    │  beta = gate([h_local, h_remote])   │
                    │  h_fused = (1-β)*h_local + β*h_remote│
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         GCN Layer 2+ (可选)          │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         评分函数                     │
                    │  score = ConvE(h_sub, r, h_obj)     │
                    └─────────────────────────────────────┘
```

---

## 五、训练流程

### 5.1 完整训练代码

```python
# train_rpg.py

import torch
import pickle
from path_mining import PathMiner

def train_rpg_hogrn(args):
    # ========== 1. 预处理：路径挖掘 ==========
    print("Step 1: 挖掘高频路径...")
    miner = PathMiner(train_triples, num_relations=args.num_rel)
    frequent_paths = miner.mine_paths(
        max_length=args.max_path_length,
        min_count=args.min_path_count
    )
    rel_to_paths = miner.get_path_to_relations(frequent_paths)

    print(f"  发现 {len(frequent_paths)} 条高频路径")

    # 保存路径（可复用）
    with open('frequent_paths.pkl', 'wb') as f:
        pickle.dump((frequent_paths, rel_to_paths), f)

    # ========== 2. 构建模型 ==========
    print("Step 2: 构建RPG-HoGRN模型...")
    model = RPG_HoGRN_ConvE(
        edge_index=edge_index,
        edge_type=edge_type,
        params=args,
        frequent_paths=frequent_paths,
        rel_to_paths=rel_to_paths
    )

    # ========== 3. 训练循环 ==========
    print("Step 3: 开始训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            sub, rel, obj, label = batch
            pred, _ = model(sub, rel)
            loss = model.loss(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        if epoch % 5 == 0:
            mrr = evaluate(model, valid_loader)
            print(f"Epoch {epoch}: Loss={total_loss:.4f}, MRR={mrr:.4f}")
```

### 5.2 超参数配置

```json
{
    "_comment": "RPG-HoGRN 配置文件",

    "基础参数": {
        "data": "NELL23K",
        "score_func": "conve",
        "batch": 256,
        "epoch": 500,
        "lr": 0.001
    },

    "路径挖掘参数": {
        "max_path_length": 3,
        "min_path_count": 100,
        "top_k_paths": 5
    },

    "融合参数": {
        "sparse_threshold": 5,
        "fusion_dropout": 0.1
    }
}
```

---

## 六、与其他方案对比

| 特性 | GloMem | VC | RPG-HoGRN |
|------|--------|-----|-----------|
| 信息来源 | 全局平均 | 关系质心 | 路径终点 |
| 语义精度 | 低（混合） | 中 | 高（路径特定） |
| 可解释性 | 低 | 中 | 高（可展示路径） |
| 计算开销 | O(N) | O(E) | O(稀疏节点×路径数) |
| 预处理 | 无 | 无 | 需要路径挖掘 |

---

## 七、总结

### 7.1 核心优势

1. **精准的远程信息获取**：通过高频路径直接获取相关的远程节点特征
2. **可解释性强**：可以展示使用了哪些推理路径
3. **计算高效**：路径挖掘是预处理，推理时只需查表

### 7.2 适用场景

- 存在明显推理模式的知识图谱
- 稀疏节点与有用信息相隔多跳
- 需要可解释性的应用场景

### 7.3 实现检查清单

- [ ] 实现 PathMiner 类
- [ ] 实现 PathEncoder 类
- [ ] 实现 PathGuidedAggregator 类
- [ ] 实现 AdaptiveFusion 类
- [ ] 修改 HoGRNBase 集成新模块
- [ ] 添加配置参数
- [ ] 测试验证

