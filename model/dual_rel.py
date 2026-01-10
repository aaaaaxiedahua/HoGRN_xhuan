import torch
import torch.nn as nn

from torch_scatter import scatter_add


class DualRelGCN(nn.Module):
	def __init__(self, dim, alpha=0.1, dropout=0.0):
		super().__init__()
		self.alpha = float(alpha)
		self.norm = nn.LayerNorm(dim, elementwise_affine=False)
		self.proj = nn.Linear(dim, dim, bias=False)
		self.drop = nn.Dropout(dropout)

	def forward(self, rel_embed, rel_edge_index, rel_edge_weight):
		if self.alpha == 0.0:
			return rel_embed
		if rel_edge_index is None or rel_edge_index.numel() == 0:
			return rel_embed

		src, dst = rel_edge_index[0], rel_edge_index[1]
		if rel_edge_weight is None or rel_edge_weight.numel() == 0:
			edge_weight = torch.ones(src.size(0), device=rel_embed.device, dtype=rel_embed.dtype)
		else:
			edge_weight = rel_edge_weight.to(device=rel_embed.device, dtype=rel_embed.dtype)

		msg = rel_embed.index_select(0, src) * edge_weight.view(-1, 1)
		agg = scatter_add(msg, dst, dim=0, dim_size=rel_embed.size(0))
		denom = scatter_add(edge_weight, dst, dim=0, dim_size=rel_embed.size(0)).clamp_min(1e-12)
		agg = agg / denom.view(-1, 1)

		delta = self.proj(self.norm(agg))
		return rel_embed + self.alpha * self.drop(delta)

