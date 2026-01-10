import numpy as np
import torch


_WALK_MODES = {"path", "share_src", "share_dst", "mixed"}
_WEIGHT_MODES = {"count", "pmi", "ppmi"}


def build_static_relation_graph(
	edge_index,
	edge_type,
	num_nodes,
	num_rel_types,
	*,
	walk="path",
	weight="ppmi",
	hops=2,
	topk=20,
	min_count=1,
	sym=True,
	eps=1e-12,
):
	if walk not in _WALK_MODES:
		raise ValueError(f"walk must be one of {sorted(_WALK_MODES)}, got: {walk}")
	if weight not in _WEIGHT_MODES:
		raise ValueError(f"weight must be one of {sorted(_WEIGHT_MODES)}, got: {weight}")
	if num_rel_types <= 0:
		raise ValueError(f"num_rel_types must be positive, got: {num_rel_types}")
	if num_nodes <= 0:
		raise ValueError(f"num_nodes must be positive, got: {num_nodes}")
	if hops < 2:
		raise ValueError(f"hops must be >= 2, got: {hops}")

	src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
	dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
	etype = edge_type.detach().cpu().numpy().astype(np.int64, copy=False)

	out_order = np.argsort(src, kind="mergesort")
	in_order = np.argsort(dst, kind="mergesort")

	out_src = src[out_order]
	out_type = etype[out_order]
	out_counts = np.bincount(out_src, minlength=num_nodes)
	out_ptr = np.concatenate(([0], np.cumsum(out_counts)))

	in_dst = dst[in_order]
	in_type = etype[in_order]
	in_counts = np.bincount(in_dst, minlength=num_nodes)
	in_ptr = np.concatenate(([0], np.cumsum(in_counts)))

	counts = np.zeros((num_rel_types, num_rel_types), dtype=np.float64)

	want_path = walk in {"path", "mixed"}
	want_share_src = walk in {"share_src", "mixed"}
	want_share_dst = walk in {"share_dst", "mixed"}

	for node in range(num_nodes):
		out_slice = out_type[out_ptr[node] : out_ptr[node + 1]]
		in_slice = in_type[in_ptr[node] : in_ptr[node + 1]]

		out_u, out_c = (None, None)
		in_u, in_c = (None, None)

		if want_share_src or want_path:
			if out_slice.size:
				out_u, out_c = np.unique(out_slice, return_counts=True)

		if want_share_dst or want_path:
			if in_slice.size:
				in_u, in_c = np.unique(in_slice, return_counts=True)

		if want_path and in_u is not None and out_u is not None:
			counts[np.ix_(in_u, out_u)] += np.outer(in_c, out_c)

		if want_share_src and out_u is not None:
			counts[np.ix_(out_u, out_u)] += np.outer(out_c, out_c)

		if want_share_dst and in_u is not None:
			counts[np.ix_(in_u, in_u)] += np.outer(in_c, in_c)

	np.fill_diagonal(counts, 0.0)

	if min_count > 1:
		counts[counts < float(min_count)] = 0.0

	if weight == "count":
		weights = counts
	else:
		total = float(counts.sum())
		if total <= 0:
			weights = np.zeros_like(counts)
		else:
			row = counts.sum(axis=1)
			col = counts.sum(axis=0)
			denom = np.outer(row, col)
			pmi = np.log((counts + eps) * total / (denom + eps))
			pmi[counts <= 0] = 0.0
			if weight == "ppmi":
				pmi = np.maximum(pmi, 0.0)
			weights = pmi

	if sym:
		weights = np.maximum(weights, weights.T)

	if hops > 2:
		base = np.maximum(weights, 0.0)
		row_sum = base.sum(axis=1, keepdims=True)
		if float(row_sum.sum()) > 0:
			p = base / (row_sum + eps)
			expanded = p.copy()
			p_power = p.copy()
			for d in range(2, hops):
				p_power = p_power @ p
				expanded += (1.0 / float(d)) * p_power
			weights = expanded
			if sym:
				weights = np.maximum(weights, weights.T)

	if topk is None:
		topk = 0

	src_list = []
	dst_list = []
	w_list = []

	for r in range(num_rel_types):
		row = weights[r]
		pos = np.flatnonzero(row > 0)
		if pos.size == 0:
			continue
		if topk > 0 and pos.size > topk:
			keep = pos[np.argpartition(row[pos], -topk)[-topk:]]
			keep = keep[np.argsort(row[keep])[::-1]]
		else:
			keep = pos[np.argsort(row[pos])[::-1]]

		src_list.append(np.full(keep.size, r, dtype=np.int64))
		dst_list.append(keep.astype(np.int64, copy=False))
		w_list.append(row[keep].astype(np.float32, copy=False))

	if not src_list:
		rel_edge_index = torch.empty((2, 0), dtype=torch.long)
		rel_edge_weight = torch.empty((0,), dtype=torch.float)
		return rel_edge_index, rel_edge_weight

	src_cat = np.concatenate(src_list, axis=0)
	dst_cat = np.concatenate(dst_list, axis=0)
	w_cat = np.concatenate(w_list, axis=0)

	rel_edge_index = torch.from_numpy(np.stack([src_cat, dst_cat], axis=0)).long()
	rel_edge_weight = torch.from_numpy(w_cat).float()
	return rel_edge_index, rel_edge_weight
