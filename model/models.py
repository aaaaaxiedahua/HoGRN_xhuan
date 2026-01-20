from re import X
import torch.nn as nn
from helper import *
from model.hogrn_conv import HoGRNConv

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p			= params
		self.act		= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class HoGRNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(HoGRNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device			= self.edge_index.device

		if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
		else: 								self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.rel_drop > 0:
			self.drop_rel	= torch.nn.Dropout(self.p.rel_drop)

		self.conv1 = HoGRNConv(self.p.init_dim, 	self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
		self.conv2 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 2 else None
		self.conv3 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 3 else None
		self.conv4 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 4 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		self.register_buffer('node_deg', self._compute_node_degree())

		# ===== GloMem-HoGRN: Global Memory Enhancement =====
		if hasattr(self.p, 'use_global_memory') and self.p.use_global_memory:
			from model.global_memory import GlobalWriteModule, GlobalReadModule, MultiHeadGlobalMemory
			mem_dim = self.p.gcn_dim

			if hasattr(self.p, 'global_memory_heads') and self.p.global_memory_heads > 1:
				# Multi-head global memory
				self.global_memory_module = MultiHeadGlobalMemory(
					dim=mem_dim,
					num_heads=self.p.global_memory_heads,
					attention_type=getattr(self.p, 'global_attention_type', 'concat'),
					gate_type=getattr(self.p, 'global_gate_type', 'mlp'),
					extra_input_dim=1
				)
				print(f"[GloMem] Multi-head mode enabled with {self.p.global_memory_heads} heads")
			else:
				# Single global memory
				self.global_memory = get_param((1, mem_dim))
				self.global_write = GlobalWriteModule(
					dim=mem_dim,
					attention_type=getattr(self.p, 'global_attention_type', 'concat')
				)
				self.global_read = GlobalReadModule(
					dim=mem_dim,
					gate_type=getattr(self.p, 'global_gate_type', 'mlp'),
					use_residual=getattr(self.p, 'global_use_residual', False),
					extra_input_dim=1
				)
				print(f"[GloMem] Single-head mode enabled")

			# For storing intermediate values for analysis
			self.last_write_attention = None
			self.last_read_gates = None
		# ===== VC-HoGRN: Virtual Centroid Enhancement =====
		if hasattr(self.p, 'use_virtual_centroid') and self.p.use_virtual_centroid:
			vc_dim = self.p.gcn_dim
			self.vc_gate = nn.Sequential(
				nn.Linear(vc_dim * 2 + 1, vc_dim),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(vc_dim, 1),
				nn.Sigmoid()
			)
			self.last_vc_gates = None

	def _edge_sampling(self, edge_index, edge_type, rate=0.5):
		n_edges = edge_index.shape[1]
		random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
		return edge_index[:, random_indices], edge_type[random_indices]

	def _compute_node_degree(self):
		row = self.edge_index[0]
		edge_weight = torch.ones_like(row, dtype=torch.float)
		deg = scatter_add(edge_weight, row, dim=0, dim_size=self.p.num_ent)
		deg = torch.log1p(deg)
		return deg / (deg.max() + 1e-12)

	def _apply_virtual_centroid(self, x, edge_index, edge_type):
		num_edges = edge_index.size(1) // 2
		fwd_index = edge_index[:, :num_edges]
		fwd_type = edge_type[:num_edges]

		tails = fwd_index[1]
		rel_sum = scatter_add(x[tails], fwd_type, dim=0, dim_size=self.p.num_rel)
		rel_cnt = scatter_add(
			torch.ones_like(fwd_type, dtype=torch.float).unsqueeze(1),
			fwd_type,
			dim=0,
			dim_size=self.p.num_rel
		)
		rel_centroid = rel_sum / (rel_cnt + 1e-12)

		head = fwd_index[0]
		head_rel = rel_centroid[fwd_type]
		head_sum = scatter_add(head_rel, head, dim=0, dim_size=self.p.num_ent)
		head_cnt = scatter_add(
			torch.ones_like(head, dtype=torch.float).unsqueeze(1),
			head,
			dim=0,
			dim_size=self.p.num_ent
		)
		proto = head_sum / (head_cnt + 1e-12)

		deg_input = self.node_deg.unsqueeze(1)
		gate_inp = torch.cat([x, proto, deg_input], dim=1)
		beta = self.vc_gate(gate_inp)
		has_rel = (head_cnt > 0).float()
		beta = beta * has_rel
		x = (1 - beta) * x + beta * proto
		return x, beta.squeeze(1)

	def _cul_cor(self, rel):
		if self.p.rel_drop > 0:
			rel_pos = self.drop_rel(rel)
		else:
			rel_pos = rel
		norm_rel_pos = rel_pos / rel_pos.norm(p=2, dim=1, keepdim=True)

		norm_rel 	= rel / rel.norm(p=2, dim=1, keepdim=True) # (num_rel, num_dim)

		pos_smi 	= torch.sum(norm_rel * norm_rel_pos, dim=1) # (num_rel, 1)
		ttl_smi 	= torch.mm(norm_rel, norm_rel.T) # (num_rel, num_rel)

		pos_scores 	= torch.exp(pos_smi / self.p.temperature)
		ttl_scores 	= torch.exp(ttl_smi / self.p.temperature)  # (num_rel, num_rel)
		semi_scores = torch.exp(torch.ones(rel.shape[0]) / self.p.temperature).to(self.device)

		ttl_scores 	= torch.sum(ttl_scores, dim=1) # (num_rel, 1)
		ttl_scores 	= ttl_scores - semi_scores + pos_scores

		mi_score 	= - torch.sum(torch.log(pos_scores / ttl_scores))

		return mi_score

	def forward_base(self, sub, rel, drop1, drop2):
		if self.p.edge_drop > 0:
			edge_index, edge_type = self._edge_sampling(self.edge_index, self.edge_type, self.p.edge_drop)
		else:
			edge_index, edge_type = self.edge_index, self.edge_type

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r)
		x	= drop1(x)

		# ===== VC-HoGRN: Virtual Centroid Enhancement =====
		if hasattr(self.p, 'use_virtual_centroid') and self.p.use_virtual_centroid:
			x, vc_gates = self._apply_virtual_centroid(x, edge_index, edge_type)
			if self.training:
				self.last_vc_gates = vc_gates

		# ===== GloMem-HoGRN: Global Memory Enhancement =====
		if hasattr(self.p, 'use_global_memory') and self.p.use_global_memory:
			deg_input = self.node_deg.unsqueeze(1)
			rel_ctx = torch.index_select(r, 0, rel).mean(dim=0, keepdim=True)
			if hasattr(self, 'global_memory_module'):
				# Multi-head global memory
				x, read_gates = self.global_memory_module(
					x,
					extra_features=deg_input,
					relation_context=rel_ctx
				)
			else:
				# Single global memory: Write-Read cycle
				g_new, write_attention = self.global_write(
					self.global_memory,
					x,
					relation_context=rel_ctx
				)
				x, read_gates = self.global_read(
					x,
					g_new,
					extra_features=deg_input,
					relation_context=rel_ctx
				)

				# Store for analysis
				if self.training:
					self.last_write_attention = write_attention
					self.last_read_gates = read_gates
		x, r	= self.conv2(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 2 else x
		x, r	= self.conv3(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 3 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 3 else x
		x, r	= self.conv4(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 4 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 4 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.p.sim_decay > 0:
			cor = self._cul_cor(r)
		else:
			cor = 0.

		return sub_emb, rel_emb, x, cor

class HoGRN_TransE(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, cor	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb + rel_emb

		x		= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score, cor

class HoGRN_DistMult(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, cor	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb * rel_emb

		x 	= torch.mm(obj_emb, all_ent.transpose(1, 0))
		x 	+= self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score, cor

class HoGRN_ConvE(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0	= torch.nn.BatchNorm2d(1)
		self.bn1	= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2	= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz	= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, cor	= self.forward_base(sub, rel, self.hidden_drop, self.hidden_drop)
		stk_inp	= self.concat(sub_emb, rel_emb)
		x		= self.bn0(stk_inp)
		x		= self.m_conv1(x)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_drop(x)
		x		= x.view(-1, self.flat_sz)
		x		= self.fc(x)
		x		= self.hidden_drop2(x)
		x		= self.bn2(x)
		x		= F.relu(x)

		x 		= torch.mm(x, all_ent.transpose(1,0))
		x 		+= self.bias.expand_as(x)

		score	= torch.sigmoid(x)
		return score, cor
