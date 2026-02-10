from helper import *
from model.hogrn_conv import HoGRNConv
from model.causal_discovery import CausalDiscovery
from model.intervention import GumbelIntervention
from model.causal_loss import CausalSparsityLoss


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
		self.num_rel		= num_rel

		if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
		else: 								self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.rel_drop > 0:
			self.drop_rel	= torch.nn.Dropout(self.p.rel_drop)

		self.conv1 = HoGRNConv(self.p.init_dim, 	self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
		self.conv2 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 2 else None
		self.conv3 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 3 else None
		self.conv4 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 4 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

		# ========== 因果结构学习模块 ==========
		self.use_causal = getattr(self.p, 'use_causal', False)
		if self.use_causal:
			# 因果发现模块
			causal_hidden = getattr(self.p, 'causal_hidden', self.p.init_dim)
			self.causal_discovery = CausalDiscovery(
				dim=self.p.init_dim,
				hidden_dim=causal_hidden,
				num_rels=num_rel
			)

			# Gumbel-Softmax 边干预模块
			causal_base = getattr(self.p, 'causal_base', 0.3)
			causal_scale = getattr(self.p, 'causal_scale', 0.7)
			temp_init = getattr(self.p, 'causal_temp_init', 1.0)
			temp_min = getattr(self.p, 'causal_temp_min', 0.1)
			self.gumbel_intervention = GumbelIntervention(
				base_weight=causal_base,
				scale=causal_scale,
				init_temperature=temp_init,
				min_temperature=temp_min
			)

			# 因果稀疏损失
			causal_sparse = getattr(self.p, 'causal_sparse', 0.0001)
			warmup_epochs = getattr(self.p, 'causal_warmup', 30)
			target_sparsity = getattr(self.p, 'causal_target', 0.5)
			self.sparsity_loss = CausalSparsityLoss(
				lambda_sparse=causal_sparse,
				warmup_epochs=warmup_epochs,
				target_sparsity=target_sparsity
			)

			# 预计算节点度数
			self._precompute_degrees()

	def _precompute_degrees(self):
		"""预计算节点度数"""
		num_edges = self.edge_index.size(1)
		src = self.edge_index[0]
		dst = self.edge_index[1]

		# 计算出度和入度
		self.src_degrees = torch.zeros(self.p.num_ent, device=self.device)
		self.dst_degrees = torch.zeros(self.p.num_ent, device=self.device)

		self.src_degrees.scatter_add_(0, src, torch.ones(num_edges, device=self.device))
		self.dst_degrees.scatter_add_(0, dst, torch.ones(num_edges, device=self.device))

	def _compute_causal_scores(self, edge_index, edge_type):
		"""
		计算边的因果分数

		Args:
			edge_index: 边索引 [2, num_edges]
			edge_type: 边类型 [num_edges]

		Returns:
			causal_scores: 因果分数 [num_edges, 1]
			causal_logits: 因果 logits [num_edges, 1]（sigmoid 之前）
		"""
		src, dst = edge_index
		h_src = self.init_embed[src]  # [num_edges, dim]
		h_dst = self.init_embed[dst]  # [num_edges, dim]

		# 获取关系嵌入
		r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		r_edge = r[edge_type]  # [num_edges, dim]

		# 获取度数特征
		src_deg = self.src_degrees[src]
		dst_deg = self.dst_degrees[dst]

		# 计算因果分数（返回 scores 和 logits）
		causal_scores, causal_logits = self.causal_discovery(
			h_src, h_dst, r_edge,
			edge_type=edge_type,
			src_deg=src_deg,
			dst_deg=dst_deg
		)

		return causal_scores, causal_logits

	def _edge_sampling(self, edge_index, edge_type, rate=0.5):
		n_edges = edge_index.shape[1]
		random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
		return edge_index[:, random_indices], edge_type[random_indices]

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

	def forward_base(self, sub, rel, drop1, drop2, edge_weight=None):
		"""
		基础前向传播

		Args:
			sub: 主语实体索引
			rel: 关系索引
			drop1: 第一层 dropout
			drop2: 后续层 dropout
			edge_weight: 边权重（因果加权），可选
		"""
		if self.p.edge_drop > 0:
			edge_index, edge_type = self._edge_sampling(self.edge_index, self.edge_type, self.p.edge_drop)
		else:
			edge_index, edge_type = self.edge_index, self.edge_type

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

		# 第一层 GCN（带边权重）
		x, r	= self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight)
		x	= drop1(x)

		# 后续层
		x, r	= self.conv2(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 2 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 2 else x
		x, r	= self.conv3(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 3 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 3 else x
		x, r	= self.conv4(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 4 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 4 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.p.sim_decay > 0:
			cor = self._cul_cor(r)
		else:
			cor = 0.

		return sub_emb, rel_emb, x, cor

	def compute_causal_loss(self, causal_info):
		"""
		计算因果稀疏损失

		Args:
			causal_info: 因果信息字典

		Returns:
			sparse_loss: 稀疏损失
			loss_dict: 损失详情
		"""
		if not self.use_causal or causal_info.get('causal_scores') is None:
			return torch.tensor(0.0, device=self.device), {}

		causal_scores = causal_info['causal_scores']
		causal_logits = causal_info.get('causal_logits')

		sparse_loss, loss_dict = self.sparsity_loss(causal_scores, logits=causal_logits)

		# edge_weight 统计
		edge_weight = causal_info.get('edge_weight')
		z = causal_info.get('z')
		if edge_weight is not None:
			ew = edge_weight.squeeze()
			loss_dict['ew_mean'] = ew.mean().item()
			loss_dict['ew_std'] = ew.std().item()
			loss_dict['ew_min'] = ew.min().item()
			loss_dict['ew_max'] = ew.max().item()
		if z is not None:
			zf = z.squeeze()
			loss_dict['z_mean'] = zf.mean().item()
			loss_dict['z_std'] = zf.std().item()

		# Temperature
		loss_dict['temperature'] = self.gumbel_intervention.temperature

		return sparse_loss, loss_dict


class HoGRN_TransE(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		causal_scores = None
		causal_logits = None
		edge_weight = None
		z = None

		if self.use_causal:
			causal_scores, causal_logits = self._compute_causal_scores(self.edge_index, self.edge_type)
			edge_weight, z = self.gumbel_intervention(causal_logits)

		sub_emb, rel_emb, all_ent, cor = self.forward_base(sub, rel, self.drop, self.drop, edge_weight=edge_weight)
		obj_emb	= sub_emb + rel_emb
		x		= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score	= torch.sigmoid(x)

		causal_info = {
			'causal_scores': causal_scores, 'causal_logits': causal_logits,
			'edge_weight': edge_weight, 'z': z
		}
		return score, cor, causal_info


class HoGRN_DistMult(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		causal_scores = None
		causal_logits = None
		edge_weight = None
		z = None

		if self.use_causal:
			causal_scores, causal_logits = self._compute_causal_scores(self.edge_index, self.edge_type)
			edge_weight, z = self.gumbel_intervention(causal_logits)

		sub_emb, rel_emb, all_ent, cor = self.forward_base(sub, rel, self.drop, self.drop, edge_weight=edge_weight)
		obj_emb	= sub_emb * rel_emb
		x 	= torch.mm(obj_emb, all_ent.transpose(1, 0))
		x 	+= self.bias.expand_as(x)
		score = torch.sigmoid(x)

		causal_info = {
			'causal_scores': causal_scores, 'causal_logits': causal_logits,
			'edge_weight': edge_weight, 'z': z
		}
		return score, cor, causal_info


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

	def _score_func(self, sub_emb, rel_emb, all_ent):
		"""ConvE 评分函数"""
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
		return score

	def forward(self, sub, rel):
		# 计算因果分数（如果启用）
		causal_scores = None
		causal_logits = None
		edge_weight = None
		z = None

		if self.use_causal:
			causal_scores, causal_logits = self._compute_causal_scores(self.edge_index, self.edge_type)
			edge_weight, z = self.gumbel_intervention(causal_logits)

		# 前向传播（只需要 1 次，不需要反事实！）
		sub_emb, rel_emb, all_ent, cor = self.forward_base(sub, rel, self.hidden_drop, self.hidden_drop, edge_weight=edge_weight)
		score = self._score_func(sub_emb, rel_emb, all_ent)

		causal_info = {
			'causal_scores': causal_scores, 'causal_logits': causal_logits,
			'edge_weight': edge_weight, 'z': z
		}
		return score, cor, causal_info
