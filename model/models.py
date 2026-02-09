from helper import *
from model.hogrn_conv import HoGRNConv
from model.edge_selector import EdgeSelector
from model.ib_loss import IBLoss, StochasticEncoder


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

		# ========== 信息瓶颈模块 ==========
		self.use_ib = getattr(self.p, 'use_ib', False)
		if self.use_ib:
			# 边选择器
			edge_hidden = getattr(self.p, 'edge_selector_hidden', self.p.init_dim)
			self.edge_selector = EdgeSelector(self.p.init_dim, edge_hidden)

			# 随机编码器（用于 KL 损失）
			self.stochastic_encoder = StochasticEncoder(self.p.embed_dim, self.p.embed_dim)

			# IB 损失
			ib_beta = getattr(self.p, 'ib_beta', 0.01)
			polar_weight = getattr(self.p, 'polar_weight', 0.1)
			self.ib_loss = IBLoss(beta=ib_beta, polar_weight=polar_weight)

	def _edge_sampling(self, edge_index, edge_type, rate=0.5):
		n_edges = edge_index.shape[1]
		random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
		return edge_index[:, random_indices], edge_type[random_indices]

	def _compute_edge_weight(self, edge_index, edge_type):
		"""
		计算边权重（信息瓶颈软过滤）

		Args:
			edge_index: 边索引 [2, num_edges]
			edge_type: 边类型 [num_edges]

		Returns:
			edge_prob: 边重要性概率 [num_edges, 1]
		"""
		src, dst = edge_index
		h_src = self.init_embed[src]  # [num_edges, dim]
		h_dst = self.init_embed[dst]  # [num_edges, dim]

		# 获取关系嵌入
		r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		r_edge = r[edge_type]  # [num_edges, dim]

		# 计算边重要性
		edge_prob = self.edge_selector(h_src, h_dst, r_edge)

		return edge_prob

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

		# ========== 计算边权重（如果启用 IB）==========
		edge_weight = None
		edge_prob = None
		if self.use_ib:
			edge_prob = self._compute_edge_weight(edge_index, edge_type)
			edge_weight = edge_prob  # 直接用概率作为权重

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

		# 第一层 GCN（带边权重）
		x, r	= self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight)
		x	= drop1(x)

		# 后续层（同样带边权重）
		x, r	= self.conv2(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 2 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 2 else x
		x, r	= self.conv3(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 3 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 3 else x
		x, r	= self.conv4(x, edge_index, edge_type, rel_embed=r, edge_weight=edge_weight) if self.p.gcn_layer >= 4 else (x, r)
		x	= drop2(x) if self.p.gcn_layer >= 4 else x

		# ========== 随机编码（如果启用 IB）==========
		mu, logvar = None, None
		if self.use_ib:
			x, mu, logvar = self.stochastic_encoder(x, training=self.training)

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.p.sim_decay > 0:
			cor = self._cul_cor(r)
		else:
			cor = 0.

		# 返回额外的 IB 信息
		ib_info = {
			'mu': mu,
			'logvar': logvar,
			'edge_prob': edge_prob
		}

		return sub_emb, rel_emb, x, cor, ib_info

	def compute_ib_loss(self, ib_info):
		"""
		计算信息瓶颈损失

		Args:
			ib_info: forward_base 返回的 IB 信息

		Returns:
			ib_loss: 标量损失
			loss_dict: 损失详情
		"""
		if not self.use_ib or ib_info['mu'] is None:
			return 0., {}

		return self.ib_loss(
			ib_info['mu'],
			ib_info['logvar'],
			ib_info['edge_prob']
		)


class HoGRN_TransE(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent, cor, ib_info = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb + rel_emb

		x		= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score	= torch.sigmoid(x)

		return score, cor, ib_info


class HoGRN_DistMult(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent, cor, ib_info = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb * rel_emb

		x 	= torch.mm(obj_emb, all_ent.transpose(1, 0))
		x 	+= self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score, cor, ib_info


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
		sub_emb, rel_emb, all_ent, cor, ib_info = self.forward_base(sub, rel, self.hidden_drop, self.hidden_drop)
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
		return score, cor, ib_info
