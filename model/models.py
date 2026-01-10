from re import X
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
		self.disen_k		= int(getattr(self.p, 'disen_k', 1))
		if self.disen_k < 1:
			raise ValueError('disen_k must be >= 1')
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.device			= self.edge_index.device

		if self.disen_k == 1:
			self.init_embed	= get_param((self.p.num_ent, self.p.init_dim))
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 								self.init_rel = get_param((num_rel*2, self.p.init_dim))

			self.conv1 = HoGRNConv(self.p.init_dim, 	self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 2 else None
			self.conv3 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 3 else None
			self.conv4 = HoGRNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 4 else None
		else:
			for dim_name in ['init_dim', 'gcn_dim', 'embed_dim']:
				dim_val = int(getattr(self.p, dim_name))
				if dim_val % self.disen_k != 0:
					raise ValueError('{} must be divisible by disen_k ({} % {} != 0)'.format(dim_name, dim_val, self.disen_k))

			self.init_dim_k		= self.p.init_dim  // self.disen_k
			self.gcn_dim_k		= self.p.gcn_dim   // self.disen_k
			self.embed_dim_k	= self.p.embed_dim // self.disen_k

			self.init_embed		= get_param((self.p.num_ent, self.disen_k, self.init_dim_k))
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.disen_k, self.init_dim_k))
			else: 								self.init_rel = get_param((num_rel*2, self.disen_k, self.init_dim_k))

			self.gate_w			= get_param((self.disen_k, self.embed_dim_k))

			self.conv1 = HoGRNConv(self.init_dim_k, self.gcn_dim_k,   num_rel, act=self.act, params=self.p)
			self.conv2 = HoGRNConv(self.gcn_dim_k,  self.embed_dim_k, num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 2 else None
			self.conv3 = HoGRNConv(self.gcn_dim_k,  self.embed_dim_k, num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 3 else None
			self.conv4 = HoGRNConv(self.gcn_dim_k,  self.embed_dim_k, num_rel, act=self.act, params=self.p) if self.p.gcn_layer >= 4 else None

		if self.p.rel_drop > 0:
			self.drop_rel	= torch.nn.Dropout(self.p.rel_drop)

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def _edge_sampling(self, edge_index, edge_type, rate=0.5):
		n_edges = edge_index.shape[1]
		random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
		return edge_index[:, random_indices], edge_type[random_indices]

	def _cul_cor(self, rel):
		if rel.dim() > 2:
			rel = rel.view(rel.size(0), -1)
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

	def _get_channel_alpha(self, rel_emb):
		"""
		Compute per-sample channel weights alpha for disentangled channels.

		rel_emb: (B, K, D)
		return:  (B, K)
		"""
		if self.disen_k == 1:
			return None

		gate_type = str(getattr(self.p, 'disen_gate', 'rel')).lower()
		if gate_type == 'uniform':
			return rel_emb.new_full((rel_emb.size(0), self.disen_k), 1.0 / float(self.disen_k))
		if gate_type != 'rel':
			raise ValueError('Unsupported disen_gate: {}'.format(gate_type))

		logits = (rel_emb * self.gate_w.view(1, self.disen_k, -1)).sum(dim=-1)
		return torch.softmax(logits, dim=1)

	def _channel_indep(self, x):
		"""
		Simple channel independence regularizer.
		x: (N, K, D)
		"""
		if self.disen_k == 1 or x.dim() != 3:
			return x.new_tensor(0.0) if torch.is_tensor(x) else 0.0

		x = x - x.mean(dim=0, keepdim=True)
		x = F.normalize(x, p=2, dim=-1)
		corr = torch.einsum('nkd,njd->kj', x, x) / float(x.size(0) * x.size(2))
		off_diag = corr - torch.diag(torch.diag(corr))
		return (off_diag * off_diag).sum()

	def forward_base(self, sub, rel, drop1, drop2):
		if self.p.edge_drop > 0:
			edge_index, edge_type = self._edge_sampling(self.edge_index, self.edge_type, self.p.edge_drop)
		else:
			edge_index, edge_type = self.edge_index, self.edge_type

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 2 else x
		x, r	= self.conv3(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 3 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 3 else x
		x, r	= self.conv4(x, edge_index, edge_type, rel_embed=r) 	if self.p.gcn_layer >= 4 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer >= 4 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.training and self.p.sim_decay > 0:
			cor = self._cul_cor(r)
		else:
			cor = 0.

		return sub_emb, rel_emb, x, cor

class HoGRN_TransE(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, rel_cor = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb + rel_emb

		reg = obj_emb.new_tensor(0.0)
		if self.training:
			if getattr(self.p, 'sim_decay', 0) > 0:
				reg = reg + float(self.p.sim_decay) * rel_cor
			disen_indep = float(getattr(self.p, 'disen_indep', 0))
			if disen_indep > 0 and self.disen_k > 1:
				reg = reg + disen_indep * self._channel_indep(all_ent)

		if self.disen_k == 1:
			x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		else:
			alpha = self._get_channel_alpha(rel_emb)
			all_ent_knd = all_ent.permute(1, 0, 2)  # (K, N, D)
			num_ent = all_ent.size(0)
			x = obj_emb.new_empty((obj_emb.size(0), num_ent))

			chunk_size = int(getattr(self.p, 'transe_chunk', 0))
			if chunk_size <= 0:
				chunk_size = 4096

			for start in range(0, num_ent, chunk_size):
				end = min(start + chunk_size, num_ent)
				ent_chunk = all_ent_knd[:, start:end, :]  # (K, chunk, D)
				diff = obj_emb.unsqueeze(2) - ent_chunk.unsqueeze(0)  # (B, K, chunk, D)
				dist_k = diff.abs().sum(dim=-1)  # (B, K, chunk)
				dist = (dist_k * alpha.unsqueeze(-1)).sum(dim=1)  # (B, chunk)
				x[:, start:end] = self.p.gamma - dist

		score = torch.sigmoid(x)

		return score, reg

class HoGRN_DistMult(HoGRNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, rel_cor = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb	= sub_emb * rel_emb

		reg = obj_emb.new_tensor(0.0)
		if self.training:
			if getattr(self.p, 'sim_decay', 0) > 0:
				reg = reg + float(self.p.sim_decay) * rel_cor
			disen_indep = float(getattr(self.p, 'disen_indep', 0))
			if disen_indep > 0 and self.disen_k > 1:
				reg = reg + disen_indep * self._channel_indep(all_ent)

		if self.disen_k == 1:
			x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		else:
			alpha = self._get_channel_alpha(rel_emb)  # (B, K)
			x = obj_emb.new_zeros((obj_emb.size(0), all_ent.size(0)))
			for k in range(self.disen_k):
				x = x + alpha[:, k].unsqueeze(1) * torch.mm(
					obj_emb[:, k, :], all_ent[:, k, :].transpose(1, 0)
				)

		x = x + self.bias.expand_as(x)
		score = torch.sigmoid(x)
		return score, reg

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
		sub_emb, rel_emb, all_ent, rel_cor	= self.forward_base(sub, rel, self.hidden_drop, self.hidden_drop)

		reg = sub_emb.new_tensor(0.0)
		if self.training:
			if getattr(self.p, 'sim_decay', 0) > 0:
				reg = reg + float(self.p.sim_decay) * rel_cor
			disen_indep = float(getattr(self.p, 'disen_indep', 0))
			if disen_indep > 0 and self.disen_k > 1:
				reg = reg + disen_indep * self._channel_indep(all_ent)

		# Scheme A for disentanglement: gate channels, then flatten back to embed_dim and reuse vanilla ConvE.
		if self.disen_k == 1:
			sub_flat, rel_flat, all_ent_flat = sub_emb, rel_emb, all_ent
		else:
			alpha = self._get_channel_alpha(rel_emb)  # (B, K)
			sub_flat = (sub_emb * alpha.unsqueeze(-1)).reshape(sub_emb.size(0), -1)
			rel_flat = (rel_emb * alpha.unsqueeze(-1)).reshape(rel_emb.size(0), -1)
			all_ent_flat = all_ent.reshape(all_ent.size(0), -1)

		stk_inp	= self.concat(sub_flat, rel_flat)
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

		x 		= torch.mm(x, all_ent_flat.transpose(1,0))
		x 		+= self.bias.expand_as(x)

		score	= torch.sigmoid(x)
		return score, reg
