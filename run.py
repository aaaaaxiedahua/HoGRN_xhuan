from helper import *
from data_loader import *
from model.models import *

class Runner(object):

	def load_data(self):
		"""
		Read in raw triplets and convert them into a standard format.
		"""

		# Build the mapping table from all the data
		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				if self.p.dataset == 'FB15k-237' or self.p.dataset == 'WN18RR':
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
				else:
					sub, obj, rel = map(str.lower, line.strip().split('\t')[:3])
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		print("Dataset: ", self.p.dataset)
		print("NUM_ENT: ", self.p.num_ent)
		print("NUM_REL: ", self.p.num_rel)
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		# Use UIDs to represent entities and relationships in the data, and inverse relationships are used to expand the training set
		self.data = ddict(list)
		sr2o = ddict(set)
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				if self.p.dataset == 'FB15k-237' or self.p.dataset == 'WN18RR':
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
				else:
					sub, obj, rel = map(str.lower, line.strip().split('\t')[:3])
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train':
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()} # train
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()} # train+valid+test

		self.triples  = ddict(list)
		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.edge_index, self.edge_type = self.construct_adj()

	def construct_adj(self):
		"""
		Construct the adjacency matrix for GCN.
		"""
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def __init__(self, params):
		"""
		Constructor of the runner class.
		"""
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())

	def add_model(self, model, score_func):
		"""
		Create the computational graph.
		"""
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'hogrn_transe': 		model = HoGRN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'hogrn_distmult': 	model = HoGRN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'hogrn_conve': 		model = HoGRN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)
		print("Model have {:.4f}M paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))
		return model

	def add_optimizer(self, parameters):
		"""
		Create an optimizer for training the parameters
		因果发现模块使用更高的学习率（梯度路径长，信号衰减严重）
		"""
		if getattr(self.p, 'use_causal', False):
			causal_lr = getattr(self.p, 'causal_lr', self.p.lr * 100)
			main_params = [p for n, p in self.model.named_parameters() if 'causal_discovery' not in n]
			causal_params = [p for n, p in self.model.named_parameters() if 'causal_discovery' in n]
			print(f"Causal LR: {causal_lr}, Main LR: {self.p.lr}, Causal params: {sum(p.numel() for p in causal_params)}")
			return torch.optim.Adam([
				{'params': main_params, 'lr': self.p.lr},
				{'params': causal_params, 'lr': causal_lr}
			], weight_decay=self.p.l2)
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'		: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'		: self.optimizer.state_dict(),
			'args'			: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model
		"""
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count

		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'], results['right_mr'], results['mr']))
		if split == 'test':
			for k in range(10):
				self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode
		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred, _, _	= self.model.forward(sub, rel)  # 增加了 causal_info 返回值
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
				ranks 			= ranks.float()

				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

		return results

	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function to run one epoch of training
		"""
		self.model.train()
		losses = []
		causal_losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')

			pred, cor, causal_info = self.model.forward(sub, rel)
			loss = self.model.loss(pred, label)

			if self.p.sim_decay > 0:
				loss += self.p.sim_decay * cor

			# 添加因果损失（对比学习版）
			if getattr(self.p, 'use_causal', False):
				# 设置当前 epoch（用于 warmup）
				self.model.causal_loss.set_epoch(epoch)

				# 计算因果损失
				cf_score = causal_info.get('cf_score')
				if cf_score is not None:
					# 计算反事实损失（BCE）
					cf_loss = self.model.loss(cf_score, label)

					# 对比因果损失：原始损失应该小于反事实损失
					# 新接口：直接传入两个损失值
					causal_loss, causal_loss_dict = self.model.compute_causal_loss(
						causal_info, loss, cf_loss  # 传入 loss 而非 pred
					)
					loss = loss + causal_loss  # 使用 = 而非 +=，确保计算图正确
					causal_losses.append(causal_loss_dict)

			loss.backward()

			# 每 50 个 step 检查一次梯度
			if step % 50 == 0 and getattr(self.p, 'use_causal', False):
				# 检查因果发现模块的梯度
				grad_info = []
				for name, param in self.model.causal_discovery.named_parameters():
					if param.grad is not None:
						grad_norm = param.grad.norm().item()
						grad_info.append(f"{name}:{grad_norm:.6f}")
				if grad_info and epoch % 5 == 0:
					self.logger.info('[Epoch:{} Step:{}] Causal Grads: {}'.format(
						epoch, step, ', '.join(grad_info[:4])))  # 只显示前4个

			self.optimizer.step()
			losses.append(loss.item())

		loss = np.mean(losses)

		# 记录因果损失信息
		if getattr(self.p, 'use_causal', False) and len(causal_losses) > 0:
			avg_contrast = np.mean([d.get('contrastive_loss', 0) for d in causal_losses])
			avg_reg = np.mean([d.get('reg_loss', 0) for d in causal_losses])
			avg_orig_loss = np.mean([d.get('original_loss', 0) for d in causal_losses])
			avg_cf_loss = np.mean([d.get('cf_loss', 0) for d in causal_losses])
			avg_loss_diff = np.mean([d.get('loss_diff', 0) for d in causal_losses])
			avg_warmup = np.mean([d.get('warmup', 0) for d in causal_losses])

			# 因果分数统计
			avg_cs_mean = np.mean([d.get('cs_mean', 0) for d in causal_losses])
			avg_cs_std = np.mean([d.get('cs_std', 0) for d in causal_losses])
			avg_cs_min = np.mean([d.get('cs_min', 0) for d in causal_losses])
			avg_cs_max = np.mean([d.get('cs_max', 0) for d in causal_losses])
			avg_cs_below = np.mean([d.get('cs_below_0.3', 0) for d in causal_losses])
			avg_cs_above = np.mean([d.get('cs_above_0.7', 0) for d in causal_losses])
			avg_cs_mid = np.mean([d.get('cs_mid_range', 0) for d in causal_losses])

			# Logit 统计
			avg_logit_mean = np.mean([d.get('logit_mean', 0) for d in causal_losses])
			avg_logit_std = np.mean([d.get('logit_std', 0) for d in causal_losses])

			self.logger.info('[Epoch:{}]:  Loss:{:.4}, Contrast:{:.4}, Reg:{:.4}'.format(
				epoch, loss, avg_contrast, avg_reg))
			self.logger.info('[Epoch:{}]:  Orig:{:.4}, CF:{:.4}, Diff:{:.4}, Warmup:{:.2f}'.format(
				epoch, avg_orig_loss, avg_cf_loss, avg_loss_diff, avg_warmup))
			self.logger.info('[Epoch:{}]:  CS: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
				epoch, avg_cs_mean, avg_cs_std, avg_cs_min, avg_cs_max))
			self.logger.info('[Epoch:{}]:  Logits: mean={:.4f}, std={:.4f}'.format(
				epoch, avg_logit_mean, avg_logit_std))
			self.logger.info('[Epoch:{}]:  CS Distribution: <0.3={:.1%}, 0.3-0.7={:.1%}, >0.7={:.1%}\n'.format(
				epoch, avg_cs_below, avg_cs_mid, avg_cs_above))

			# 每 5 个 epoch 记录更详细的统计
			if epoch % 5 == 0:
				# Edge weight 统计
				avg_ew_mean = np.mean([d.get('ew_mean', 0) for d in causal_losses])
				avg_ew_std = np.mean([d.get('ew_std', 0) for d in causal_losses])
				avg_ew_min = np.mean([d.get('ew_min', 0) for d in causal_losses])
				avg_ew_max = np.mean([d.get('ew_max', 0) for d in causal_losses])
				avg_cw_mean = np.mean([d.get('cw_mean', 0) for d in causal_losses])

				self.logger.info('[Epoch:{}]:  EdgeWeight: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
					epoch, avg_ew_mean, avg_ew_std, avg_ew_min, avg_ew_max))
				self.logger.info('[Epoch:{}]:  CF_Weight: mean={:.4f}'.format(epoch, avg_cw_mean))

				if causal_info.get('causal_scores') is not None:
					causal_stats = self.model.causal_discovery.get_stats(causal_info['causal_scores'])
					self.logger.info('[Epoch:{}]:  Detailed Stats: mean={:.4f}, std={:.4f}, high_ratio={:.3f}, very_high={:.3f}, very_low={:.3f}'.format(
						epoch, causal_stats['causal_mean'], causal_stats['causal_std'],
						causal_stats['high_causal_ratio'], causal_stats['very_high_ratio'], causal_stats['very_low_ratio']))
		else:
			self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))

		return loss

	def fit(self):
		"""
		Function to run training and evaluation of model.
		"""
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			print("########")
			t0 = time.time()
			train_loss  = self.run_epoch(epoch, val_mrr)
			print("Time cost in one epoch for training: {:.4f}s".format((time.time()-t0)/60))

			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25:
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)
		self.logger.info('Test Avg MRR: {:.5}'.format(test_results['mrr']))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Config file support
	parser.add_argument('-config_file',	dest='config_file',	default=None,		help='Path to JSON config file (e.g., exp_configs/nell23k_conve.json)')

	parser.add_argument('-name',		dest='name',		default='testrun',		help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',		default='FB15K-237-10',	help='Dataset to use.')
	parser.add_argument('-model',		dest='model',		default='hogrn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',         dest='opn',			default='mult',			help='Composition Operation to be used in HoGRN')

	parser.add_argument('-batch',       dest='batch_size',	type=int, 	default=128,	help='Batch size')
	parser.add_argument('-epoch',		dest='max_epochs',	type=int,	default=9999,  	help='Number of epochs')
	parser.add_argument('-gamma',		dest='gamma',		type=float,	default=40,		help='Margin')
	parser.add_argument('-gpu',			type=str,			default='0',				help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

	parser.add_argument('-l2',			type=float,	default=0,		help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',			type=float,	default=0.001,	help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',	type=float,	default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,	default=2,		help='Number of processes to construct batches')
	parser.add_argument('-seed',		type=int,	default=41504, 	help='Seed for randomization')

	parser.add_argument('-restore',     dest='restore',		action='store_true',	help='Restore from the previously saved model')
	parser.add_argument('-bias',		dest='bias',		action='store_true',	help='Whether to use bias in the model')

	parser.add_argument('-rel_reason', 	dest='rel_reason',	action='store_true',	help='Whether to optimize the relation representation by relation reasoning')
	parser.add_argument('-pre_reason', 	dest='pre_reason',	action='store_true',	help='Whether to use the relation reasoning firstly')
	parser.add_argument('-reason_type', dest='reason_type',	default='mixdrop',		help='Relation Reason Operation to be used in HoGRN')
	parser.add_argument('-act_type', 	dest='act_type',	default='tanh',			help='Activation funtion to be used in HoGRN')
	parser.add_argument('-rel_norm', 	dest='rel_norm',	action='store_true',	help='Whether to optimize the relation representation by normalization')

	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=100,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=100,   	type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0,  	type=float,	help='Dropout after GCN')
	parser.add_argument('-relmix_dim',	dest='relmix_dim',	default=200,	type=int,	help='Number of hidden units in inter-relation learning')
	parser.add_argument('-chamix_dim',	dest='chamix_dim', 	default=200,  	type=int, 	help='Number of hidden units in intra-relation learning')
	parser.add_argument('-rel_mask',  	dest='rel_mask', 	default=0,  	type=float,	help='Dropout in inter-relation learning')
	parser.add_argument('-chan_drop',  	dest='chan_drop', 	default=0,  	type=float,	help='Dropout in intra-relation learning')
	parser.add_argument('-edge_drop',  	dest='edge_drop', 	default=0,  	type=float,	help='Dropout in edge')

	# Relational contrastive loss
	parser.add_argument('-temperature', dest='temperature', default=1,  	type=float,	help='temperature coefficient')
	parser.add_argument('-sim_decay',	dest='sim_decay',	default=0,		type=float, help='Regularization weight for independence modeling')
	parser.add_argument('-rel_drop',  	dest='rel_drop', 	default=0,  	type=float,	help='Dropout for generate positive relation')

	# Causal Structure Learning parameters
	parser.add_argument('-use_causal',      dest='use_causal',      action='store_true',    help='Enable Causal Structure Learning')
	parser.add_argument('-causal_alpha',    dest='causal_alpha',    default=0.5,   type=float, help='Weight for contrastive loss (core)')
	parser.add_argument('-causal_beta',     dest='causal_beta',     default=0.1,   type=float, help='Reserved for compatibility')
	parser.add_argument('-causal_gamma',    dest='causal_gamma',    default=0.0001,type=float, help='Weight for separation loss (auxiliary)')
	parser.add_argument('-causal_margin',   dest='causal_margin',   default=0.1,   type=float, help='Margin for contrastive loss')
	parser.add_argument('-causal_base',     dest='causal_base',     default=0.3,   type=float, help='Base weight for residual connection')
	parser.add_argument('-causal_scale',    dest='causal_scale',    default=0.7,   type=float, help='Scale for causal score in edge weight')
	parser.add_argument('-causal_hidden',   dest='causal_hidden',   default=100,   type=int,   help='Hidden dim for causal discovery')
	parser.add_argument('-causal_warmup',   dest='causal_warmup',   default=15,    type=int,   help='Warmup epochs for causal loss')
	parser.add_argument('-causal_lr',       dest='causal_lr',       default=0.001, type=float, help='Learning rate for causal discovery module')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  		dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  		dest='k_h', 		default=10,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=32,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=3,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',		dest='log_dir',		default='./log/',		help='Log directory')
	parser.add_argument('-config',		dest='config_dir',	default='./config/',	help='Config directory')
	args = parser.parse_args()

	# Load config file if specified
	if args.config_file is not None:
		import json
		print(f"Loading config from: {args.config_file}")
		with open(args.config_file, 'r', encoding='utf-8') as f:
			config = json.load(f)

		# Override args with config values (command line args take precedence)
		for key, value in config.items():
			# Skip special keys starting with underscore
			if key.startswith('_'):
				continue

			# Convert key format: batch -> batch_size
			if key == 'batch':
				key = 'batch_size'
			elif key == 'data':
				key = 'dataset'
			elif key == 'epoch':
				key = 'max_epochs'
			elif key == 'gcn_drop':
				key = 'dropout'

			# Only override if not explicitly set in command line
			if hasattr(args, key):
				# Check if this was set by user or is default value
				default_value = parser.get_default(key)
				current_value = getattr(args, key)

				# Override only if current value is default
				if current_value == default_value:
					setattr(args, key, value)
			else:
				# Add new attribute if it doesn't exist
				setattr(args, key, value)

		print(f"✓ Config loaded successfully")

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H-%M-%S')

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

	model = Runner(args)
	model.fit()
