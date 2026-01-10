from helper import *
from data_loader import *
from model.models import *
from dual_graph import build_static_relation_graph
import json
from pathlib import Path


def _apply_json_config(parser, cfg, config_path):
	if not isinstance(cfg, dict):
		raise ValueError(f'Config must be a JSON object: {config_path}')

	known_dests = {a.dest for a in parser._actions}
	key_to_dest = {}
	for action in parser._actions:
		for opt in action.option_strings:
			if opt.startswith('-'):
				key_to_dest[opt.lstrip('-')] = action.dest

	overrides = {}
	unknown_keys = []
	for key, value in cfg.items():
		if key == '_entry':
			continue
		dest = key_to_dest.get(key)
		if dest is None and key in known_dests:
			dest = key
		if dest is None:
			unknown_keys.append(key)
			continue
		overrides[dest] = value

	if unknown_keys:
		print(f'[config] Warning: unknown keys ignored in {config_path}: {sorted(unknown_keys)}')
	parser.set_defaults(**overrides)

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
		self.rel_edge_index, self.rel_edge_weight = None, None
		if getattr(self.p, 'dual_rel', False):
			t_rel = time.time()
			self.logger.info('DualRel: Building static relation graph (walk={}, weight={}, hops={}, topk={}, min_count={}, sym={}, alpha={}, drop={})'.format(
				getattr(self.p, 'dual_walk', 'path'),
				getattr(self.p, 'dual_weight', 'ppmi'),
				getattr(self.p, 'dual_hops', 2),
				getattr(self.p, 'dual_topk', 20),
				getattr(self.p, 'dual_min_count', 1),
				getattr(self.p, 'dual_sym', True),
				getattr(self.p, 'dual_alpha', 0.1),
				getattr(self.p, 'dual_drop', 0.0),
			))
			self.rel_edge_index, self.rel_edge_weight = build_static_relation_graph(
				self.edge_index,
				self.edge_type,
				self.p.num_ent,
				self.p.num_rel * 2,
				walk=getattr(self.p, 'dual_walk', 'path'),
				weight=getattr(self.p, 'dual_weight', 'ppmi'),
				hops=getattr(self.p, 'dual_hops', 2),
				topk=getattr(self.p, 'dual_topk', 20),
				min_count=getattr(self.p, 'dual_min_count', 1),
				sym=getattr(self.p, 'dual_sym', True),
			)
			self.rel_edge_index = self.rel_edge_index.to(self.device)
			self.rel_edge_weight = self.rel_edge_weight.to(self.device)

			num_rel_nodes = self.p.num_rel * 2
			num_rel_edges = int(self.rel_edge_index.size(1))
			avg_deg = float(num_rel_edges) / float(max(1, num_rel_nodes))
			if num_rel_edges > 0:
				w_cpu = self.rel_edge_weight.detach().float().cpu()
				w_min = float(w_cpu.min().item())
				w_max = float(w_cpu.max().item())
				w_mean = float(w_cpu.mean().item())
				w_std = float(w_cpu.std(unbiased=False).item())
			else:
				w_min, w_max, w_mean, w_std = 0.0, 0.0, 0.0, 0.0
			self.logger.info('DualRel: Relation graph ready: nodes={}, edges={}, avg_deg={:.3f}, w[min/mean/std/max]={:.4f}/{:.4f}/{:.4f}/{:.4f}, build_time={:.3f}s'.format(
				num_rel_nodes, num_rel_edges, avg_deg, w_min, w_mean, w_std, w_max, time.time() - t_rel
			))

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

		if   model_name.lower()	== 'hogrn_transe': 		model = HoGRN_TransE(self.edge_index, self.edge_type, self.rel_edge_index, self.rel_edge_weight, params=self.p)
		elif model_name.lower()	== 'hogrn_distmult': 	model = HoGRN_DistMult(self.edge_index, self.edge_type, self.rel_edge_index, self.rel_edge_weight, params=self.p)
		elif model_name.lower()	== 'hogrn_conve': 		model = HoGRN_ConvE(self.edge_index, self.edge_type, self.rel_edge_index, self.rel_edge_weight, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)
		print("Model have {:.4f}M paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))
		return model

	def add_optimizer(self, parameters):
		"""
		Create an optimizer for training the parameters
		"""
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

	def evaluate(self, split):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		
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
		self.logger.info('MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_mrr'], results['right_mrr'], results['mrr']))
		self.logger.info('MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_mr'], results['right_mr'], results['mr']))
		# for k in range(10):
		# 	self.logger.info('[Epoch {} {}]: Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))
		if split == 'test':
			for k in range(10):
				self.logger.info('Hit@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(k+1, results['left_hits@{}'.format(k+1)], results['right_hits@{}'.format(k+1)], results['hits@{}'.format(k+1)]))
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
				pred, _			= self.model.forward(sub, rel)
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

				# if step % 100 == 0:
				# 	self.logger.info('[{}, {} Step {}]'.format(split.title(), mode.title(), step))

		return results
	

	def fit(self):
		"""
		Function to run training and evaluation of model.
		"""
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test')
		self.logger.info('Test Avg MRR: {:.5}'.format(test_results['mrr']))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config',	dest='config_path',	default=None, help='Path to a JSON experiment config (exp_configs/*.json).')

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

	# Dual-graph (relation graph) settings
	parser.add_argument('-dual_rel',		dest='dual_rel',		action='store_true',	help='Enable dual relation graph message passing')
	parser.add_argument('-dual_walk',		dest='dual_walk',		default='path',	choices=['path', 'share_src', 'share_dst', 'mixed'], help='Relation co-occurrence pattern')
	parser.add_argument('-dual_weight',	dest='dual_weight',	default='ppmi',	choices=['count', 'pmi', 'ppmi'], help='Edge weight type for relation graph')
	parser.add_argument('-dual_hops',		dest='dual_hops',		type=int,	default=2,	help='Max hop length for relation-graph diffusion (>=2); 2 disables diffusion')
	parser.add_argument('-dual_topk',		dest='dual_topk',		type=int,	default=20,	help='Top-k neighbors per relation type in relation graph')
	parser.add_argument('-dual_min_count', dest='dual_min_count',	type=int,	default=1,	help='Minimum co-occurrence count for relation graph edges')
	parser.add_argument('-dual_asym',		dest='dual_sym',		action='store_false',	help='Do not symmetrize the relation graph')
	parser.set_defaults(dual_sym=True)
	parser.add_argument('-dual_alpha',	dest='dual_alpha',	type=float,	default=0.1,	help='Residual step size for dual relation update')
	parser.add_argument('-dual_drop',		dest='dual_drop',		type=float,	default=0.0,	help='Dropout in dual relation update')

	# Relational contrastive loss
	parser.add_argument('-temperature', dest='temperature', default=1,  	type=float,	help='temperature coefficient')
	parser.add_argument('-sim_decay',	dest='sim_decay',	default=0,		type=float, help='Regularization weight for independence modeling')
	parser.add_argument('-rel_drop',  	dest='rel_drop', 	default=0,  	type=float,	help='Dropout for generate positive relation')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  		dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  		dest='k_h', 		default=10,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=32,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=3,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',		dest='log_dir',		default='./log/',		help='Log directory')
	parser.add_argument('-config',		dest='config_dir',	default='./config/',	help='Config directory')

	pre_args, _ = parser.parse_known_args()
	if pre_args.config_path:
		config_path = Path(pre_args.config_path)
		cfg = json.loads(config_path.read_text(encoding='utf-8'))
		entry = cfg.get('_entry')
		if entry and entry != 'restore.py':
			print(f'[config] Warning: config _entry={entry!r} but running restore.py')
		_apply_json_config(parser, cfg, str(config_path))
		print(f'[config] Loaded: {config_path}')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H-%M-%S')

	# set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
	# torch.autograd.set_detect_anomaly(True)
	
	model = Runner(args)
	model.fit()
