import torch


def sinkhorn_distance(cost, a, b, eps=0.1, iters=20):
	"""
	Batched Sinkhorn distance in log-space for stability.
	Args:
		cost: Tensor [N, K, K] pairwise costs
		a:    Tensor [N, K] source weights (sum to 1)
		b:    Tensor [N, K] target weights (sum to 1)
	"""
	log_a = torch.log(a + 1e-12)
	log_b = torch.log(b + 1e-12)
	log_K = -cost / eps  # [N, K, K]

	u = torch.zeros_like(log_a)
	v = torch.zeros_like(log_b)

	for _ in range(iters):
		u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
		v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)

	log_P = log_K + u.unsqueeze(2) + v.unsqueeze(1)
	P = torch.exp(log_P)
	return torch.sum(P * cost, dim=(1, 2))
