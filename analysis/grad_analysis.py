from utils.generic import *
from main.distributions import Poisson, GumbelSoftmaxPoisson


def exact_loss_fn(log_rate_input, phi, x):
	"""Exact Poisson reconstruction loss (Eq 24)."""
	lam = torch.exp(log_rate_input)
	recon = x - (lam @ phi.T)
	mse_term = (recon ** 2).sum()
	phi_norm_sq = (phi ** 2).sum(dim=0)
	var_term = (lam * phi_norm_sq).sum()
	return mse_term + var_term


def exact_loss_grad_hessian(log_rate, phi, x):
	"""
	Computes loss, exact gradient, and exact Hessian w.r.t log_rate analytically.
	Handles batched inputs correctly.

	Args:
	   log_rate: (B, K) parameter
	   phi: (M, K) decoder weights
	   x: (B, M) input

	Returns:
	   loss: scalar (sum over batch)
	   grad: (B, K)
	   hessian: (B*K, B*K) block diagonal matrix
	"""
	# Ensure shapes are consistent (B, K) and (B, D)
	if log_rate.dim() == 1: log_rate = log_rate.unsqueeze(0)
	if x.dim() == 1: x = x.unsqueeze(0)

	# 1. Precomputations
	lam = torch.exp(log_rate)  # (B, K)

	# G = phi.T @ phi is used in both grad and hessian
	# G (gram_mat) is shared across the batch
	gram_mat = phi.T @ phi  # (K, K)

	# 2. Forward / Loss Terms
	recon = x - (lam @ phi.T)  # (B, D)
	d = torch.diagonal(gram_mat)  # (K,)

	# 3. Exact Gradient
	# Vectorized computation for the batch
	# grad_lambda = 2 * lam @ G - 2 * x @ phi + d
	term1 = 2 * (lam @ gram_mat)  # (B, K)
	term2 = 2 * (x @ phi)  # (B, K)

	grad_lambda = term1 - term2 + d  # (B, K) broadcast d

	# Gradient w.r.t log_rate = lambda * grad_lambda
	grad_log = lam * grad_lambda  # (B, K)

	# 4. Exact Hessian
	# For a batch, the total Hessian is block diagonal.
	# We compute the blocks (B, K, K) first.

	# Term 1: diag(grad_log) per sample
	# Shape: (B, K, K)
	hessian_term1 = torch.diag_embed(grad_log)

	# Term 2: 2 * Lambda_i @ G @ Lambda_i
	# We use broadcasting to compute this for all b in B efficiently.
	# lam.unsqueeze(2) is (B, K, 1)
	# G.unsqueeze(0)   is (1, K, K)
	# lam.unsqueeze(1) is (B, 1, K)
	# Result corresponds to scaling rows by lam and cols by lam
	hessian_term2 = 2 * lam.unsqueeze(2) * gram_mat.unsqueeze(0) * lam.unsqueeze(1)

	# Per-sample Hessian blocks
	hessian_blocks = hessian_term1 + hessian_term2  # (B, K, K)

	# Construct full (B*K, B*K) block diagonal matrix
	# This matches the output shape of torch.autograd.functional.hessian
	hessian = torch.block_diag(*hessian_blocks)

	# Recompute total loss
	mse_term = (recon ** 2).sum()
	var_term = (lam * d).sum()
	loss = mse_term + var_term

	return loss, grad_log, hessian


def get_ground_truth_grad(lambda_val, phi, x):
	"""
	Computes EXACT gradient of Recon Loss w.r.t log_rate using Autograd.
	"""
	log_rate = lambda_val.log().detach().clone().requires_grad_(True)
	loss = exact_loss_fn(log_rate, phi, x)
	return torch.autograd.grad(loss, log_rate)[0]


def compute_exact_hessian(log_rate_fixed, phi, x):
	"""Compute flattened Hessian of the loss w.r.t. log_rate using Autograd."""
	loss_fn = lambda log_rate: exact_loss_fn(log_rate, phi, x)
	hessian_exact = torch.autograd.functional.hessian(
		loss_fn, log_rate_fixed)
	bk = log_rate_fixed.numel()
	return hessian_exact.view(bk, bk)


def verify_analytical_vs_autograd(log_rate, phi, x, atol=1e-5):
	"""Debug utility to verify analytical expressions."""
	_, grad_analytic, hess_analytic = exact_loss_grad_hessian(log_rate, phi, x)
	grad_auto = get_ground_truth_grad(torch.exp(log_rate), phi, x)
	hess_auto = compute_exact_hessian(log_rate, phi, x)

	grad_ok = torch.allclose(grad_analytic, grad_auto, atol=atol)
	hess_ok = torch.allclose(hess_analytic, hess_auto, atol=atol)

	print(f"Gradient match: {grad_ok}, Hessian match: {hess_ok}")
	if not grad_ok:
		print(f"  Grad diff: {(grad_analytic - grad_auto).abs().max():.2e}")
	if not hess_ok:
		print(f"  Hess diff: {(hess_analytic - hess_auto).abs().max():.2e}")

	return grad_ok and hess_ok


def compute_gradient_statistics(grads_tensor, g_star, hessian_flat, normalize=True):
	"""
	Compute all gradient statistics from sampled gradients.
	"""
	n_samples = grads_tensor.shape[0]

	g_bar = grads_tensor.mean(dim=0)
	g_star_norm = g_star.norm().item()
	g_star_norm_sq = g_star_norm ** 2

	# --- Raw Metrics ---
	bias_l2 = (g_bar - g_star).norm().item()
	var_total = torch.var(grads_tensor, dim=0).sum().item()
	cos_sim = (g_bar * g_star).sum() / (g_bar.norm() * g_star.norm() + 1e-9)

	# Per-sample cosine similarities
	grads_flat = grads_tensor.view(n_samples, -1)
	g_star_flat = g_star.view(-1)
	dots = grads_flat @ g_star_flat
	grad_norms = grads_flat.norm(dim=1)
	per_sample_cos = dots / (grad_norms * g_star_flat.norm() + 1e-9)
	cos_sim_mean = per_sample_cos.mean().item()
	cos_sim_std = per_sample_cos.std().item()

	# Hessian-weighted Metrics
	diff_vec = (g_bar - g_star).view(-1)
	bias_energy = (diff_vec @ hessian_flat @ diff_vec).item()

	grads_centered = grads_flat - grads_flat.mean(dim=0)
	sigma = (grads_centered.T @ grads_centered) / (n_samples - 1)
	noise_energy = torch.trace(hessian_flat @ sigma).item()

	# Signal energy: g*^T H g* (natural scale)
	signal_energy = (g_star_flat @ hessian_flat @ g_star_flat).item()

	# --- Normalization ---
	if normalize:
		bias_l2 = bias_l2 / (g_star_norm + 1e-9)
		var_total = var_total / (g_star_norm_sq + 1e-9)
		# Normalize energies by signal energy (unitless ratios)
		bias_energy_ratio = bias_energy / (signal_energy + 1e-9)
		noise_energy_ratio = noise_energy / (signal_energy + 1e-9)
		# Normalzied var_total is simply the inverse of SNR
		snr = 1.0 / (var_total + 1e-9)
	else:
		bias_energy_ratio = bias_energy
		noise_energy_ratio = noise_energy
		snr = g_star_norm_sq / (var_total + 1e-9)

	return {
	    'Bias': bias_l2,
	    'Variance': var_total,
	    'SNR': snr,
	    'CosSim': cos_sim.item(),
	    'CosSimMean': cos_sim_mean,
	    'CosSimStd': cos_sim_std,
	    'BiasEnergy': bias_energy_ratio,
	    'NoiseEnergy': noise_energy_ratio,
	    'SignalEnergy': signal_energy,  # for reference
	}


def sample_eat_gradients(lambda_fixed, phi, x, tau, indicator_approx, n_samples):
	b, k = lambda_fixed.shape
	log_rate_base = lambda_fixed.log().detach()
	log_rate_expanded = log_rate_base.unsqueeze(0).expand(n_samples, b, k).clone()
	log_rate_expanded.requires_grad_(True)
	log_rate_flat = log_rate_expanded.reshape(n_samples * b, k)

	dist = Poisson(
	    log_rate=log_rate_flat,
	    temp=tau,
	    indicator_approx=indicator_approx,
	    n_exp='infer',
	)
	z_flat = dist.rsample()
	z = z_flat.reshape(n_samples, b, k)

	x_recon = z @ phi.T
	losses = ((x.unsqueeze(0) - x_recon) ** 2).sum(dim=(1, 2))

	# Single backward pass
	# (gradients are block-diagonal)
	grads = torch.autograd.grad(
		outputs=losses.sum(),
		inputs=log_rate_expanded,
	)[0]
	return grads


def sample_gs_gradients(lambda_fixed, phi, x, tau, n_samples):
	b, k = lambda_fixed.shape
	rate_mag = lambda_fixed.mean().item()
	upperbound_safe = int(rate_mag + 4 * (rate_mag ** 0.5) + 5)

	log_rate_base = lambda_fixed.log().detach()
	log_rate_expanded = log_rate_base.unsqueeze(0).expand(n_samples, b, k).clone()
	log_rate_expanded.requires_grad_(True)
	log_rate_flat = log_rate_expanded.reshape(n_samples * b, k)

	dist = GumbelSoftmaxPoisson(
	    log_rate=log_rate_flat,
	    temp=tau,
	    upperbound_method='fixed',
	    upperbound_param=upperbound_safe,
	)
	samples_flat = dist.rsample()
	z_soft_flat = dist.aggregate_samples(samples_flat)
	z_soft = z_soft_flat.reshape(n_samples, b, k)

	x_recon = z_soft @ phi.T
	losses = ((x.unsqueeze(0) - x_recon) ** 2).sum(dim=(1, 2))

	# Single backward pass
	# (gradients are block-diagonal)
	grads = torch.autograd.grad(
		outputs=losses.sum(),
		inputs=log_rate_expanded,
	)[0]
	return grads


def run_experiment(
		x: torch.Tensor,
		phi: torch.Tensor,
		n_samples: int = 100,
		temperatures: list = None,
		rates_to_test: list = None,
		exact_grad_hessian: bool = True, ):
	"""
	Sweep over firing rates and temperatures, comparing EAT vs Gumbel-Softmax.

	Args:
		x: Input data
		phi: Decoder weights
		n_samples: Number of gradient samples
		temperatures: obvious
		rates_to_test: obvious
		exact_grad_hessian: If True, uses analytic expressions for Ground Truth.
							If False, uses Autograd (slower).
	"""
	temperatures = temperatures or [
		1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
	rates_to_test = rates_to_test or [
		0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
	results = []

	print(f"Starting Sweep. n_samples={n_samples}. Analytic Mode: {exact_grad_hessian}")

	for rate_mag in tqdm(rates_to_test):
		# Setup for this rate
		lambda_fixed = torch.ones(x.shape[0], phi.shape[1], device=x.device) * rate_mag
		log_rate_fixed = lambda_fixed.log().detach().clone().requires_grad_(True)

		# Compute Ground Truth (Gradient + Hessian)
		if exact_grad_hessian:
			# Fast analytic computation
			_, g_star, hessian_flat = exact_loss_grad_hessian(log_rate_fixed, phi, x)
		else:
			# Slow autograd computation
			g_star = get_ground_truth_grad(lambda_fixed, phi, x)
			hessian_flat = compute_exact_hessian(log_rate_fixed, phi, x)

		for tau in tqdm(temperatures, leave=False):
			# --- Exponential Arrival Time (EAT) Methods ---
			for indicator_approx in ['sigmoid', 'cubic']:
				grads = sample_eat_gradients(
					lambda_fixed, phi, x, tau,
					indicator_approx, n_samples
				)
				stats = compute_gradient_statistics(
					grads, g_star, hessian_flat)
				results.append({
					'Method': f'EAT_{indicator_approx}',
					'Rate': rate_mag,
					'Temp': tau,
					**stats,
				})

			# --- Gumbel-Softmax Method---
			grads = sample_gs_gradients(
				lambda_fixed, phi, x, tau, n_samples)
			stats = compute_gradient_statistics(
				grads, g_star, hessian_flat)
			results.append({
				'Method': 'GS',
				'Rate': rate_mag,
				'Temp': tau,
				**stats,
			})

	return pd.DataFrame(results)
