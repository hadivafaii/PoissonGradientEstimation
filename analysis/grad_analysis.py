from . import *


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
	Computes analytical loss, exact gradient,
	and exact Hessian blocks w.r.t log_rate.

	Args:
	   log_rate: (B, K) parameter
	   phi: (M, K) decoder weights
	   x: (B, M) input

	Returns:
	   loss: scalar (sum over batch)
	   grad_log: (B, K)
	   hessian_blocks: (B, K, K)
	"""
	# Ensure shapes are consistent (B, K) and (B, D)
	if log_rate.dim() == 1: log_rate = log_rate.unsqueeze(0)
	if x.dim() == 1: x = x.unsqueeze(0)

	# 1. Precomputations
	lam = torch.exp(log_rate)  # (B, K)
	gram_mat = phi.T @ phi  # (K, K)

	# 2. Forward / Loss Terms
	recon = x - (lam @ phi.T)  # (B, D)
	d = torch.diagonal(gram_mat)  # (K,)

	# 3. Exact Gradient
	# grad_lambda = 2 * lam @ G - 2 * x @ phi + d
	term1 = 2 * (lam @ gram_mat)  # (B, K)
	term2 = 2 * (x @ phi)  # (B, K)
	grad_lambda = term1 - term2 + d  # (B, K) broadcast d

	# Gradient w.r.t log_rate = lambda * grad_lambda
	grad_log = lam * grad_lambda  # (B, K)

	# 4. Exact Hessian Blocks (B, K, K)
	# Term 1: diag(grad_log) per sample -> (B, K, K)
	hessian_term1 = torch.diag_embed(grad_log)

	# Term 2: 2 * Lambda_i @ G @ Lambda_i
	# Broadcasting: (B, K, 1) * (1, K, K) * (B, 1, K)
	hessian_term2 = (
		2 * lam.unsqueeze(2) *
		gram_mat.unsqueeze(0) *
		lam.unsqueeze(1)
	)

	# Sum blocks
	hessian_blocks = hessian_term1 + hessian_term2  # (B, K, K)

	# Recompute total loss
	mse_term = (recon ** 2).sum()
	var_term = (lam * d).sum()
	loss = mse_term + var_term

	return loss, grad_log, hessian_blocks


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
	_, grad_analytic, hess_analytic_blocks = exact_loss_grad_hessian(log_rate, phi, x)
	grad_auto = get_ground_truth_grad(torch.exp(log_rate), phi, x)
	hess_auto_flat = compute_exact_hessian(log_rate, phi, x)

	# Convert blocks (B, K, K) to block-diagonal (B*K, B*K) for comparison
	hess_analytic_flat = torch.block_diag(*hess_analytic_blocks)

	grad_ok = torch.allclose(grad_analytic, grad_auto, atol=atol)
	hess_ok = torch.allclose(hess_analytic_flat, hess_auto_flat, atol=atol)

	print(f"Gradient match: {grad_ok}, Hessian match: {hess_ok}")
	if not grad_ok:
		print(f"  Grad diff: {(grad_analytic - grad_auto).abs().max():.2e}")
	if not hess_ok:
		print(f"  Hess diff: {(hess_analytic_flat - hess_auto_flat).abs().max():.2e}")

	return grad_ok and hess_ok


def verify_score_gradient(lambda_fixed, phi, x, n_samples=100_000):
	"""Verify score gradient is unbiased by checking against exact gradient."""
	grads = sample_score_gradients(lambda_fixed, phi, x, n_samples)
	g_bar = grads.mean(dim=0)

	log_rate = lambda_fixed.log().detach().requires_grad_(True)
	_, g_star, _ = exact_loss_grad_hessian(log_rate, phi, x)

	# Score function should be unbiased (g_bar ≈ g_star for large N)
	bias = (g_bar - g_star).abs().mean()
	return bias


def compute_gradient_statistics(grads_tensor, g_star, hessian_blocks, normalize=True):
	"""
	Compute all gradient statistics from sampled gradients using Block Hessian.
	All metrics are computed per-batch to avoid magnitude-weighted mixing,
	then summarized with mean and std.

	Args:
		grads_tensor: (N, B, K) sampled gradients
		g_star: (B, K) true gradients
		hessian_blocks: (B, K, K) exact Hessian blocks
		normalize: bool, whether to normalize metrics by signal strength

	Returns:
		Dictionary with the following metrics:

		Standard Metrics (per-batch, then aggregated):
		-----------------------------------------------
		BiasMean/Std     : L2 norm of (g_bar_b - g*_b), normalized by ||g*_b||.
		                   Measures systematic error per batch.
		                   → 0 is ideal (unbiased).

		VarianceMean/Std : Sum of per-coordinate variances for batch b,
		                   normalized by ||g*_b||².
		                   → 0 is ideal (deterministic).

		SNRMean/Std      : Signal-to-noise ratio per batch.
		                   → Higher is better.

		Cosine Similarity (per-batch, then aggregated):
		------------------------------------------------
		CosMean/Std      : cos(g_bar_b, g*_b) per batch.
		                   Directional accuracy of the expected gradient.
		                   → 1 = perfect, 0 = orthogonal, <0 = anti-aligned.

		CosSampleMean/Std: cos(g_nb, g*_b) per (sample, batch).
		                   Directional accuracy of individual samples.
		                   → Typically ≤ CosMean (averaging helps).

		Hessian-Weighted Energy (per-batch, then aggregated):
		-----------------------------------------------------
		BiasEnergyMean/Std  : (b_b^T H_b b_b) / (g*_b^T H_b g*_b) per batch.
		                      Fraction of optimization signal lost to bias.
		                      → 0 is ideal.

		NoiseEnergyMean/Std : Tr(H_b Sigma_b) / (g*_b^T H_b g*_b) per batch.
		                      Fraction of optimization signal lost to variance.
		                      → 0 is ideal.

		SignalEnergyMean/Std: g*_b^T H_b g*_b per batch (for reference).

		Summary:
		--------
		| Metric           | Question It Answers                              |
		|------------------|--------------------------------------------------|
		| BiasMean/Std     | "How far is mean gradient from truth?"           |
		| VarianceMean/Std | "How noisy are gradient samples?"                |
		| SNRMean/Std      | "What's the signal-to-noise ratio?"              |
		| CosMean/Std      | "Does the average gradient point right?"         |
		| CosSampleMean/Std| "Do individual samples point right?"             |
		| BiasEnergyMean/Std  | "How much does bias hurt optimization?"       |
		| NoiseEnergyMean/Std | "How much does variance hurt optimization?"   |
	"""
	g_bar = grads_tensor.mean(dim=0)  # (B, K)

	# === Per-Batch Norms ===
	g_star_norm = g_star.norm(dim=-1)  # (B,)
	g_star_norm_sq = g_star_norm ** 2  # (B,)

	# === Standard Metrics (Per-Batch) ===
	bias_vec = g_bar - g_star  # (B, K)
	bias_l2 = bias_vec.norm(dim=-1)  # (B,)
	var_per_batch = torch.var(grads_tensor, dim=0).sum(dim=-1)  # (B,)

	if normalize:
		bias_l2 = bias_l2 / (g_star_norm + EPS)
		var_per_batch = var_per_batch / (g_star_norm_sq + EPS)
		snr_per_batch = 1.0 / (var_per_batch + EPS)
	else:
		snr_per_batch = g_star_norm_sq / (var_per_batch + EPS)

	# === Cosine Similarity (Per-Batch) ===
	norms_bar = g_bar.norm(dim=-1)  # (B,)

	# Cosine of mean gradient per batch: (B,)
	dots_mean = (g_bar * g_star).sum(dim=-1)
	cos_of_mean = dots_mean / (norms_bar * g_star_norm + EPS)

	# Per-sample cosine per batch: (N, B)
	dots_samples = (grads_tensor * g_star.unsqueeze(0)).sum(dim=-1)
	norms_samples = grads_tensor.norm(dim=-1)
	cos_per_sample = dots_samples / (norms_samples * g_star_norm.unsqueeze(0) + EPS)

	# === Hessian Energy Metrics (Per-Batch) ===

	# Signal energy per batch: (B,)
	signal_energy = torch.einsum('bi,bij,bj->b', g_star, hessian_blocks, g_star)

	# Bias energy per batch: (B,)
	bias_energy = torch.einsum('bi,bij,bj->b', bias_vec, hessian_blocks, bias_vec)

	# Noise energy per batch: (B,)
	grads_centered = grads_tensor - g_bar  # (N, B, K)
	noise_energy_per_sample = torch.einsum(
		'nbi,bij,nbj->nb',
		grads_centered,
		hessian_blocks,
		grads_centered,
	)  # (N, B)
	noise_energy = (  # (B,)
		noise_energy_per_sample.sum(dim=0) /
		max(grads_tensor.shape[0] - 1, 1)
	)

	# Normalize by signal energy per batch
	if normalize:
		bias_energy_ratio = bias_energy / (signal_energy + EPS)
		noise_energy_ratio = noise_energy / (signal_energy + EPS)
	else:
		bias_energy_ratio = bias_energy
		noise_energy_ratio = noise_energy

	return {
		# Standard metrics
		'BiasMean': bias_l2.mean().item(),
		'BiasStd': bias_l2.std().item(),
		'VarianceMean': var_per_batch.mean().item(),
		'VarianceStd': var_per_batch.std().item(),
		'SNRMean': snr_per_batch.mean().item(),
		'SNRStd': snr_per_batch.std().item(),
		# Cosine of mean gradient
		'CosMean': cos_of_mean.mean().item(),
		'CosStd': cos_of_mean.std().item(),
		# Per-sample cosine
		'CosSampleMean': cos_per_sample.mean().item(),
		'CosSampleStd': cos_per_sample.std().item(),
		# Hessian-weighted energy
		'BiasEnergyMean': bias_energy_ratio.mean().item(),
		'BiasEnergyStd': bias_energy_ratio.std().item(),
		'NoiseEnergyMean': noise_energy_ratio.mean().item(),
		'NoiseEnergyStd': noise_energy_ratio.std().item(),
		'SignalEnergyMean': signal_energy.mean().item(),
		'SignalEnergyStd': signal_energy.std().item(),
	}


def sample_eat_gradients(lambda_fixed, phi, x, tau, indicator_approx, n_samples):
	b, k = lambda_fixed.shape
	log_rate_base = lambda_fixed.log().detach()
	log_rate_expanded = (
		log_rate_base.unsqueeze(0)
	    .expand(n_samples, b, k).clone()
	)
	log_rate_expanded.requires_grad_(True)
	log_rate_flat = log_rate_expanded.reshape(
		n_samples * b, k)

	dist = Poisson(
	    log_rate=log_rate_flat,
	    indicator_approx=indicator_approx,
		temp=tau,
		n_exp=N_EXP,
		n_exp_p=N_EXP_P,
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


def sample_score_gradients(lambda_fixed, phi, x, n_samples, baseline=None):
	b, k = lambda_fixed.shape
	log_rate_base = lambda_fixed.log().detach()
	log_rate_expanded = (
		log_rate_base.unsqueeze(0)
		.expand(n_samples, b, k).clone()
	)
	log_rate_expanded.requires_grad_(True)
	log_rate_flat = log_rate_expanded.reshape(
		n_samples * b, k)

	dist = Poisson(log_rate=log_rate_flat)

	z_flat = dist.sample()
	z = z_flat.reshape(n_samples, b, k)

	x_recon = z @ phi.T
	losses = ((x.unsqueeze(0) - x_recon) ** 2).sum(dim=2)

	if baseline is None:
		baseline = losses.mean(dim=0, keepdim=True).detach()

	advantages = losses - baseline  # Shape: (N, B)

	log_probs = dist.log_prob(z_flat).reshape(n_samples, b, k).sum(dim=2)
	surrogate = advantages.detach() * log_probs

	# Single backward pass
	# (chain rule maintains separation between batch items)
	grads = torch.autograd.grad(
		outputs=surrogate.sum(),
		inputs=log_rate_expanded,
	)[0]
	return grads


def run_gradient_analysis(
		x: torch.Tensor,
		phi: torch.Tensor,
		n_samples: int = 100,
		temperatures: list = None,
		rates_to_test: list = None, ):
	"""
	Sweep over firing rates and temperatures, comparing EAT vs Gumbel-Softmax.

	Args:
		x: Input data, (B, M)
		phi: Decoder weights, (M, K)
		n_samples: Number of gradient samples
		temperatures: see default values above
		rates_to_test: see default values above
	"""
	temperatures = temperatures or DEFAULT_TEMPERATURES
	rates_to_test = rates_to_test or DEFAULT_RATES

	print(
		f"Starting Sweep "
		f"(n_samples={n_samples}, batch_size={len(x)})\n\n"
		f"firing rates:\n{rates_to_test}\n\n"
		f"temperatures:\n{temperatures}\n"
	)

	clear_gpu_memory()  # Initial cleanup

	results = []
	for rate_mag in tqdm(rates_to_test):
		# Setup for this rate
		lambda_fixed = torch.ones(
			size=(x.shape[0], phi.shape[1]),
			device=x.device,
			dtype=x.dtype,
		) * rate_mag
		log_rate_fixed = lambda_fixed.log().detach().clone()

		# Compute Ground Truth (Analytic Mode Only)
		# Returns hessian_blocks (B, K, K)
		_, g_star, hessian_blocks = exact_loss_grad_hessian(
			log_rate_fixed, phi, x)

		for tau in tqdm(temperatures, leave=False):
			# --- Exponential Arrival Time (EAT) Methods ---
			for indicator_approx in ['sigmoid', 'cubic', 'quintic']:
				grads = sample_eat_gradients(
					lambda_fixed, phi, x, tau,
					indicator_approx, n_samples
				)
				stats = compute_gradient_statistics(
					grads, g_star, hessian_blocks)
				results.append({
					'Method': f'EAT_{indicator_approx}',
					'Rate': rate_mag,
					'Temp': tau,
					**stats,
				})
				del grads
				clear_gpu_memory()

			# --- Gumbel-Softmax Method---
			grads = sample_gs_gradients(
				lambda_fixed, phi, x, tau, n_samples)
			stats = compute_gradient_statistics(
				grads, g_star, hessian_blocks)
			results.append({
				'Method': 'GS',
				'Rate': rate_mag,
				'Temp': tau,
				**stats,
			})
			del grads
			clear_gpu_memory()

		# --- Score-Based Method ---
		grads = sample_score_gradients(
			lambda_fixed, phi, x, n_samples)
		stats = compute_gradient_statistics(
			grads, g_star, hessian_blocks)
		results.append({
			'Method': 'Score',
			'Rate': rate_mag,
			'Temp': np.nan,
			**stats,
		})
		del grads
		clear_gpu_memory()

		# Cleanup after each rate iteration
		del lambda_fixed, log_rate_fixed, g_star, hessian_blocks
		clear_gpu_memory()

	return pd.DataFrame(results)
