from . import *


def run_wasserstein_analysis(
        n_samples: int = 50_000,
        n_trials: int = 100,
        rates: list = None,
        temps: list = None,
        normalization: str = 'std',
        device: torch.device = 'cuda',
        dtype=torch.bfloat16,
):
    """
    Compute Wasserstein distances between relaxed Poisson samples and true Poisson.

    Args:
        n_samples: Number of samples per trial
        n_trials: Number of trials for statistics
        rates: List of Poisson rates to test
        temps: List of temperatures to test
        normalization: How to normalize distances. Options:
            - 'none': Raw distance (units of counts)
            - 'mean': Divide by λ (fractional displacement)
            - 'std': Divide by √λ (displacement in std units)
        device: torch device for computation
        dtype: bfloat16 for reduced memory

    Returns:
        DataFrame with W1 and W2 distances for each (rate, temp, method) combination

    Notes:
        For 1D empirical distributions with N samples each:
        - W1 = mean(|x_sorted - y_sorted|)  "average displacement"
        - W2 = sqrt(mean((x_sorted - y_sorted)^2))  "RMS displacement"
    """
    temps = temps or DEFAULT_TEMPERATURES
    rates = rates or DEFAULT_RATES

    assert normalization in ['none', 'mean', 'std'], \
        f"normalization must be 'none', 'mean', or 'std', got {normalization}"

    print(
        f"Running Wasserstein Analysis "
        f"(n_samples={n_samples}, n_trials={n_trials}, norm={normalization})\n\n"
		f"firing rates:\n{rates}\n\n"
		f"temperatures:\n{temps}\n"
    )

    clear_gpu_memory()  # Initial cleanup

    dfs = []

    for rate in tqdm(rates, desc="Rates"):
        # Normalization factor
        if normalization == 'none':
            norm_factor = 1.0
        elif normalization == 'mean':
            norm_factor = rate
        else:  # 'std'
            norm_factor = np.sqrt(rate)

        # GS Upperbound (shared)
        upperbound_safe = int(rate + 4 * (rate ** 0.5) + 5)

        for temp in tqdm(temps, leave=False, desc="Temps"):
            for method in ['sigmoid', 'cubic', 'quintic', 'GS']:

                # Create log_rate_batch fresh each iteration (allows cleanup)
                log_rate_batch = torch.full(
                    size=(n_trials, n_samples),
                    fill_value=np.log(rate),
                    device=device,
                    dtype=dtype,
                )

                # --- Sample from true Poisson ---
                true_samples = torch.poisson(
                    torch.full(
                        size=(n_trials, n_samples),
                        fill_value=rate,
                        device=device,
                        dtype=dtype,
                    )
                )
                true_sorted, _ = torch.sort(true_samples, dim=1)
                del true_samples

                # --- Sample from relaxed distribution ---
                if method == 'GS':
                    dist = GumbelSoftmaxPoisson(
                        log_rate=log_rate_batch,
                        temp=temp,
                        upperbound_method='fixed',
                        upperbound_param=upperbound_safe,
                    )
                    z_soft = dist.rsample()
                    relaxed_samples = dist.aggregate_samples(z_soft)
                    del z_soft, dist
                else:
                    dist = Poisson(
                        log_rate=log_rate_batch,
                        indicator_approx=method,
                        temp=temp,
                        n_exp=N_EXP,
                        n_exp_p=N_EXP_P,
                    )
                    relaxed_samples = dist.rsample()
                    del dist

                del log_rate_batch

                # --- Sort relaxed samples ---
                relaxed_sorted, _ = torch.sort(relaxed_samples, dim=1)
                del relaxed_samples

                # --- Compute Wasserstein distances ---
                diff = relaxed_sorted - true_sorted
                del relaxed_sorted, true_sorted

                # W1 per trial
                w1_per_trial = torch.mean(torch.abs(diff), dim=1) / norm_factor
                w1_np = tonp(w1_per_trial)
                del w1_per_trial

                # W2 per trial
                w2_per_trial = torch.sqrt(torch.mean(diff ** 2, dim=1)) / norm_factor
                w2_np = tonp(w2_per_trial)
                del w2_per_trial, diff

                # --- Bulk DataFrame Creation ---
                batch_df = pd.DataFrame({
                    'Rate': rate,
                    'Temp': temp,
                    'Method': method
                    if method == 'GS'
                    else f"EAT_{method}",
                    'Trial': np.arange(n_trials),
                    'W1': w1_np,
                    'W2': w2_np,
                })
                dfs.append(batch_df)

                # Cleanup after each method
                clear_gpu_memory()

    return pd.concat(dfs, ignore_index=True)


def run_moment_consistency_test(
        n_samples: int = 100,
        n_trials: int = 1_000,
        temps: list = None,
        rates: list = None,
        device: torch.device = 'cuda',
        dtype=torch.bfloat16,  # Changed from float32
):
    """
    Memory usage: (n_trials * n_samples * 2 bytes) for bfloat16.
    For 1k trials * 100k samples, this is ~200MB VRAM.
    """
    temps = temps or DEFAULT_TEMPERATURES
    rates = rates or DEFAULT_RATES

    print(
        f"Running Moment Consistency Test "
        f"(n_samples={n_samples}, n_trials={n_trials})\n\n"
		f"firing rates:\n{rates}\n\n"
		f"temperatures:\n{temps}\n"
    )

    clear_gpu_memory()  # Initial cleanup

    dfs = []

    for r in tqdm(rates, desc="Rates"):
        # GS Upperbound (shared)
        upperbound_safe = int(r + 4 * (r ** 0.5) + 5)

        for tau in tqdm(temps, leave=False, desc="Temps"):
            for method in ['sigmoid', 'cubic', 'quintic', 'GS']:

                # Create log_rate_batch fresh each iteration
                log_rate_batch = torch.full(
                    size=(n_trials, n_samples),
                    fill_value=np.log(r),
                    device=device,
                    dtype=dtype,
                )

                # --- Sampling ---
                if method == 'GS':
                    dist = GumbelSoftmaxPoisson(
                        log_rate=log_rate_batch,
                        temp=tau,
                        upperbound_method='fixed',
                        upperbound_param=upperbound_safe,
                    )
                    z_soft = dist.rsample()
                    z = dist.aggregate_samples(z_soft)
                    del z_soft, dist
                else:
                    dist = Poisson(
                        log_rate=log_rate_batch,
                        indicator_approx=method,
                        temp=tau,
                        n_exp=N_EXP,
                        n_exp_p=N_EXP_P,
                    )
                    z = dist.rsample()
                    del dist

                del log_rate_batch

                # --- Moment Computation ---
                means_np = tonp(z.mean(dim=1))
                vars_np = tonp(z.var(dim=1))
                del z

                # --- Bulk DataFrame Creation ---
                batch_df = pd.DataFrame({
                    'Rate': r,
                    'Temp': tau,
                    'Method': method
                    if method == 'GS'
                    else f"EAT_{method}",
                    'Trial': np.arange(n_trials),
                    'Mean_Ratio': means_np / r,
                    'Mean_Bias': means_np - r,
                    'Var_Ratio': vars_np / r,
                    'Var_Bias': vars_np - r,
                })
                dfs.append(batch_df)

                # Cleanup after each method
                clear_gpu_memory()

    return pd.concat(dfs, ignore_index=True)
