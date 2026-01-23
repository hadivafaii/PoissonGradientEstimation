from utils.plotting import *


def plot_metrics_vs_temp(
		df,
		metrics,
		x="Temp",
		hue="Method",
		marker="o",
		sharey=True,
		order=None,
		ylim=None,
		legend="global", ):
	"""
	Plot multiple metrics (rows) vs temperature for each Rate (columns).

	- nrows = len(metrics)
	- ncols = number of unique Rate values
	- xlabel only on last row
	- ylabel only on first column
	- title only on first row (per column: Rate = ...)
	- ylim can be:
		- None
		- dict: {metric: (ymin, ymax), ...}
		- list/tuple aligned with metrics: [(ymin,ymax), ...]
	- legend can be: "global" | "each" | "none"
	"""

	rates = sorted(df["Rate"].dropna().unique())
	nrows = len(metrics)
	ncols = len(rates)

	# noinspection PyTypeChecker
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(3.1 * ncols, 3.0 * nrows),
		sharey='row' if sharey else 'none',
		sharex='all',
	)

	def _ylim_for_metric(metric_name):
		if ylim is None:
			return None
		if isinstance(ylim, dict):
			return ylim.get(metric_name, None)
		# assume sequence aligned with metrics
		try:
			idx = metrics.index(metric_name)
			return ylim[idx]
		except IndexError:
			return None

	# For a single global legend
	global_handles, global_labels = None, None

	for r, metric in enumerate(metrics):
		y = f"{metric}Mean"
		yerr = f"{metric}Std"
		cur_ylim = _ylim_for_metric(metric)

		for c, lam in enumerate(rates):
			ax = axes[r, c]
			df_selected = df.loc[df["Rate"] == lam]

			g = (
				df_selected.groupby([hue, x], as_index=False)
				.agg({y: "mean", yerr: "mean"})
				.sort_values([hue, x])
			)

			methods = order if order is not None else list(g[hue].dropna().unique())

			for method in methods:
				sub = g[g[hue] == method].sort_values(x)
				if sub.empty:
					continue
				xx = sub[x].to_numpy()
				mu = sub[y].to_numpy()
				sd = sub[yerr].to_numpy()

				ax.plot(xx, mu, marker=marker, linewidth=2, label=str(method))
				ax.fill_between(xx, mu - sd, mu + sd, alpha=0.2)

			ax.invert_xaxis()
			ax.set_ylim(cur_ylim)

			# Titles only on first row
			if r == 0:
				ax.set_title(f"Rate = {lam}", fontsize=17)

			# Y label only on first column
			if c == 0:
				ax.set_ylabel(y, fontsize=17)
			else:
				ax.set_ylabel("")
				ax.tick_params(labelleft=True)  # keep ticks, just remove label text

			# X label only on last row
			if r == nrows - 1:
				ax.set_xlabel(x, fontsize=17)
			else:
				ax.set_xlabel("")
				ax.tick_params(labelbottom=False)

			if legend == "each":
				ax.legend(title=hue)

			if legend == "global" and global_handles is None:
				h, lab = ax.get_legend_handles_labels()
				if len(h) > 0:
					global_handles, global_labels = h, lab

	# One legend for the whole figure (optional)
	if legend == "global" and global_handles is not None:
		fig.legend(
			global_handles,
			global_labels,
			title=hue,
			loc="upper center",
			bbox_to_anchor=(0.5, 1.12),
			ncol=min(len(global_labels), 5),
			frameon=False,
			fontsize=20,
			title_fontsize=20,
		)

	add_grid(axes)
	plt.show()

	return fig


def plot_metric_vs_temp(
		df,
		metric,
		x='Temp',
		hue='Method',
		marker='o',
		sharey=True,
		order=None,
		ylim=None, ):
	"""
	use for a single metric
	"""
	rates = sorted(df['Rate'].unique())

	# noinspection PyTypeChecker
	fig, axes = create_figure(
		nrows=1,
		ncols=len(rates),
		figsize=(3 * len(rates), 3.5),
		sharey='row' if sharey else 'none',
		sharex='all',
	)

	y = f"{metric}Mean"
	yerr = f"{metric}Std"

	for i, lam in enumerate(rates):
		df_selected = df.loc[df['Rate'] == lam]

		g = (
			df_selected.groupby([hue, x], as_index=False)
			.agg({y: "mean", yerr: "mean"})
			.sort_values([hue, x])
		)
		methods = order if order is not None else list(g[hue].dropna().unique())

		ax = axes[i]

		for method in methods:
			sub = g[g[hue] == method].sort_values(x)
			if sub.empty:
				continue
			xx = sub[x].to_numpy()
			mu = sub[y].to_numpy()
			sd = sub[yerr].to_numpy()

			ax.plot(xx, mu, marker=marker, linewidth=2, label=str(method))
			ax.fill_between(xx, mu - sd, mu + sd, alpha=0.2)

		ax.set_xlabel(x)
		if i == 0:
			ax.set_ylabel(y)
		ax.legend(title=hue)
		ax.invert_xaxis()
		ax.set_title(f"Rate = {lam}")
		ax.set_ylim(ylim)

	add_grid(axes)

	plt.show()

	return fig


def plot_mean_results(df, lam: float):
	fig, axes = create_figure(2, 3, figsize=(12, 8), sharex='all')

	# Filter for a specific rate to see Temp trade-off clearly (e.g. Rate=5.0)
	subset = df[df['Rate'] == lam]

	sns.lineplot(data=subset, x='Temp', y='BiasMean', hue='Method', marker='o', ax=axes[0, 0])
	axes[0, 0].set_title(f"Gradient Bias (Rate={lam})")
	axes[0, 0].set_yscale('log')
	axes[0, 0].invert_xaxis()  # High temp (left) -> Low temp (right/discrete)

	sns.lineplot(data=subset, x='Temp', y='VarianceMean', hue='Method', marker='o', ax=axes[0, 1])
	axes[0, 1].set_title(f"Gradient Variance (Rate={lam})")
	axes[0, 1].set_yscale('log')
	axes[0, 1].invert_xaxis()

	sns.lineplot(data=subset, x='Temp', y='SNRMean', hue='Method', marker='o', ax=axes[0, 2])
	axes[0, 2].set_title(f"Signal to Noise Ratio (Rate={lam})")
	axes[0, 2].invert_xaxis()

	sns.lineplot(data=subset, x='Temp', y='BiasEnergyMean', hue='Method', marker='o', ax=axes[1, 0])
	axes[1, 0].set_title(f"Bias Energy (Rate={lam})")
	axes[1, 0].set_yscale('log')
	axes[1, 0].invert_xaxis()

	sns.lineplot(data=subset, x='Temp', y='NoiseEnergyMean', hue='Method', marker='o', ax=axes[1, 1])
	axes[1, 1].set_title(f"Noise Energy (Rate={lam})")
	axes[1, 1].set_yscale('log')
	axes[1, 1].invert_xaxis()

	sns.lineplot(data=subset, x='Temp', y='CosMean', hue='Method', marker='o', ax=axes[1, 2])
	axes[1, 2].set_title(f"CosSim(mean_grad, true_grad) (Rate={lam})")
	axes[1, 2].set_ylim(-0.05, 1.05)
	axes[1, 2].invert_xaxis()

	add_grid(axes)

	plt.show()

	return fig

def plot_moments_bias(df, sharey=False):
	rates = sorted(df['Rate'].unique())

	# noinspection PyTypeChecker
	fig, axes = create_figure(
		nrows=2, ncols=len(rates),
		figsize=(3.2 * len(rates), 6),
		sharey='row' if sharey else 'none',
		sharex='all',
	)

	for i, lam in enumerate(rates):
		df_selected = df.loc[df['Rate'] == lam]

		ax = axes[0, i]
		sns.lineplot(
			data=df_selected,
			x='Temp',
			y='Mean_Bias',
			hue='Method',
			errorbar='sd',
			marker='o',
			ax=ax,
		)
		ax.set_xlabel('')
		ax.set_title(f"Rate = {lam}", fontsize=12)

		ax = axes[1, i]
		sns.lineplot(
			data=df_selected,
			x='Temp',
			y='Var_Bias',
			hue='Method',
			errorbar='sd',
			marker='o',
			ax=ax,
		)

	for ax in axes.flat:
		ax.axhline(0, ls='--', color='k', zorder=0)
		ax.set(ylabel='')
	axes[0, 0].set_ylabel('Mean Bias (Estiamte - True)', fontsize=12)
	axes[1, 0].set_ylabel('Var Bias (Estiamte - True)', fontsize=12)

	add_grid(axes)

	plt.show()

	return fig


def plot_moments_ratio(df, sharey=False):
	rates = sorted(df['Rate'].unique())

	# noinspection PyTypeChecker
	fig, axes = create_figure(
		nrows=2, ncols=len(rates),
		figsize=(3.2 * len(rates), 6),
		sharey='row' if sharey else 'none',
		sharex='all',
	)

	for i, lam in enumerate(rates):
		df_selected = df.loc[df['Rate'] == lam]

		ax = axes[0, i]
		sns.lineplot(
			data=df_selected,
			x='Temp',
			y='Mean_Ratio',
			hue='Method',
			errorbar='sd',
			marker='o',
			ax=ax,
		)
		ax.set_xlabel('')
		ax.set_title(f"Rate = {lam}", fontsize=12)

		ax = axes[1, i]
		sns.lineplot(
			data=df_selected,
			x='Temp',
			y='Var_Ratio',
			hue='Method',
			errorbar='sd',
			marker='o',
			ax=ax,
		)

	for ax in axes.flat:
		ax.axhline(1, ls='--', color='k', zorder=0)
		ax.set(ylabel='')
	axes[0, 0].set_ylabel('Mean Bias (Estiamte / True)', fontsize=12)
	axes[1, 0].set_ylabel('Var Bias (Estiamte / True)', fontsize=12)

	add_grid(axes)

	plt.show()

	return fig
