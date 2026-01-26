from utils.plotting import *
from . import COLORMAPS


def plot_metrics_vs_temp(
		df,
		metrics,
		x='Temp',
		hue='Method',
		marker='o',
		sharey=True,
		order=None,
		ylim=None,
		yscale='linear',
		legend='global',
		add_labels=False,
		metrics_as_rows=True, ):
	"""
	Plot multiple metrics vs temperature for each Rate.

	- metrics_as_rows
	   - True: metrics as rows, rates as columns
	   - False: metrics as columns, rates as rows
	- add_labels: if True, add xlabel, ylabel, and titles
	- xlabel only on last row
	- ylabel only on first column
	- title only on first row
	- ylim can be:
	   - None
	   - dict: {metric: (ymin, ymax), ...}
	   - list/tuple aligned with metrics: [(ymin,ymax), ...]
	- yscale options: {'linear', 'log'}
	   - None
	   - dict: {metric: 'log', ...}
	   - list/tuple aligned with metrics: ['log', ...]
	- legend can be:
	   - "global": single legend for entire figure
	   - "each": legend on every subplot
	   - "none": no legends
	   - (i, j): legend on subplot at row i, column j
	   - [(i1, j1), (i2, j2), ...]: legends on specified subplots
	"""

	rates = sorted(df["Rate"].dropna().unique())

	if metrics_as_rows:
		nrows = len(metrics)
		ncols = len(rates)
		sharey_mode = 'row' if sharey else 'none'
	else:
		nrows = len(rates)
		ncols = len(metrics)
		sharey_mode = 'col' if sharey else 'none'

	# Normalize legend positions to a set of (i, j) tuples
	if isinstance(legend, tuple) and len(legend) == 2 and isinstance(legend[0], int):
		legend_positions = {legend}
	elif isinstance(legend, list) and all(isinstance(pos, tuple) for pos in legend):
		legend_positions = set(legend)
	else:
		legend_positions = None  # handled by string options

	# noinspection PyTypeChecker
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(3.1 * ncols, 3.0 * nrows),
		sharey=sharey_mode,
		sharex='all',
		reshape=True,
	)

	def _value_for_metric(spec, _metric):
		if spec is None:
			return None
		if isinstance(spec, dict):
			return spec.get(_metric, None)
		if isinstance(spec, (list, tuple)):
			return spec[metrics.index(metric)]
		return spec

	# For a single global legend
	global_handles, global_labels = None, None

	for r in range(nrows):
		for c in range(ncols):
			if metrics_as_rows:
				metric = metrics[r]
				lam = rates[c]
			else:  # metrics_cols
				lam = rates[r]
				metric = metrics[c]

			y = f"{metric}Mean"
			yerr = f"{metric}Std"

			ax = axes[r, c]
			df_selected = df.loc[df["Rate"] == lam]

			g = (
				df_selected.groupby([hue, x], as_index=False)
				.agg({y: "mean", yerr: "mean"})
				.sort_values([hue, x])
			)

			methods = order if order is not None \
				else list(g[hue].dropna().unique())

			for method in methods:
				sub = g[g[hue] == method].sort_values(x)
				if sub.empty:
					continue
				xx = sub[x].to_numpy()
				mu = sub[y].to_numpy()
				sd = sub[yerr].to_numpy()

				ax.plot(
					xx, mu,
					marker=marker,
					linewidth=2,
					label=str(method),
					color=COLORMAPS[method],
				)
				ax.fill_between(
					xx, mu - sd, mu + sd,
					color=COLORMAPS[method],
					alpha=0.2,
				)

			ax.invert_xaxis()
			ax.set(
				ylim=_value_for_metric(ylim, metric),
				yscale=_value_for_metric(yscale, metric),
			)

			if add_labels:
				# Titles only on first row
				if r == 0:
					if metrics_as_rows:
						ax.set_title(f"Rate = {lam}", fontsize=17)
					else:
						ax.set_title(f"{y}", fontsize=17)

				# Y label only on first column
				if c == 0:
					if metrics_as_rows:
						ax.set_ylabel(y, fontsize=17)
					else:
						ax.set_ylabel(f"Rate = {lam}", fontsize=17)
				else:
					ax.set_ylabel("")
					ax.tick_params(labelleft=True)

				# X label only on last row
				if r == nrows - 1:
					ax.set_xlabel(x, fontsize=17)
				else:
					ax.set_xlabel("")
					ax.tick_params(labelbottom=False)

			# Handle legend placement
			if legend == "each":
				ax.legend(title=hue, fontsize=13)
			elif legend_positions is not None and (r, c) in legend_positions:
				ax.legend(title=hue, fontsize=13)

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

	return fig, axes


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
		methods = order if order is not None \
			else list(g[hue].dropna().unique())

		ax = axes[i]

		for method in methods:
			sub = g[g[hue] == method].sort_values(x)
			if sub.empty:
				continue
			xx = sub[x].to_numpy()
			mu = sub[y].to_numpy()
			sd = sub[yerr].to_numpy()

			ax.plot(
				xx, mu,
				marker=marker,
				linewidth=2,
				label=str(method),
				color=COLORMAPS[method],
			)
			ax.fill_between(
				xx, mu - sd, mu + sd,
				color=COLORMAPS[method],
				alpha=0.2,
			)

		ax.set_xlabel(x)
		if i == 0:
			ax.set_ylabel(y)
		ax.legend(title=hue)
		ax.invert_xaxis()
		ax.set_title(f"Rate = {lam}")
		ax.set_ylim(ylim)

	add_grid(axes)

	plt.show()

	return fig, axes


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

	return fig, axes


def plot_dist_consistency(
		df: pd.DataFrame,
		kind: str = 'wasser',  # 'wasser' | 'bias' | 'ratio'
		add_labels: bool = True,
		sharey: bool = False,
		x: str = 'Temp',
		hue: str = 'Method',
		errorbar: str = 'sd',
		marker: str = 'o',
		legend=True, ):
	"""
	kind:
	- "wasser": plots (W1, W2), horizontal reference line at 0
	- "bias":   plots (Mean_Bias, Var_Bias), horizontal reference line at 0
	- "ratio":  plots (Mean_Ratio, Var_Ratio), horizontal reference line at 1

	add_labels: if True, add xlabel, ylabel, and titles

	legend can be:
	   - True: legend on every subplot
	   - None or False: no legends
	   - (i, j): legend on subplot at row i, column j
	   - [(i1, j1), (i2, j2), ...]: legends on specified subplots
	"""
	kind = str(kind).lower()
	if kind not in {"wasser", "bias", "ratio"}:
		raise ValueError(
			f"kind must be one of "
			f"{{'wasser','bias','ratio'}}, got {kind!r}"
		)

	rates = sorted(df['Rate'].unique())
	nrows, ncols = 2, len(rates)

	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(3.2 * ncols, 6),
		sharey='row' if sharey else 'none',
		sharex='all',
	)

	# Normalize legend positions
	if legend in [True, 'true']:
		legend_all = True
		legend_positions = set()
	elif isinstance(legend, tuple) and len(legend) == 2 and isinstance(legend[0], int):
		legend_all = False
		legend_positions = {legend}
	elif isinstance(legend, list) and all(isinstance(pos, tuple) for pos in legend):
		legend_all = False
		legend_positions = set(legend)
	else:
		legend_all = False
		legend_positions = set()  # no legends

	# Configure plotting targets
	if kind == "wasser":
		y_top, y_bot = "W1", "W2"
		hline_y = 0.0
		ylabel_top, ylabel_bot = "Wasserstein 1", "Wasserstein 2"
	elif kind == "bias":
		y_top, y_bot = "Mean_Bias", "Var_Bias"
		hline_y = 0.0
		ylabel_top = "Mean Bias (Estimate - True)"
		ylabel_bot = "Var Bias (Estimate - True)"
	elif kind == "ratio":
		y_top, y_bot = "Mean_Ratio", "Var_Ratio"
		hline_y = 1.0
		ylabel_top = "Mean Ratio (Estimate / True)"
		ylabel_bot = "Var Ratio (Estimate / True)"
	else:
		raise ValueError(kind)

	for i, lam in enumerate(rates):
		df_selected = df.loc[df["Rate"] == lam]

		ax = axes[0, i]
		sns.lineplot(
			data=df_selected,
			x=x,
			y=y_top,
			hue=hue,
			palette=COLORMAPS,
			errorbar=errorbar,
			marker=marker,
			ax=ax,
			legend=legend_all or (0, i) in legend_positions,
		)
		ax.set_xlabel('')
		if add_labels:
			ax.set_title(f'Rate = {lam}', fontsize=17)
		else:
			ax.set_title('')

		ax = axes[1, i]
		sns.lineplot(
			data=df_selected,
			x=x,
			y=y_bot,
			hue=hue,
			palette=COLORMAPS,
			errorbar=errorbar,
			marker=marker,
			ax=ax,
			legend=legend_all or (1, i) in legend_positions,
		)
		if add_labels and i == nrows - 1:
			ax.set_xlabel(x, fontsize=12)

	for ax in axes.flat:
		ax.axhline(hline_y, ls='--', color='k', zorder=1.5)
		ax.set(ylabel='')

	if add_labels:
		axes[0, 0].set_ylabel(ylabel_top, fontsize=12)
		axes[1, 0].set_ylabel(ylabel_bot, fontsize=12)
	else:
		for ax in axes.flat:
			ax.set(xlabel='', ylabel='', title='')

	add_grid(axes)
	plt.show()
	return fig, axes
