# A hitchhiker's guide to Poisson gradient estimation

Official code for the ICML 2026 paper:

> Michael Ibrahim*, Hanqi Zhao*, Eli Sennesh, Zhi Li, Anqi Wu, Jacob L. Yates, Chengrui Li (co-senior), Hadi Vafaii (co-senior).
> **A hitchhiker's guide to Poisson gradient estimation**. Proceedings of the 43rd International Conference on Machine Learning, 2026.

This repository contains PyTorch code for experiments comparing differentiable estimators for Poisson-distributed latent variables, including Exponential Arrival Time (EAT), the cubic EAT variant introduced in the paper, and Gumbel-Softmax Poisson relaxation. It includes both the current VAE/sweeping code and the earlier POGLM, COM-Poisson, negative-binomial, and analysis code from `_PoissonGradEstim`.

## Main Components

- `poisson_grad_estimators/`: lightweight public API for directly importing EAT and Gumbel-Softmax Poisson estimators.
- `base/distributions.py`: VAE-facing Poisson, Gumbel-Softmax Poisson, and differentiable count relaxations. The Poisson relaxation supports `indicator_approx="sigmoid"`, `"linear"`, `"cubic"`, and `"cosine"`.
- `main/`: VAE models, configs, training loop, checkpointing, and resume logic.
- `main/com_poisson/`, `main/negative_binomial/`, `main/poglm/`: legacy experiment code for the paper's COM-Poisson, negative-binomial, and POGLM experiments.
- `analysis/`: analysis utilities for model evaluation and result tables.
- `figures/recreate_eat_cubic_tau_plot.py`: standalone script for recreating the EAT/GSM temperature relaxation comparison.
- `scripts/run_experiment_grid.py`: launches the legacy paper experiment grids from one command.
- `scripts/reproduce_core.sh`: runs estimator smoke test and regenerates the EAT/GSM relaxation figure.
- `examples/use_estimators.py`: minimal example showing direct estimator use.
- `train_sweeping.py`, `train_sweeping_tmux.py`, `run_sweeping.sh`: VAE sweeping experiment helpers.
- `results.ipynb`, `p-vae.ipynb`, `load_models.ipynb`: notebooks for experiments, results, and checkpoint inspection.
- `notebooks/`: legacy relaxed-Poisson notebooks from the earlier codebase.
- `results/`: cached result data frames used by notebooks and plotting code.

## Setup

Python version used for this repository:

```bash
python --version
# Python 3.12.3
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Use Estimators Directly

Minimal direct use:

```python
import torch
from poisson_grad_estimators import EATPoisson, GumbelSoftmaxPoisson

log_rate = torch.log(torch.tensor([0.5, 1.0, 3.0]))

eat = EATPoisson(log_rate, temp=0.1, indicator_approx="cubic")
z_eat = eat.rsample()

gsm = GumbelSoftmaxPoisson(log_rate, temp=0.1, upperbound_method="quantile")
z_gsm = gsm.aggregate_samples(gsm.rsample())
```

Runnable smoke test:

```bash
python3 examples/use_estimators.py
```

## Data

The VAE experiments expect datasets under `~/Datasets/`:

1. `~/Datasets/DOVES/vH16`
2. `~/Datasets/CIFAR16/xtract16`
3. `~/Datasets/MNIST/processed`

The dataset loader is implemented in `base/dataset.py`. Existing processed dataset links from the P-VAE codebase remain compatible:

- Complete folder: <https://drive.google.com/drive/folders/1mCrsYtxcbNODcCTCLdaTi5v8yN_n5AMA?usp=sharing>
- van Hateren: <https://drive.google.com/drive/folders/1zaQPZm-8LhRXA24wMj4JeJf3s7Z0iIkM?usp=sharing>
- CIFAR_16x16: <https://drive.google.com/drive/folders/1q0TAKHxaEfRfU0YwgykD8TTiYhCpZ400?usp=sharing>
- MNIST: <https://drive.google.com/drive/folders/1WQVqoUU1vbNTs4fd5jgA3zZR1j_XN3cC?usp=sharing>

## Reproduce Figures And Results

The figure recreation script uses available CUDA devices when CUDA is available:

```bash
python figures/recreate_eat_cubic_tau_plot.py \
  --samples 6000000 \
  --pdf figures/eat_cubic_tau_recreation.pdf \
  --png figures/eat_cubic_tau_recreation.png
```

One-command core reproduction:

```bash
./scripts/reproduce_core.sh
```

Quick version for smoke testing:

```bash
SAMPLES=200000 ./scripts/reproduce_core.sh
```

Legacy paper experiment grids:

```bash
python scripts/run_experiment_grid.py --experiment negative_binomial --all
python scripts/run_experiment_grid.py --experiment com_poisson --all
python scripts/run_experiment_grid.py --experiment poglm --all
```

Single-job smoke test:

```bash
python scripts/run_experiment_grid.py --experiment negative_binomial --max-jobs 1
```

The experiment runner sets `WANDB_MODE=offline` by default. Outputs are written inside each experiment folder under `results_*`.

## Train VAE Experiments

Generic VAE training entrypoint:

```bash
python -m main.train_vae <device> <dataset> <model> <archi>
```

Example with van Hateren patches, Poisson latents, and linear encoder/decoder:

```bash
python -m main.train_vae <device> vH16 poisson 'lin|lin'
```

Shell wrapper:

```bash
./scripts/fit_vae.sh <device> <dataset> <model> <archi>
```

Arguments:

- `<device>`: CUDA device argument passed to the trainer.
- `<dataset>`: one of `vH16`, `CIFAR16`, `MNIST`.
- `<model>`: one of `poisson`, `categorical`, `gaussian`, `laplace`.
- `<archi>`: architecture string such as `lin|lin`, `conv+b|lin`, or `conv+b|conv+b`.

Additional hyperparameters can be passed through to `main.train_vae`:

```bash
./scripts/fit_vae.sh <device> vH16 poisson 'lin|lin' --n_latents 1024 --kl_beta 2.5
```

## VAE-Facing Estimators

EAT with cubic compact-support smoothstep:

```python
from base.distributions import Poisson

dist = Poisson(log_rate, temp=0.1, indicator_approx="cubic")
z = dist.rsample()
```

Original sigmoid EAT:

```python
dist = Poisson(log_rate, temp=0.1, indicator_approx="sigmoid")
z = dist.rsample()
```

Gumbel-Softmax Poisson:

```python
from base.distributions import GumbelSoftmaxPoisson

dist = GumbelSoftmaxPoisson(log_rate, temp=0.1, upperbound_method="quantile")
soft_sample = dist.rsample()
z = dist.aggregate_samples(soft_sample)
```

## Citation

```bibtex
@inproceedings{ibrahim2026hitchhikers,
  title={A hitchhiker's guide to Poisson gradient estimation},
  author={Ibrahim, Michael and Zhao, Hanqi and Sennesh, Eli and Li, Zhi and Wu, Anqi and Yates, Jacob L. and Li, Chengrui and Vafaii, Hadi},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## Contact

For code issues, open a GitHub issue. For paper questions, contact Chengrui Li `<cnlichengrui@meta.com>` or Hadi Vafaii `<vafaii@berkeley.edu>`.
