# A hitchhiker's guide to Poisson gradient estimation

Official code for the ICML 2026 paper:

> Michael Ibrahim*, Hanqi Zhao*, Eli Sennesh, Zhi Li, Anqi Wu, Jacob L. Yates, Chengrui Li (co-senior), Hadi Vafaii (co-senior).
> **A hitchhiker's guide to Poisson gradient estimation**. Proceedings of the 43rd International Conference on Machine Learning, 2026.

This repository contains PyTorch code for experiments comparing differentiable estimators for Poisson-distributed latent variables, including Exponential Arrival Time (EAT), the cubic EAT variant introduced in the paper, and Gumbel-Softmax Poisson relaxation.

## Main Components

- `base/distributions.py`: Poisson, Gumbel-Softmax Poisson, and differentiable count relaxations. The Poisson relaxation supports `indicator_approx="sigmoid"`, `"linear"`, `"cubic"`, and `"cosine"`.
- `main/`: VAE models, configs, training loop, checkpointing, and resume logic.
- `analysis/`: analysis utilities for model evaluation and result tables.
- `figures/recreate_eat_cubic_tau_plot.py`: standalone script for recreating the EAT/GSM temperature relaxation comparison.
- `train_sweeping.py`, `train_sweeping_tmux.py`, `run_sweeping.sh`: sweeping experiment helpers.
- `results.ipynb`, `p-vae.ipynb`, `load_models.ipynb`: notebooks for experiments, results, and checkpoint inspection.
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

For GPU runs on shared machines, GPU 0 can be hidden before launching Python:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3
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

## Recreate Relaxation Plot

The figure recreation script defaults to GPUs 1, 2, and 3 when CUDA is available:

```bash
python figures/recreate_eat_cubic_tau_plot.py \
  --pdf figures/eat_cubic_tau_recreation.pdf \
  --png figures/eat_cubic_tau_recreation.png
```

## Train VAE Experiments

Generic VAE training entrypoint:

```bash
python -m main.train_vae <device> <dataset> <model> <archi>
```

Example with physical GPUs 1-3 visible, visible device 0 selected, van Hateren patches, Poisson latents, and linear encoder/decoder:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m main.train_vae 0 vH16 poisson 'lin|lin'
```

Shell wrapper:

```bash
./scripts/fit_vae.sh <device> <dataset> <model> <archi>
```

Arguments:

- `<device>`: CUDA device index visible to PyTorch.
- `<dataset>`: one of `vH16`, `CIFAR16`, `MNIST`.
- `<model>`: one of `poisson`, `categorical`, `gaussian`, `laplace`.
- `<archi>`: architecture string such as `lin|lin`, `conv+b|lin`, or `conv+b|conv+b`.

Additional hyperparameters can be passed through to `main.train_vae`:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 ./scripts/fit_vae.sh 0 vH16 poisson 'lin|lin' --n_latents 1024 --kl_beta 2.5
```

## Key Estimators

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
