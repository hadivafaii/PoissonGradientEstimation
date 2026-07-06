#!/usr/bin/env python3
"""Minimal estimator usage example."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson_grad_estimators import EATPoisson, GumbelSoftmaxPoisson


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_rate = torch.log(torch.tensor([0.5, 1.0, 3.0], device=device))

    eat = EATPoisson(log_rate, temp=0.1, indicator_approx="cubic")
    eat_sample = eat.rsample()

    gsm = GumbelSoftmaxPoisson(log_rate, temp=0.1, upperbound_method="quantile")
    gsm_soft = gsm.rsample()
    gsm_sample = gsm.aggregate_samples(gsm_soft)

    print(f"device: {device}")
    print(f"rates: {torch.exp(log_rate).detach().cpu().tolist()}")
    print(f"EAT cubic sample: {eat_sample.detach().cpu().tolist()}")
    print(f"GSM aggregated sample: {gsm_sample.detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
