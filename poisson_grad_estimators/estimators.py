"""Self-contained Poisson count relaxations.

This module is intentionally small: users can copy/import it without the VAE
training stack. It provides EAT sigmoid/cubic/quintic relaxations and the
Gumbel-Softmax Poisson relaxation used throughout the paper code.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from scipy import stats as sp_stats


Indicator = Literal["sigmoid", "linear", "cubic", "quintic", "cosine"]


def compute_n_exp(rate: float, p: float = 1e-3) -> int:
    """Upper truncation for arrival-time simulation from Poisson tail mass."""
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    return max(1, int(sp_stats.poisson(rate).ppf(1.0 - p)))


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)


def cubic_smoothstep(x: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
    return 3.0 * u.pow(2) - 2.0 * u.pow(3)


def quintic_smoothstep(x: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
    return 6.0 * u.pow(5) - 15.0 * u.pow(4) + 10.0 * u.pow(3)


def cosine_smoothstep(x: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
    return 0.5 * (1.0 - torch.cos(torch.pi * u))


_INDICATOR_FNS = {
    "sigmoid": torch.sigmoid,
    "linear": hard_sigmoid,
    "cubic": cubic_smoothstep,
    "quintic": quintic_smoothstep,
    "cosine": cosine_smoothstep,
}


class EATPoisson:
    """Exponential Arrival Time relaxation for Poisson random variables."""

    def __init__(
        self,
        log_rate: torch.Tensor,
        temp: float = 0.0,
        indicator_approx: Indicator = "cubic",
        n_exp: int | Literal["infer"] = "infer",
        n_exp_p: float = 1e-3,
    ):
        if temp < 0.0:
            raise ValueError(f"temp must be non-negative, got {temp}")
        if indicator_approx not in _INDICATOR_FNS:
            raise ValueError(f"unknown indicator_approx: {indicator_approx}")

        eps = torch.finfo(log_rate.dtype).eps
        self.log_rate = log_rate
        self.rate = torch.exp(log_rate).clamp_min(eps)
        self.temp = float(temp)
        self.indicator_approx = indicator_approx
        self._exp = torch.distributions.Exponential(self.rate)

        if n_exp == "infer":
            max_rate = float(self.rate.detach().max().cpu())
            n_exp = compute_n_exp(max_rate, n_exp_p)
        self.n_exp = int(n_exp)

    @property
    def mean(self) -> torch.Tensor:
        return self.rate

    @property
    def variance(self) -> torch.Tensor:
        return self.rate

    @torch.no_grad()
    def sample(self, n_samples: int | None = None) -> torch.Tensor:
        if n_samples is None:
            return torch.poisson(self.rate)
        rate = self.rate.unsqueeze(0).expand(n_samples, *self.rate.shape)
        return torch.poisson(rate)

    def rsample(self) -> torch.Tensor:
        if self.temp == 0.0:
            return self.sample()

        interarrival = self._exp.rsample((self.n_exp,))
        arrival = interarrival.cumsum(dim=0)
        logits = (1.0 - arrival) / self.temp
        indicator = _INDICATOR_FNS[self.indicator_approx](logits)
        return indicator.sum(dim=0).to(dtype=self.rate.dtype)

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return (
            -self.rate
            - torch.lgamma(samples + 1.0)
            + samples * torch.log(self.rate)
        )


class GumbelSoftmaxPoisson:
    """Gumbel-Softmax categorical relaxation over Poisson count support."""

    def __init__(
        self,
        log_rate: torch.Tensor,
        temp: float = 1.0,
        upperbound_method: Literal["fixed", "std_ratio", "quantile"] = "quantile",
        upperbound_param: int | float = 1e-3,
    ):
        if temp < 0.0:
            raise ValueError(f"temp must be non-negative, got {temp}")
        self.log_rate = log_rate
        self.temp = float(temp)
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param

        max_rate = float(self.rate.detach().max().cpu())
        if upperbound_method == "fixed":
            self.upperbound = int(upperbound_param)
        elif upperbound_method == "std_ratio":
            self.upperbound = int(max_rate**0.5 * float(upperbound_param))
        elif upperbound_method == "quantile":
            self.upperbound = compute_n_exp(max_rate, float(upperbound_param))
        else:
            raise ValueError(f"unknown upperbound_method: {upperbound_method}")
        self.upperbound = max(1, self.upperbound)

    @property
    def rate(self) -> torch.Tensor:
        return torch.exp(self.log_rate)

    @property
    def logit_pi(self) -> torch.Tensor:
        k = torch.arange(
            self.upperbound,
            device=self.rate.device,
            dtype=self.rate.dtype,
        )
        return k * self.log_rate.unsqueeze(-1) - torch.lgamma(k + 1.0)

    @torch.no_grad()
    def sample(self, n_samples: int | None = None) -> torch.Tensor:
        if n_samples is None:
            return torch.poisson(self.rate)
        rate = self.rate.unsqueeze(0).expand(n_samples, *self.rate.shape)
        return torch.poisson(rate)

    def rsample(self, n_samples: int | None = None) -> torch.Tensor:
        if self.temp == 0.0:
            return self.sample(n_samples=n_samples)
        logits = self.logit_pi
        if n_samples is not None:
            logits = logits.unsqueeze(0).expand(n_samples, *logits.shape)
        return F.gumbel_softmax(logits=logits, tau=self.temp, hard=False)

    def aggregate_samples(self, gumbel_samples: torch.Tensor) -> torch.Tensor:
        k = torch.arange(
            self.upperbound,
            device=self.rate.device,
            dtype=self.rate.dtype,
        )
        return gumbel_samples @ k

    def log_prob(self, samples: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        ln_x = torch.clamp(samples, min=eps).log()
        logit_pi = self.logit_pi
        return (
            math.lgamma(self.upperbound)
            + (self.upperbound - 1) * math.log(max(self.temp, eps))
            - self.upperbound * torch.logsumexp(logit_pi - self.temp * ln_x, dim=-1)
            + (logit_pi - (self.temp + 1) * ln_x).sum(dim=-1)
        )


Poisson = EATPoisson
