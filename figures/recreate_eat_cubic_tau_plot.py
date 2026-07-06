#!/usr/bin/env python3
"""Recreate Poisson vs temperature-relaxed count estimators."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D


RATE = 1.0
TAUS = (0.1,)
METHODS = (
    ("EAT_cubic", "#cf5d5d"),
    ("GSM", "#55a868"),
    ("EAT_sigmoid", "#4c72b0"),
)
N_EXP = 12
TOTAL_SAMPLES = 6_000_000
# Choose edges so integer counts lie at bin centers, not bin edges.
X_MIN = -0.25
X_MAX = 5.35
POISSON_BIN_WIDTH = 0.10
RELAXED_BIN_WIDTH = POISSON_BIN_WIDTH
LINE_WIDTH = 2.4


def cubic_smoothstep(x: torch.Tensor) -> torch.Tensor:
    w = (0.5 * x + 0.5).clamp(0.0, 1.0)
    return 3.0 * w.square() - 2.0 * w.square() * w


def sample_eat_histogram(
    total_samples: int,
    rate: float,
    tau: float,
    n_exp: int,
    bins: int,
    xmin: float,
    xmax: float,
    method: str,
) -> torch.Tensor:
    counts = torch.zeros(bins, dtype=torch.float64)
    seed_offsets = {
        "EAT_sigmoid": 20260626,
        "EAT_cubic": 20260627,
    }

    if torch.cuda.is_available():
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]

    base = total_samples // len(devices)
    extras = total_samples % len(devices)

    for idx, device in enumerate(devices):
        n = base + (1 if idx < extras else 0)
        generator = torch.Generator(device=device).manual_seed(seed_offsets[method] + idx)
        interarrival = torch.empty((n_exp, n), device=device).exponential_(
            rate, generator=generator
        )
        arrival = interarrival.cumsum(dim=0)
        logits = (1.0 - arrival) / tau
        if method == "EAT_sigmoid":
            z = torch.sigmoid(logits).sum(dim=0)
        elif method == "EAT_cubic":
            z = cubic_smoothstep(logits).sum(dim=0)
        else:
            raise ValueError(f"Unknown EAT method: {method}")
        counts += torch.histc(z, bins=bins, min=xmin, max=xmax).cpu().double()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return counts


def sample_gumbel_softmax_histogram(
    total_samples: int,
    rate: float,
    tau: float,
    n_exp: int,
    bins: int,
    xmin: float,
    xmax: float,
) -> torch.Tensor:
    counts = torch.zeros(bins, dtype=torch.float64)

    if torch.cuda.is_available():
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]

    base = total_samples // len(devices)
    extras = total_samples % len(devices)

    for idx, device in enumerate(devices):
        n = base + (1 if idx < extras else 0)
        generator = torch.Generator(device=device).manual_seed(20260628 + idx)
        k = torch.arange(n_exp, device=device, dtype=torch.float32)
        logits = k * math.log(rate) - torch.lgamma(k + 1.0)
        uniform = torch.rand((n, n_exp), device=device, generator=generator).clamp_(
            1e-8, 1.0 - 1e-8
        )
        gumbel = -torch.log(-torch.log(uniform))
        probs = torch.softmax((logits.unsqueeze(0) + gumbel) / tau, dim=1)
        z = probs @ k
        counts += torch.histc(z, bins=bins, min=xmin, max=xmax).cpu().double()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return counts


def poisson_pmf(rate: float, k_max: int) -> np.ndarray:
    return np.array(
        [math.exp(-rate) * rate**k / math.factorial(k) for k in range(k_max + 1)]
    )


def smooth_density(
    counts: torch.Tensor,
    bin_width: float,
    smooth_sigma_x: float,
) -> np.ndarray:
    density = counts / (counts.sum() * bin_width)
    sigma_bins = smooth_sigma_x / bin_width
    radius = int(math.ceil(4.0 * sigma_bins))
    kernel_x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (kernel_x / sigma_bins).square())
    kernel /= kernel.sum()
    return torch.nn.functional.conv1d(
        density.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=radius,
    ).view(-1).numpy()


def rasterize_pdf(pdf_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "220",
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        check=True,
    )


def make_plot(pdf_path: Path, png_path: Path | None, total_samples: int) -> None:
    bins = int(round((X_MAX - X_MIN) / RELAXED_BIN_WIDTH))
    edges = np.linspace(X_MIN, X_MAX, bins + 1)

    k = np.arange(0, 6)
    pmf_density = poisson_pmf(RATE, int(k.max())) / POISSON_BIN_WIDTH

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 20,
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.bar(
        k,
        pmf_density,
        width=POISSON_BIN_WIDTH,
        align="center",
        facecolor="none",
        edgecolor="#222222",
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    for tau in TAUS:
        for method_idx, (method, color) in enumerate(METHODS):
            if method == "GSM":
                counts = sample_gumbel_softmax_histogram(
                    total_samples=total_samples,
                    rate=RATE,
                    tau=tau,
                    n_exp=N_EXP,
                    bins=bins,
                    xmin=X_MIN,
                    xmax=X_MAX,
                )
            else:
                counts = sample_eat_histogram(
                    total_samples=total_samples,
                    rate=RATE,
                    tau=tau,
                    n_exp=N_EXP,
                    bins=bins,
                    xmin=X_MIN,
                    xmax=X_MAX,
                    method=method,
                )
            density = (counts / (counts.sum() * RELAXED_BIN_WIDTH)).numpy()
            ax.stairs(
                density,
                edges,
                color=color,
                lw=LINE_WIDTH,
                baseline=None,
                zorder=3 + method_idx,
            )

    ax.set_xlim(-0.4, 5.3)
    ax.set_ylim(0.0, 4.2)
    ax.set_ylabel("Percent", fontsize=30)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 4.5, 1))
    ax.tick_params(axis="both", labelsize=22, pad=6)
    sns.despine(ax=ax, top=True, right=True)

    handles = [
        Line2D([0], [0], color="#222222", lw=LINE_WIDTH),
        *(Line2D([0], [0], color=color, lw=LINE_WIDTH) for _, color in METHODS),
    ]
    labels = [
        r"$\tau = 0$ (Poisson)",
        *(rf"$\tau = {TAUS[0]}$ ({method})" for method, _ in METHODS),
    ]
    ax.legend(
        handles,
        labels,
        title="Temperature",
        loc="upper right",
        frameon=True,
        fontsize=18,
        title_fontsize=22,
        borderpad=0.55,
        handlelength=2.6,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    plt.close(fig)
    if png_path is not None:
        rasterize_pdf(pdf_path, png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("figures/eat_cubic_tau_recreation.pdf"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("figures/eat_cubic_tau_recreation.png"),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=TOTAL_SAMPLES,
        help="Total Monte Carlo samples split across available devices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plot(args.pdf, args.png, args.samples)


if __name__ == "__main__":
    main()
