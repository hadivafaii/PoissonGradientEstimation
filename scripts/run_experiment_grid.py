#!/usr/bin/env python3
"""Run legacy paper experiment job grids from one entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "com_poisson": {
        "workdir": ROOT / "main" / "com_poisson",
        "num_jobs": 3 * 4 * 5 * 5,
    },
    "negative_binomial": {
        "workdir": ROOT / "main" / "negative_binomial",
        "num_jobs": 5 * 5 * 5,
    },
    "poglm": {
        "workdir": ROOT / "main" / "poglm",
        "num_jobs": 1 * 3 * 4 * 4 * 4 * 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = EXPERIMENTS[args.experiment]
    total = spec["num_jobs"]
    stop = total if args.all else min(total, args.start + args.max_jobs)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")
    env.setdefault("WANDB_MODE", "offline")

    for idx in range(args.start, stop):
        cmd = [
            sys.executable,
            "main.py",
            "--idx",
            str(idx),
            "--mode",
            args.mode,
        ]
        print(f"[{args.experiment}] job {idx + 1}/{total}: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=spec["workdir"], env=env, check=True)


if __name__ == "__main__":
    main()
