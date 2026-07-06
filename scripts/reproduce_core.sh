#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="${PYTHON:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

SAMPLES="${SAMPLES:-6000000}"
PDF="${PDF:-figures/eat_cubic_tau_recreation.pdf}"
PNG="${PNG:-figures/eat_cubic_tau_recreation.png}"

"${PYTHON_BIN}" examples/use_estimators.py
"${PYTHON_BIN}" figures/recreate_eat_cubic_tau_plot.py \
  --samples "${SAMPLES}" \
  --pdf "${PDF}" \
  --png "${PNG}"
