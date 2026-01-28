import argparse
import os
import sys

import lightning as L
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(1, "/".join(os.path.abspath(__file__).split("/")[0:-2]))
import models

torch.set_float32_matmul_precision("high")

## arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
parser.add_argument(
    "--mode", type=str, default="both", choices=["train", "evaluate", "both"]
)
args = parser.parse_args()

tag = "linear"


method_list = ["score", "GS", "GS2", "exp-sigmoid", "exp-cubic"]
tau_list = [0.02, 0.05, 0.1, 0.2, 0.5]
seed_list = np.arange(0, 5)

arg_index = np.unravel_index(
    args.idx,
    (
        len(method_list),
        len(tau_list),
        len(seed_list),
    ),
)
method = method_list[arg_index[0]]
tau = tau_list[arg_index[1]]
seed = seed_list[arg_index[2]]
print(f"method: {method}")
print(f"tau: {tau}")
print(f"seed: {seed}")
name = f"{method}_{tau}_{seed}"
results_folder = f"results_{tag}/{name}"

## data
df_data = pd.read_pickle("data/data.pkl")
train_dataloader = DataLoader(
    TensorDataset(df_data.at[0, "x_train"], df_data.at[0, "z_train"]),
    batch_size=100,
    shuffle=True,
    num_workers=1,
)
test_dataloader = DataLoader(
    TensorDataset(df_data.at[0, "x_test"], df_data.at[0, "z_test"]),
    batch_size=100,
    shuffle=False,
    num_workers=1,
)

## model
torch.manual_seed(seed)

## Lightning module
model = models.TwoLayerNetwork(
    dims=(8, 32),
)


if args.mode in ["train", "both"]:
    wandb_logger = WandbLogger(
        name=name,
        project=f"poissongradestim-{__file__.split('/')[-2]}",
        save_dir=results_folder,
        tags=[tag],
        offline=(False if seed < 2 else True),
    )
else:
    wandb_logger = False

checkpoint_callback = ModelCheckpoint(
    save_last=True,
    dirpath=results_folder,
    enable_version_counter=False,
)
timer = Timer()

trainer = L.Trainer(
    logger=wandb_logger,
    max_epochs=100,
    log_every_n_steps=1,
    enable_progress_bar=True,
    devices=1,
    accelerator="cpu",
    callbacks=[
        checkpoint_callback,
        timer,
    ],
)

if args.mode in ["train", "both"]:
    if method == "score":
        lit = models.LitScore(
            model=model,
        )
    elif method == "GS":
        lit = models.LitGS(
            model=model,
            tau=tau,
            upperbound_param=20,
        )
    elif method == "GS2":
        lit = models.LitGS2(
            model=model,
            tau=tau,
            upperbound_param=20,
        )
    elif method == "exp-sigmoid":
        lit = models.LitExpSigmoid(
            model=model,
            tau=tau,
            upperbound_param=20,
        )
    elif method == "exp-cubic":
        lit = models.LitExpCubic(
            model=model,
            tau=tau,
            upperbound_param=20,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    trainer.fit(
        model=lit,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

if args.mode in ["evaluate", "both"]:
    torch.manual_seed(seed)
    lit = models.LitScore.load_from_checkpoint(
        f"{results_folder}/last.ckpt",
        model=model,
    )

    trainer.test(model=lit, dataloaders=test_dataloader)
    df_metric = lit.df_metrics
    df_metric["running_time"] = timer.time_elapsed("train")

    df_metric.to_csv(f"{results_folder}/metrics_last.csv", index=False)
