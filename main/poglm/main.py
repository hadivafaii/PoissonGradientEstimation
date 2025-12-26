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

tag = "baseline"


trial_list = np.arange(10)[:1]
method_list = ["score", "GS", "exp"]
upperbound_list = [5, 8, 10, 15]
temp_list = [0.1, 0.2, 0.5, 1.0]
n_monte_carlo_list = [1, 2, 5, 10]
seed_list = np.arange(0, 3)

arg_index = np.unravel_index(
    args.idx,
    (
        len(trial_list),
        len(method_list),
        len(upperbound_list),
        len(temp_list),
        len(n_monte_carlo_list),
        len(seed_list),
    ),
)
trial = trial_list[arg_index[0]]
method = method_list[arg_index[1]]
upperbound = upperbound_list[arg_index[2]]
temp = temp_list[arg_index[3]]
n_monte_carlo = n_monte_carlo_list[arg_index[4]]
seed = seed_list[arg_index[5]]
print(f"trial: {trial}")
print(f"method: {method}")
print(f"upperbound: {upperbound}")
print(f"temp: {temp}")
print(f"n_monte_carlo: {n_monte_carlo}")
print(f"seed: {seed}")
name = f"{trial}_{method}_{upperbound}_{temp}_{n_monte_carlo}_{seed}"
results_folder = f"results_{tag}/{name}"

## data
df_data = pd.read_pickle("data/data.pkl")
train_dataloader = DataLoader(
    TensorDataset(df_data.at[trial, "y_train"]),
    batch_size=8,
    shuffle=True,
    num_workers=1,
)
test_dataloader = DataLoader(
    TensorDataset(df_data.at[trial, "y_test"]),
    batch_size=20,
    shuffle=False,
    num_workers=1,
)

## model
torch.manual_seed(seed)

## Lightning module
poglm = models.POGLM(
    n_vis_neurons=3,
    n_hid_neurons=2,
    kernel_size=3,
    max_rate=10,
)


if args.mode in ["train", "both"]:
    wandb_logger = WandbLogger(
        name=name,
        project=f"poissongradestim-{__file__.split('/')[-2]}",
        save_dir=results_folder,
        tags=[tag],
        offline=(
            False
            if (trial == 0 and upperbound == 8 and temp == 0.2 and n_monte_carlo == 5)
            else True
        ),
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
        lit = models.LitPOGLM(
            poglm=poglm,
            n_monte_carlo=n_monte_carlo,
            true_model_state_dict=df_data.at[trial, "model"],
        )
    elif method == "GS":
        lit = models.LitGSPOGLM(
            poglm=poglm,
            n_monte_carlo=n_monte_carlo,
            true_model_state_dict=df_data.at[trial, "model"],
            temp=temp,
            upperbound_param=upperbound,
        )
    elif method == "exp":
        lit = models.LitExpPOGLM(
            poglm=poglm,
            n_monte_carlo=n_monte_carlo,
            true_model_state_dict=df_data.at[trial, "model"],
            temp=temp,
            upperbound_param=upperbound,
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
    lit = models.LitPOGLM.load_from_checkpoint(
        f"{results_folder}/last.ckpt",
        poglm=poglm,
    )

    trainer.test(model=lit, dataloaders=test_dataloader)
    df_metric = lit.df_metrics
    df_metric["running_time"] = timer.time_elapsed("train")

    df_metric.to_csv(f"{results_folder}/metrics_last.csv", index=False)
