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


trial_list = np.arange(0, 3)
method_list = ["score", "GS", "exp-sigmoid", "exp-cubic"]
tau_list = [0.02, 0.05, 0.1, 0.2, 0.5]
seed_list = np.arange(0, 5)

arg_index = np.unravel_index(
    args.idx,
    (
        len(trial_list),
        len(method_list),
        len(tau_list),
        len(seed_list),
    ),
)

trial = trial_list[arg_index[0]]
method = method_list[arg_index[1]]
tau = tau_list[arg_index[2]]
seed = seed_list[arg_index[3]]
print(f"trial: {trial}")
print(f"method: {method}")
print(f"tau: {tau}")
print(f"seed: {seed}")
name = f"{trial}_{method}_{tau}_{seed}"
results_folder = f"results_{tag}/{name}"

## data
df_data = pd.read_pickle("data/data.pkl")
train_dataloader = DataLoader(
    TensorDataset(df_data.at[trial, "x_train"], df_data.at[trial, "z_train"]),
    batch_size=100,
    shuffle=True,
    num_workers=1,
)
test_dataloader = DataLoader(
    TensorDataset(df_data.at[trial, "x_test"], df_data.at[trial, "z_test"]),
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
    every_n_epochs=20,
    save_top_k=-1,
    filename="{epoch}",
)
timer = Timer()

trainer = L.Trainer(
    logger=wandb_logger,
    max_epochs=300,
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
        )
    elif method == "exp-sigmoid":
        lit = models.LitExpSigmoid(
            model=model,
            tau=tau,
        )
    elif method == "exp-cubic":
        lit = models.LitExpCubic(
            model=model,
            tau=tau,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    trainer.fit(
        model=lit,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

if args.mode in ["evaluate", "both"]:
    for version in [
        f"epoch={epoch}"
        for epoch in range(
            checkpoint_callback.every_n_epochs - 1,
            trainer.max_epochs,
            checkpoint_callback.every_n_epochs,
        )
    ] + ["last"]:
        torch.manual_seed(seed)
        lit = models.LitScore.load_from_checkpoint(
            f"{results_folder}/{version}.ckpt",
            model=model,
        )

        trainer.test(model=lit, dataloaders=test_dataloader)
        df_metric = lit.df_metrics
        df_metric["running_time"] = timer.time_elapsed("train")
        df_metric.to_csv(f"{results_folder}/metrics_{version}.csv", index=False)
        torch.save(lit.z_samples, f"{results_folder}/z_samples_{version}.pt")
