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
method_list = ["score", "GS"]
seed_list = np.arange(0, 3)

arg_index = np.unravel_index(
    args.idx,
    (
        len(trial_list),
        len(method_list),
        len(seed_list),
    ),
)
trial = trial_list[arg_index[0]]
method = method_list[arg_index[1]]
seed = seed_list[arg_index[2]]
print(f"trial: {trial}")
print(f"method: {method}")
print(f"seed: {seed}")
name = f"{trial}_{method}_{seed}"
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
    )
else:
    wandb_logger = False

checkpoint_callback = ModelCheckpoint(
    save_last=True,
    dirpath=results_folder,
    enable_version_counter=False,
)

trainer = L.Trainer(
    logger=wandb_logger,
    max_epochs=100,
    log_every_n_steps=1,
    enable_progress_bar=True,
    devices=1,
    accelerator="cpu",
    callbacks=[
        checkpoint_callback,
    ],
)

if args.mode in ["train", "both"]:
    if method == "score":
        lit = models.LitPOGLM(
            poglm=poglm,
            n_monte_carlo=5,
            true_weights=df_data.at[trial, "model"]["conv_generative.weight"],
        )
    elif method == "GS":
        lit = models.LitGSPOGLM(
            poglm=poglm,
            n_monte_carlo=5,
            true_weights=df_data.at[trial, "model"]["conv_generative.weight"],
        )
    else:
        raise ValueError(f"unknown method: {method}")
    trainer.fit(
        model=lit,
        train_dataloaders=train_dataloader,
    )

if args.mode in ["evaluate", "both"]:
    lit = models.LitPOGLM.load_from_checkpoint(
        f"{results_folder}/last.ckpt",
        poglm=poglm,
    )

    # trainer.test(
    #     model=lit,
    #     dataloaders=DataLoader(
    #         train_dataset,
    #         batch_size=len(train_dataset),
    #         shuffle=False,
    #         collate_fn=utils.collate_fn,
    #     ),
    # )
    # df_metric_train, result_list_train = lit.df_metric, lit.result_list
    # df_metric_train["split"] = "train"
    # trainer.test(model=lit, dataloaders=val_dataloader)
    # df_metric_test, result_list_test = lit.df_metric, lit.result_list
    # df_metric_test["split"] = "test"
    # df_metric = pd.concat([df_metric_train, df_metric_test])
    # result_list = result_list_train + result_list_test
    # result_list = sorted(result_list, key=lambda x: x["trial"])

    # df_metric.to_csv(f"{results_folder}/metrics_last.csv", index=False)
    # torch.save(result_list, f"{results_folder}/result_last.pt")
