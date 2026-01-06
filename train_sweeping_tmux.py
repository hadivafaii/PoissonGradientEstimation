#!/usr/bin/env python3
"""
Training script for sweeping experiment with tmux execution.
This replicates the exact training from the notebook.
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base.config_base import ConfigPoisVAE, ConfigTrainVAE
from base.dataset import get_dataset
from main.vae import PoissonVAE
from main.train_vae import TrainerVAE

def main():
    # Configuration - EXACT SAME as notebook
    MODEL_NAME = "poisson_lin_lin"
    DATA_DIR = "Datasets"
    DATASET = "vH16"
    DEVICE_IDX = 1  # GPU 1 as requested
    DEVICE = f"cuda:{DEVICE_IDX}" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    CHECKPOINT_PATH = "./checkpoints/sweeping_tmux"

    # Model config - linear encoder, linear decoder
    PRIOR_CLAMP = -4
    PRIOR_LOG_DIST = "uniform"
    HARD_FWD = False
    EXC_ONLY = False
    RMAX_Q = 1.0
    N_CH = 32
    N_LATENTS = 512
    ENC_TYPE = "lin"
    DEC_TYPE = "lin"
    ENC_BIAS = False
    DEC_BIAS = False
    ENC_NORM = False
    DEC_NORM = False
    FIT_PRIOR = True
    ACTIVATION_FN = "swish"
    INIT_DIST = "Normal"
    INIT_SCALE = 0.0001
    RES_EPS = 1.0
    USE_BN = False
    USE_SE = True
    SEED = 0

    # Training config
    METHOD = "mc"
    KL_BETA = 1.0
    KL_BETA_MIN = 0.0001
    KL_ANNEAL_CYCLES = 0
    KL_ANNEAL_PORTION = 0.5
    KL_CONST_PORTION = 0.0
    LAMBDA_ANNEAL = False
    LAMBDA_INIT = 0.0
    LAMBDA_NORM = 0.0
    TEMP_ANNEAL_PORTION = 0.5
    TEMP_ANNEAL_TYPE = "lin"
    TEMP_START = 1.0
    TEMP_STOP = 0.05
    LEARNING_RATE = 0.005
    EPOCHS = 1500  # Same as notebook sweeping
    BATCH_SIZE = 1000
    WARM_RESTART = 0
    WARMUP_EPOCHS = 5
    OPTIMIZER = "adamax_fast"
    OPTIMIZER_KWS = {"weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-08}
    SCHEDULER_TYPE = "cosine"
    SCHEDULER_KWS = {"T_max": 1495.0, "eta_min": 1e-05}
    EMA_RATE = None
    GRAD_CLIP = 500
    USE_AMP = False
    CHKPT_FREQ = EPOCHS
    EVAL_FREQ = 20
    LOG_FREQ = 10

    print("="*80)
    print("SWEEPING EXPERIMENT - TMUX TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE} (GPU {DEVICE_IDX})")
    print(f"  Dataset: {DATASET}")
    print(f"  Architecture: {ENC_TYPE}|{DEC_TYPE}")
    print(f"  Latent dims: {N_LATENTS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Set device
    device = torch.device(DEVICE)
    print(f"\n✓ Using device: {device}")

    # Load dataset
    print(f"\nLoading {DATASET} dataset...")
    ds_trn, ds_vld = get_dataset(DATASET, DATA_DIR)
    dl_trn = torch.utils.data.DataLoader(
        ds_trn, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    dl_vld = torch.utils.data.DataLoader(
        ds_vld, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"✓ Dataset loaded!")
    print(f"  Training samples: {len(ds_trn)}")
    print(f"  Validation samples: {len(ds_vld)}")

    # Create model configuration
    model_cfg = ConfigPoisVAE(
        prior_clamp=PRIOR_CLAMP,
        prior_log_dist=PRIOR_LOG_DIST,
        hard_fwd=HARD_FWD,
        exc_only=EXC_ONLY,
        rmax_q=RMAX_Q,
        dataset=DATASET,
        n_ch=N_CH,
        n_latents=N_LATENTS,
        enc_type=ENC_TYPE,
        dec_type=DEC_TYPE,
        enc_bias=ENC_BIAS,
        dec_bias=DEC_BIAS,
        enc_norm=ENC_NORM,
        dec_norm=DEC_NORM,
        fit_prior=FIT_PRIOR,
        activation_fn=ACTIVATION_FN,
        init_dist=INIT_DIST,
        init_scale=INIT_SCALE,
        res_eps=RES_EPS,
        use_bn=USE_BN,
        use_se=USE_SE,
        seed=SEED,
        save=False,
        base_dir=CHECKPOINT_DIR,
        data_dir=DATA_DIR,
    )
    model_cfg.mods_dir = CHECKPOINT_PATH

    # Create training configuration
    train_cfg = ConfigTrainVAE(
        method=METHOD,
        kl_beta=KL_BETA,
        kl_beta_min=KL_BETA_MIN,
        kl_anneal_cycles=KL_ANNEAL_CYCLES,
        kl_anneal_portion=KL_ANNEAL_PORTION,
        kl_const_portion=KL_CONST_PORTION,
        lambda_anneal=LAMBDA_ANNEAL,
        lambda_init=LAMBDA_INIT,
        lambda_norm=LAMBDA_NORM,
        temp_anneal_portion=TEMP_ANNEAL_PORTION,
        temp_anneal_type=TEMP_ANNEAL_TYPE,
        temp_start=TEMP_START,
        temp_stop=TEMP_STOP,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        warm_restart=WARM_RESTART,
        warmup_epochs=WARMUP_EPOCHS,
        optimizer=OPTIMIZER,
        optimizer_kws=OPTIMIZER_KWS,
        scheduler_type=SCHEDULER_TYPE,
        scheduler_kws=SCHEDULER_KWS,
        ema_rate=EMA_RATE,
        grad_clip=GRAD_CLIP,
        use_amp=USE_AMP,
        chkpt_freq=CHKPT_FREQ,
        eval_freq=EVAL_FREQ,
        log_freq=LOG_FREQ,
    )

    print(f"\n✓ Configurations created")

    # Save configurations to checkpoint directory
    config_dict = {
        'model_config': {
            'prior_clamp': PRIOR_CLAMP,
            'prior_log_dist': PRIOR_LOG_DIST,
            'hard_fwd': HARD_FWD,
            'exc_only': EXC_ONLY,
            'rmax_q': RMAX_Q,
            'dataset': DATASET,
            'n_ch': N_CH,
            'n_latents': N_LATENTS,
            'enc_type': ENC_TYPE,
            'dec_type': DEC_TYPE,
            'enc_bias': ENC_BIAS,
            'dec_bias': DEC_BIAS,
            'enc_norm': ENC_NORM,
            'dec_norm': DEC_NORM,
            'fit_prior': FIT_PRIOR,
            'activation_fn': ACTIVATION_FN,
            'init_dist': INIT_DIST,
            'init_scale': INIT_SCALE,
            'res_eps': RES_EPS,
            'use_bn': USE_BN,
            'use_se': USE_SE,
            'seed': SEED,
        },
        'training_config': {
            'method': METHOD,
            'kl_beta': KL_BETA,
            'kl_beta_min': KL_BETA_MIN,
            'kl_anneal_cycles': KL_ANNEAL_CYCLES,
            'kl_anneal_portion': KL_ANNEAL_PORTION,
            'kl_const_portion': KL_CONST_PORTION,
            'lambda_anneal': LAMBDA_ANNEAL,
            'lambda_init': LAMBDA_INIT,
            'lambda_norm': LAMBDA_NORM,
            'temp_anneal_portion': TEMP_ANNEAL_PORTION,
            'temp_anneal_type': TEMP_ANNEAL_TYPE,
            'temp_start': TEMP_START,
            'temp_stop': TEMP_STOP,
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'warm_restart': WARM_RESTART,
            'warmup_epochs': WARMUP_EPOCHS,
            'optimizer': OPTIMIZER,
            'optimizer_kws': OPTIMIZER_KWS,
            'scheduler_type': SCHEDULER_TYPE,
            'scheduler_kws': SCHEDULER_KWS,
            'ema_rate': EMA_RATE,
            'grad_clip': GRAD_CLIP,
            'use_amp': USE_AMP,
            'chkpt_freq': CHKPT_FREQ,
            'eval_freq': EVAL_FREQ,
            'log_freq': LOG_FREQ,
        }
    }

    config_path = os.path.join(CHECKPOINT_PATH, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Saved configuration to {config_path}")

    # Initialize model
    print(f"\nInitializing model...")
    model = PoissonVAE(model_cfg)
    model = model.to(device)

    # Override the create_chkpt_dir method to use simple directory
    def simple_chkpt_dir(self, fit_name=None):
        self.chkpt_dir = CHECKPOINT_PATH
        os.makedirs(self.chkpt_dir, exist_ok=True)

    model.create_chkpt_dir = lambda fit_name=None: simple_chkpt_dir(model, fit_name)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized")
    print(f"  Trainable parameters: {n_params:,}")

    # Initialize trainer
    print(f"\nInitializing trainer...")
    trainer = TrainerVAE(
        model=model,
        device=device,
        train_cfg=train_cfg,
        dl_trn=dl_trn,
        dl_vld=dl_vld,
    )
    print(f"✓ Trainer initialized")

    # Start training
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    trainer.train()

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Model saved to {CHECKPOINT_PATH}")
    
    # Print final metrics
    if hasattr(trainer, 'stats') and len(trainer.stats) > 0:
        stats = trainer.stats
        if 'eval/mse' in stats and isinstance(stats['eval/mse'], dict):
            final_epoch = max([int(k) for k in stats['eval/mse'].keys() if str(k).isdigit()])
            final_mse = stats['eval/mse'][final_epoch]
            final_kl = stats['eval/kl'].get(final_epoch, None)
            
            print(f"\nFinal metrics (epoch {final_epoch}):")
            print(f"  MSE: {final_mse:.4f}")
            if final_kl is not None:
                print(f"  KL: {final_kl:.4f}")
                print(f"  Neg ELBO: {final_mse + final_kl:.4f}")

if __name__ == "__main__":
    main()
