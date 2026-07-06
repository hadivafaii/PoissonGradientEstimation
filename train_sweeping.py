#!/usr/bin/env python3
"""
Training script for poisson_lin_lin model - sweeping experiment
Replicates the exact configuration from p-vae.ipynb
"""

import os
import sys
import json
from pathlib import Path

# Hide physical GPU 0 before Torch sees CUDA devices.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3")

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base.dataset import get_dataset
from base.config_base import ConfigPoisVAE, ConfigTrainVAE
from main.vae import PoissonVAE
from main.train_vae import TrainerVAE

def main():
    # Configuration - EXACT same as notebook
    MODEL_NAME = "poisson_lin_lin"
    DATA_DIR = "Datasets"
    DATASET = "vH16"
    DEVICE_IDX = int(os.environ.get("POISSON_DEVICE_IDX", "0"))
    DEVICE = f"cuda:{DEVICE_IDX}" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints/sweeping/poisson_lin_lin_tmux"
    
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
    EPOCHS = 1500
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
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("="*80)
    print("POISSON VAE TRAINING - SWEEPING EXPERIMENT (TMUX)")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}")
    print(f"Architecture: {ENC_TYPE}|{DEC_TYPE}")
    print(f"Latent dims: {N_LATENTS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print("="*80 + "\n")
    
    # Set device
    device = torch.device(DEVICE)
    
    # Load dataset
    print("Loading dataset...")
    ds_trn, ds_vld = get_dataset(DATASET, DATA_DIR)
    dl_trn = torch.utils.data.DataLoader(
        ds_trn, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    dl_vld = torch.utils.data.DataLoader(
        ds_vld, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"✓ Dataset loaded: {len(ds_trn)} train, {len(ds_vld)} validation samples\n")
    
    # Create model config
    model_cfg = ConfigPoisVAE(
        type='poisson',
        dataset=DATASET,
        data_dir=DATA_DIR,
        base_dir=CHECKPOINT_DIR,
        n_latents=N_LATENTS,
        n_ch=N_CH,
        enc_type=ENC_TYPE,
        dec_type=DEC_TYPE,
        enc_bias=ENC_BIAS,
        dec_bias=DEC_BIAS,
        enc_norm=ENC_NORM,
        dec_norm=DEC_NORM,
        fit_prior=FIT_PRIOR,
        prior_clamp=PRIOR_CLAMP,
        prior_log_dist=PRIOR_LOG_DIST,
        hard_fwd=HARD_FWD,
        exc_only=EXC_ONLY,
        rmax_q=RMAX_Q,
        activation_fn=ACTIVATION_FN,
        init_dist=INIT_DIST,
        init_scale=INIT_SCALE,
        res_eps=RES_EPS,
        use_bn=USE_BN,
        use_se=USE_SE,
        seed=SEED,
    )
    
    # Create training config
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
    
    # Save configs
    print("Saving configurations...")
    config_dict = {
        'model_config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                        for k, v in vars(model_cfg).items() if not k.startswith('_')},
        'train_config': {k: str(v) if not isinstance(v, (int, float, bool, str, dict, type(None))) else v 
                        for k, v in vars(train_cfg).items() if not k.startswith('_')},
    }
    config_path = Path(CHECKPOINT_DIR) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Configs saved to {config_path}\n")
    
    # Initialize model
    print("Initializing model...")
    model = PoissonVAE(model_cfg)
    model = model.to(device)
    
    # Override checkpoint directory
    def simple_chkpt_dir(self, fit_name=None):
        self.chkpt_dir = CHECKPOINT_DIR
        os.makedirs(self.chkpt_dir, exist_ok=True)
    
    model.create_chkpt_dir = lambda fit_name=None: simple_chkpt_dir(model, fit_name)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized on {device}")
    print(f"  Trainable parameters: {n_params:,}\n")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = TrainerVAE(
        model=model,
        cfg_train=train_cfg,
        dl_trn=dl_trn,
        dl_vld=dl_vld,
        device=device,
    )
    print("✓ Trainer initialized\n")
    
    # Train
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.fit()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final checkpoint saved to: {CHECKPOINT_DIR}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    model.eval()
    with torch.no_grad():
        x_val = next(iter(dl_vld))[0].to(device)
        dist, log_dr, spks, y = model(x_val)
        kl_diag = model.loss_kl(log_dr)
        mse = torch.mean((x_val - y)**2, dim=[1,2,3])
        kl_per_sample = torch.sum(kl_diag, dim=1)
        recon_loss = torch.mean(mse) * (model.cfg.input_sz ** 2)
        kl_loss = torch.mean(kl_per_sample)
        neg_elbo = recon_loss + kl_loss
        
        print(f"Negative ELBO: {neg_elbo.item():.2f}")
        print(f"Reconstruction Loss: {recon_loss.item():.2f}")
        print(f"KL Divergence: {kl_loss.item():.2f}")
    
    print("\n✓ Training script completed successfully!")

if __name__ == "__main__":
    main()
