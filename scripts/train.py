"""
HybridEditDif — Multi-GPU Training Script
==========================================
Implements training per paper Section 4.1:

  Loss (Eq. 11):
    L = E_{t,y0,ε} || ε_θ(y_t, m̄⊙X_s, c_i, c_t, t) - ε ||²₂

  Training setup:
    - Optimizer: AdamW, lr=1e-4, weight_decay=0.01%
    - Dataset: OpenImages (1.9M images, 512×512)
    - 30 epochs on single V100 (~10 days)
    - Multi-GPU: DDP via accelerate
    - Random dropout of image/text conditions with prob=0.05

Usage:
    # Single-node multi-GPU (e.g. 4 GPUs)
    accelerate launch --num_processes 4 train.py --config configs/train_config.yaml

    # Or with explicit GPU selection
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train.py
"""

import os
import sys
import math
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers.optimization import get_cosine_schedule_with_warmup
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.hybrid_edit_dif import HybridEditDif
from src.data.openimages_dataset import (
    OpenImagesDownloader,
    OpenImagesEditingDataset,
    collate_fn,
    get_dataloaders,
)
from src.utils.metrics import compute_validation_metrics

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Training Step
# ══════════════════════════════════════════════════════════════════════════════

def training_step(
    model: HybridEditDif,
    batch: dict,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Single training step implementing Eq. (11).

    Pipeline:
    1. Encode source image X_s to latent z_0
    2. Encode masked source (1-m)·X_s to latent z_masked
    3. Sample timestep t, add noise → z_t
    4. Encode reference X_r → c_i (CLIP image)
    5. Encode text T → c_t (CLIP text)
    6. Encode text → SD text embeddings (for original UNet cross-attn)
    7. Concatenate [z_t, z_masked, mask] → 9-channel input
    8. Predict noise ε̂
    9. L = MSE(ε̂, ε)  [masked loss on inpainted region only]
    """
    device = accelerator.device

    # ── Unwrap DDP wrapper ────────────────────────────────────────────────────
    # accelerator.prepare() membungkus model dengan DistributedDataParallel.
    # Custom methods (encode_image_to_latent, dll) tidak bisa diakses langsung
    # dari DDP wrapper — harus lewat unwrap_model() untuk akses atribut.
    # Forward pass tetap pakai `model` (DDP) agar gradient sync benar.
    m = accelerator.unwrap_model(model)

    source        = batch["source"].to(device, dtype=weight_dtype)        # [B,3,H,W]
    masked_source = batch["masked_source"].to(device, dtype=weight_dtype) # [B,3,H,W]
    reference     = batch["reference"].to(device, dtype=weight_dtype)     # [B,3,224,224]
    mask          = batch["mask"].to(device, dtype=weight_dtype)          # [B,1,H,W]
    text_tokens   = batch["text_tokens"].to(device)                       # [B,seq]
    drop_image    = batch["drop_image"]
    drop_text     = batch["drop_text"]

    B = source.shape[0]

    # ── Step 1-2: VAE encode ──────────────────────────────────────────────────
    with torch.no_grad():
        latents        = m.encode_image_to_latent(source)         # [B,4,h,w]
        masked_latents = m.encode_image_to_latent(masked_source)  # [B,4,h,w]
        h, w = latents.shape[2], latents.shape[3]
        mask_latent = F.interpolate(mask, size=(h, w), mode='nearest')  # [B,1,h,w]

    # ── Step 3: Add noise ─────────────────────────────────────────────────────
    noise     = torch.randn_like(latents)
    timesteps = torch.randint(
        0, m.noise_scheduler.config.num_train_timesteps,
        (B,), device=device
    ).long()
    noisy_latents = m.noise_scheduler.add_noise(latents, noise, timesteps)

    # ── Step 4: Image conditioning c_i (Eq. 6) ───────────────────────────────
    with torch.no_grad():
        image_context = m.image_encoder(reference)      # [B, 257, 1024]

    # ── Step 5: Text conditioning c_t (Eq. 7) ────────────────────────────────
    with torch.no_grad():
        text_context = m.clip_text_encoder(text_tokens) # [B, 1, 1024]

    # ── Step 6: SD text embeddings (frozen SD cross-attn) ────────────────────
    with torch.no_grad():
        sd_text_emb = m.text_encoder(text_tokens)[0]    # [B, seq, 768]

    # ── Step 7-8: Forward pass via DDP wrapper (gradient sync) ───────────────
    batch_drop_image = any(drop_image)
    batch_drop_text  = any(drop_text)

    noise_pred = model(  # pakai `model` (DDP), bukan `m`
        noisy_latents=noisy_latents,
        masked_image_latents=masked_latents,
        mask_latents=mask_latent,
        timesteps=timesteps,
        sd_text_embeddings=sd_text_emb,
        image_context=image_context,
        text_context=text_context,
        drop_image_cond=batch_drop_image,
        drop_text_cond=batch_drop_text,
    )

    # ── Step 9: Loss ──────────────────────────────────────────────────────────
    pred_type = m.noise_scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "v_prediction":
        target = m.noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

    # Masked loss: only on inpainted region
    loss = F.mse_loss(
        noise_pred.float() * mask_latent,
        target.float()     * mask_latent,
        reduction="mean"
    )

    return loss


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(config_path: str):
    config = OmegaConf.load(config_path)

    # ── Accelerator setup (multi-GPU DDP) ─────────────────────────────────────
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if config.get("use_wandb", False) else None,
        project_config=project_config,
    )

    set_seed(config.training.get("seed", 42))
    # Tentukan dtype sesuai mixed_precision config (fp16 / bf16 / no)
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(accelerator.mixed_precision, torch.float32)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output dir: {config.output_dir}")
        logger.info(f"Num GPUs: {accelerator.num_processes}")
        logger.info(OmegaConf.to_yaml(config))

    # ── Model ─────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Initializing HybridEditDif model...")

    model = HybridEditDif(
        sd_model_path=config.model.sd_model_path,
        image_context_dim=config.model.get("image_context_dim", 1024),
        text_context_dim=config.model.get("text_context_dim", 1024),
        lambda1=config.model.get("lambda1", 1.0),
        lambda2=config.model.get("lambda2", 1.0),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Loading dataset...")

    import json

    # Cek apakah ada JSON annotations dari scripts/download_openimages.py
    # JSON lebih cepat dimuat dan sudah berformat yang benar
    json_bbox_path = Path(config.data.data_root) / "annotations" / "train_bbox_annotations.json"

    if json_bbox_path.exists():
        logger.info(f"Loading bbox annotations from JSON: {json_bbox_path}")
        with open(json_bbox_path) as f:
            raw = json.load(f)
        # JSON menyimpan list-of-list, konversi ke list-of-tuple
        bbox_annotations = {
            img_id: [tuple(b) for b in bboxes]
            for img_id, bboxes in raw.items()
        }
        logger.info(f"Loaded {len(bbox_annotations):,} images with bbox annotations")
    else:
        # Fallback: download annotations CSV dari OpenImages (bisa lama, file ~5GB)
        logger.info("JSON annotations tidak ditemukan, fallback ke CSV downloader...")
        downloader = OpenImagesDownloader(config.data.data_root)
        bbox_annotations = downloader.load_bbox_annotations(
            split="train",
            max_images=config.data.get("max_images", 200000),
        )

    # Load text annotations if available
    text_annotations = None
    text_ann_path = Path(config.data.data_root) / "annotations" / "text_annotations.json"
    if text_ann_path.exists():
        with open(text_ann_path) as f:
            text_annotations = json.load(f)

    train_loader, val_loader = get_dataloaders(
        images_dir=os.path.join(config.data.data_root, "images", "train"),
        bbox_annotations=bbox_annotations,
        text_annotations=text_annotations,
        image_size=config.data.get("image_size", 512),
        batch_size=config.training.train_batch_size,
        num_workers=config.data.get("num_workers", 8),
        max_samples=config.data.get("max_samples", None),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.training.get("weight_decay", 1e-4),
        eps=1e-8,
    )

    # ── LR Scheduler (cosine with warmup) ────────────────────────────────────
    num_training_steps = config.training.num_epochs * len(train_loader)
    num_warmup_steps   = config.training.get("warmup_steps", 500)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ── Accelerate prepare (handles DDP) ─────────────────────────────────────
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # ── Cast semua frozen components ke weight_dtype ──────────────────────────
    # Setelah accelerator.prepare(), semua component harus dtype yang sama dengan
    # input tensor (weight_dtype). Frozen encoders di-cast ke bf16/fp16 agar
    # tidak ada RuntimeError: "Input type != weight type".
    device = accelerator.device
    frozen = accelerator.unwrap_model(model)
    frozen.vae.to(device, dtype=weight_dtype)
    frozen.image_encoder.to(device, dtype=weight_dtype)       # CLIP image encoder
    frozen.clip_text_encoder.to(device, dtype=weight_dtype)   # CLIP text encoder
    frozen.text_encoder.to(device, dtype=weight_dtype)        # SD text encoder (CLIP ViT-L)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    ckpt_dir = Path(config.output_dir) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if config.training.get("resume_from_checkpoint"):
        ckpt_path = config.training.resume_from_checkpoint
        if ckpt_path == "latest":
            ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
            ckpt_path = str(ckpts[-1]) if ckpts else None
        if ckpt_path and Path(ckpt_path).exists():
            logger.info(f"Resuming from: {ckpt_path}")
            accelerator.load_state(ckpt_path)
            global_step = int(Path(ckpt_path).name.split("-")[-1])
            start_epoch = global_step // len(train_loader)

    # ── Training loop ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Starting HybridEditDif training")
        logger.info(f"  Epochs:      {config.training.num_epochs}")
        logger.info(f"  Steps/epoch: {len(train_loader)}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Batch size:  {config.training.train_batch_size} × {accelerator.num_processes} GPUs")
        logger.info(f"  Effective:   {config.training.train_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps}")
        logger.info("=" * 60)

    progress_bar = tqdm(
        range(num_training_steps),
        initial=global_step,
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss = training_step(model, batch, accelerator, weight_dtype)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                epoch_steps += 1

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "epoch": f"{epoch+1}/{config.training.num_epochs}",
                })

                # ── Logging ───────────────────────────────────────────────────
                if global_step % config.training.get("log_every", 50) == 0:
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step} | Epoch {epoch+1} | "
                            f"Loss: {loss.item():.4f} | "
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                        )

                # ── Checkpoint ────────────────────────────────────────────────
                if global_step % config.training.get("save_every", 2000) == 0:
                    if accelerator.is_main_process:
                        save_path = ckpt_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(save_path))
                        logger.info(f"Saved checkpoint: {save_path}")

                # ── Validation ────────────────────────────────────────────────
                if global_step % config.training.get("eval_every", 5000) == 0:
                    if accelerator.is_main_process:
                        logger.info("Running validation...")
                        # (inference + metrics computed in evaluate.py)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    # ── Final save ────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = Path(config.output_dir) / "final_model"
        unwrapped = accelerator.unwrap_model(model)
        # Save DDCA weights + MLP heads (trainable parts only)
        torch.save(
            {
                "ddca_layers": unwrapped.ddca_layers.state_dict(),
                "image_encoder_mlp": unwrapped.image_encoder.mlp.state_dict(),
                "clip_text_encoder_mlp": unwrapped.clip_text_encoder.mlp.state_dict(),
                "config": OmegaConf.to_container(config),
            },
            final_path / "hybrid_edit_dif_weights.pt"
        )
        logger.info(f"Training complete! Model saved to: {final_path}")
        logger.info("Run evaluate.py for COCOEE/MagicBrush/EmuEdit benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
