"""
HybridEditDif — Inference Pipeline
=====================================
Implements classifier-free guidance per paper Eq. (12):

  ε̂_θ(y_t, m̄⊙X_s, c_t, c_i, t) =
      w1 · ε_θ(y_t, m̄⊙X_s, c_t, t)           [text-conditioned]
    + w2 · ε_θ(y_t, m̄⊙X_s, c_i, t)           [image-conditioned]
    + (1 - w1 - w2) · ε_θ(y_t, m̄⊙X_s, t)     [unconditional]

Supports:
  - Text-only editing (λ2=0, w2=0)
  - Image-only editing (λ1=0, w1=0)
  - Multimodal editing (both active)
  - Batch inference for evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Union
import logging
from tqdm import tqdm

from diffusers import DDIMScheduler, PNDMScheduler
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class HybridEditDifInferencePipeline:
    """
    Inference pipeline for HybridEditDif.
    Handles the full denoising loop with classifier-free guidance.
    """

    def __init__(
        self,
        model,
        scheduler_type: str = "ddim",
        num_inference_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Swap to DDIM for faster inference
        if scheduler_type == "ddim":
            self.scheduler = DDIMScheduler.from_config(
                model.noise_scheduler.config
            )
        else:
            self.scheduler = model.noise_scheduler

        self.num_inference_steps = num_inference_steps

        # Image preprocessing for CLIP
        self.clip_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        # Source image preprocessing
        self.source_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # Keep as-is for varying sizes
        ])

        import open_clip
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-g-14')

    @torch.no_grad()
    def __call__(
        self,
        source_image: Image.Image,        # Original image to edit
        mask_image: Image.Image,          # Binary mask (white=edit, black=keep)
        reference_image: Optional[Image.Image] = None,   # X_r
        text_prompt: Optional[str] = None,               # T
        w1: float = 7.5,                 # text guidance weight
        w2: float = 7.5,                 # image guidance weight
        lambda1: float = 1.0,            # text branch weight in DDCA
        lambda2: float = 1.0,            # image branch weight in DDCA
        seed: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Run HybridEditDif inference.

        Args:
            source_image    : PIL Image, the source to edit
            mask_image      : PIL Image, white=inpaint region, black=preserve
            reference_image : PIL Image, reference for image conditioning
            text_prompt     : text description for editing
            w1              : classifier-free guidance scale for text
            w2              : classifier-free guidance scale for image
            lambda1, lambda2: DDCA branch weights (set to 0 to disable modality)
            seed            : random seed
            output_size     : (W, H) output size; defaults to source size

        Returns:
            Edited PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        orig_size = source_image.size
        target_size = output_size or orig_size

        # ── Preprocess inputs ─────────────────────────────────────────────────
        # Resize to 512×512 for SD (standard)
        PROC_SIZE = 512
        source_512 = source_image.resize((PROC_SIZE, PROC_SIZE), Image.LANCZOS)
        mask_512   = mask_image.resize((PROC_SIZE, PROC_SIZE), Image.NEAREST)

        # Source tensor [1, 3, 512, 512] in [-1, 1]
        source_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])(source_512).unsqueeze(0).to(self.device)

        # Mask tensor [1, 1, 512, 512]
        mask_np = np.array(mask_512.convert("L")) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # ── Encode source to latent ───────────────────────────────────────────
        source_latent = self.model.encode_image_to_latent(source_tensor)  # [1,4,h,w]

        # Masked source (1-m)·X_s
        masked_source = source_tensor * (1.0 - mask_tensor)
        masked_latent = self.model.encode_image_to_latent(masked_source)  # [1,4,h,w]

        # Latent-space mask
        h, w = source_latent.shape[2:]
        mask_latent = F.interpolate(mask_tensor, size=(h, w), mode='nearest')  # [1,1,h,w]

        # ── Encode conditions ─────────────────────────────────────────────────
        # Image conditioning c_i
        image_context = None
        if reference_image is not None and lambda2 != 0.0:
            ref_tensor = self.clip_transform(reference_image).unsqueeze(0).to(self.device)
            image_context = self.model.image_encoder(ref_tensor)  # [1, 257, 1024]

        # Text conditioning c_t + SD text embeddings
        text_context    = None
        sd_text_emb     = None
        null_sd_text    = None

        if text_prompt is not None and lambda1 != 0.0:
            text_tokens = self.clip_tokenizer([text_prompt]).to(self.device)
            text_context = self.model.clip_text_encoder(text_tokens)  # [1, 1, 1024]

        # SD text embeddings (always needed for original cross-attn)
        sd_text_emb  = self.model.encode_sd_text([text_prompt or ""], self.device)  # [1,77,768]
        null_sd_text = self.model.encode_sd_text([""], self.device)

        # Null contexts for CFG
        null_image_context = torch.zeros_like(image_context) if image_context is not None else None
        null_text_context  = torch.zeros_like(text_context)  if text_context  is not None else None

        # ── Denoising loop (Eq. 12) ───────────────────────────────────────────
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latent with noise in masked region
        latents = torch.randn_like(source_latent)
        # Blend: keep source info in unmasked region
        latents = latents * mask_latent + source_latent * (1.0 - mask_latent)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            t_batch = t.unsqueeze(0)

            # ── Three forward passes for Eq. (12) ────────────────────────────
            # 1) Text-conditioned
            eps_text = self.model(
                noisy_latents=latents,
                masked_image_latents=masked_latent,
                mask_latents=mask_latent,
                timesteps=t_batch,
                sd_text_embeddings=sd_text_emb,
                image_context=null_image_context,
                text_context=text_context,
                lambda1=lambda1,
                lambda2=0.0,         # image branch off
            )

            # 2) Image-conditioned
            eps_image = self.model(
                noisy_latents=latents,
                masked_image_latents=masked_latent,
                mask_latents=mask_latent,
                timesteps=t_batch,
                sd_text_embeddings=null_sd_text,
                image_context=image_context,
                text_context=null_text_context,
                lambda1=0.0,         # text branch off
                lambda2=lambda2,
            )

            # 3) Unconditional
            eps_uncond = self.model(
                noisy_latents=latents,
                masked_image_latents=masked_latent,
                mask_latents=mask_latent,
                timesteps=t_batch,
                sd_text_embeddings=null_sd_text,
                image_context=null_image_context,
                text_context=null_text_context,
                lambda1=0.0,
                lambda2=0.0,
            )

            # Guidance (Eq. 12):
            # ε̂ = w1·eps_text + w2·eps_image + (1-w1-w2)·eps_uncond
            eps = (
                w1 * eps_text
                + w2 * eps_image
                + (1.0 - w1 - w2) * eps_uncond
            )

            # Scheduler step
            latents = self.scheduler.step(eps, t, latents).prev_sample

            # Re-apply source latent to unmasked region each step
            # (preserves background — critical for inpainting quality)
            noisy_source = self.scheduler.add_noise(
                source_latent, torch.randn_like(source_latent), t.unsqueeze(0)
            )
            latents = latents * mask_latent + noisy_source * (1.0 - mask_latent)

        # ── Decode latent to image ────────────────────────────────────────────
        decoded = self.model.decode_latent(latents)  # [1,3,512,512] in [-1,1]
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
        decoded_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_pil = Image.fromarray(decoded_np)

        # Resize back to original (or target) size
        if output_pil.size != target_size:
            output_pil = output_pil.resize(target_size, Image.LANCZOS)

        return output_pil

    @torch.no_grad()
    def batch_infer(
        self,
        samples: List[dict],
        **kwargs,
    ) -> List[Image.Image]:
        """
        Run inference on a list of samples (for evaluation).
        Each sample: {"source": PIL, "mask": PIL, "reference": PIL, "text": str}
        """
        results = []
        for sample in tqdm(samples, desc="Batch inference"):
            out = self(
                source_image=sample["source"],
                mask_image=sample["mask"],
                reference_image=sample.get("reference"),
                text_prompt=sample.get("text"),
                **kwargs,
            )
            results.append(out)
        return results
