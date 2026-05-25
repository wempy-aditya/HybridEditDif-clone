"""
HybridEditDif — Main Model
============================
Reconstructed from Liu et al., Pattern Recognition 2026.

Architecture Overview:
  - Base: Stable Diffusion (frozen UNet + VAE + CLIP text encoder)
  - Addition: 16 Dynamic Decoupled Cross-Attention (DDCA) layers
              injected into UNet cross-attention positions
  - Image conditioning: OpenCLIP ViT-H/14 → MLP → DDCA image branch
  - Text conditioning:  OpenCLIP ViT-g/14 → MLP → DDCA text branch

Training (Section 4.1, Eq. 11):
  L = E_{t,y0,ε} || ε_θ(y_t, m̄⊙X_s, c_i, c_t, t) - ε ||²₂

Inference (Eq. 12, classifier-free guidance):
  ε̂_θ = w1·ε_θ(..., c_t) + w2·ε_θ(..., c_i) + (1-w1-w2)·ε_θ(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Optional, Tuple, List
import logging

from .attention import DynamicDecoupledCrossAttention
from .encoders import CLIPImageEncoder, CLIPTextEncoder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DDCA Injection Wrapper for UNet Cross-Attention
# ══════════════════════════════════════════════════════════════════════════════

class DDCAInjectedAttention(nn.Module):
    """
    Wraps an existing UNet cross-attention layer and adds the DDCA module
    in parallel. The original SD cross-attention handles text (frozen),
    while the new DDCA handles the multimodal fusion.

    Per paper: "we incorporated 16 cross-attention layers and added new
    image cross-attention layers to each"
    """

    def __init__(
        self,
        original_attn: nn.Module,
        query_dim: int,
        image_context_dim: int = 1024,
        text_context_dim: int = 1024,
        heads: int = 8,
    ):
        super().__init__()
        self.original_attn = original_attn  # frozen SD cross-attn

        # New DDCA layer (trainable)
        self.ddca = DynamicDecoupledCrossAttention(
            query_dim=query_dim,
            image_context_dim=image_context_dim,
            text_context_dim=text_context_dim,
            heads=heads,
            dim_head=query_dim // heads,
        )

        # Scale factor for DDCA output (initialized small to not disrupt SD)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # DDCA-specific contexts (passed via hooks)
        image_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:

        # Original SD cross-attention output (frozen)
        original_out = self.original_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

        # DDCA output (trainable), only if contexts are available
        if image_context is not None or text_context is not None:
            # Use SD encoder_hidden_states as text_context if not separately provided
            if text_context is None:
                text_context = encoder_hidden_states

            ddca_out = self.ddca(
                hidden_states,
                image_context=image_context,
                text_context=text_context,
                lambda1=lambda1,
                lambda2=lambda2,
            )
            # Additive injection with learned scale (zero-init → safe training start)
            return original_out + torch.tanh(self.scale) * ddca_out

        return original_out


# ══════════════════════════════════════════════════════════════════════════════
# HybridEditDif Main Model
# ══════════════════════════════════════════════════════════════════════════════

class HybridEditDif(nn.Module):
    """
    Full HybridEditDif model.

    Components:
        - SD UNet (frozen base weights, DDCA layers trainable)
        - SD VAE (frozen)
        - SD Text Encoder (frozen)
        - CLIPImageEncoder with MLP (trainable MLP only)
        - CLIPTextEncoder with MLP (trainable MLP only)
        - 16 DDCAInjectedAttention layers (trainable)

    Trainable parameters: ~only MLP heads + DDCA weights
    Frozen: VAE, SD UNet base weights, SD text encoder, CLIP encoders
    """

    # SD v1.5 UNet cross-attention layer dimensions
    # (query_dim varies by resolution level)
    UNET_ATTN_DIMS = {
        "down": [320, 320, 640, 640, 1280, 1280],
        "mid":  [1280],
        "up":   [1280, 1280, 640, 640, 320, 320],
    }

    def __init__(
        self,
        sd_model_path: str = "runwayml/stable-diffusion-v1-5",
        image_context_dim: int = 1024,
        text_context_dim: int = 1024,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        logger.info(f"Loading Stable Diffusion from: {sd_model_path}")

        # ── Load SD components ───────────────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd_model_path, subfolder="scheduler"
        )

        # Freeze SD base components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # ── Load CLIP encoders ───────────────────────────────────────────────
        logger.info("Loading CLIP image encoder (ViT-H/14)...")
        self.image_encoder = CLIPImageEncoder(
            output_dim=image_context_dim, freeze=True
        )

        logger.info("Loading CLIP text encoder (ViT-g/14)...")
        self.clip_text_encoder = CLIPTextEncoder(
            output_dim=text_context_dim, freeze=True
        )

        # ── Inject DDCA into UNet ─────────────────────────────────────────────
        logger.info("Injecting 16 DDCA layers into UNet...")
        self._inject_ddca(image_context_dim, text_context_dim)

        # ── Expand UNet conv_in: 4 → 9 channels (inpainting format) ──────────
        # Paper: UNet input = [z_t (4ch) | masked_z_s (4ch) | mask (1ch)] = 9ch
        # Standard SD1.5 conv_in expects 4ch → must be expanded.
        # New channels initialized to 0 → training starts as identity on z_t.
        self._expand_unet_conv_in(in_channels=9)

        logger.info("HybridEditDif initialized.")
        self._log_trainable_params()

    def _inject_ddca(self, image_context_dim: int, text_context_dim: int):
        """
        Inject DDCA into all 16 cross-attention layers of the SD UNet.
        Paper: "we incorporated 16 cross-attention layers and added new
        image cross-attention layers to each"
        """
        self.ddca_layers = nn.ModuleList()
        injected = 0

        def _inject_into_block(block):
            nonlocal injected
            if injected >= 16:
                return
            # SD UNet uses BasicTransformerBlock with attn1 (self) + attn2 (cross)
            for name, module in block.named_modules():
                if hasattr(module, 'attn2') and injected < 16:
                    # attn2 is the cross-attention in SD
                    orig_attn = module.attn2
                    query_dim = orig_attn.to_q.in_features if hasattr(orig_attn, 'to_q') else 320
                    heads = orig_attn.heads if hasattr(orig_attn, 'heads') else 8

                    ddca_attn = DDCAInjectedAttention(
                        original_attn=orig_attn,
                        query_dim=query_dim,
                        image_context_dim=image_context_dim,
                        text_context_dim=text_context_dim,
                        heads=heads,
                    )
                    module.attn2 = ddca_attn
                    self.ddca_layers.append(ddca_attn)
                    injected += 1

        # Traverse UNet blocks
        for block in self.unet.down_blocks:
            _inject_into_block(block)
        _inject_into_block(self.unet.mid_block)
        for block in self.unet.up_blocks:
            _inject_into_block(block)

        logger.info(f"  ✓ Injected {injected} DDCA layers")

    def _log_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,} params "
                    f"({100*trainable/total:.1f}%)")

    def _expand_unet_conv_in(self, in_channels: int = 9):
        """
        Expand UNet conv_in from 4 → 9 channels for inpainting.

        SD v1.5 UNet conv_in: Conv2d(4, 320, 3, padding=1)
        After expansion:       Conv2d(9, 320, 3, padding=1)

        Strategy (per SD-Inpainting paper):
          - Copy original 4-ch weights as-is (preserves pretrained features)
          - Initialize extra 5 channels (masked_latent + mask) with zeros
          - This ensures at t=0, model behaves identically to SD base
        """
        old_conv = self.unet.conv_in
        old_in   = old_conv.in_channels  # 4

        if old_in == in_channels:
            logger.info(f"  conv_in already has {in_channels} channels, skip.")
            return

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            # Copy pretrained weights for first `old_in` channels
            new_conv.weight[:, :old_in] = old_conv.weight
            # Zero-init new channels (masked latent + mask)
            new_conv.weight[:, old_in:] = 0.0
            # Copy bias unchanged
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv
        # Update config so UNet knows its new input size
        self.unet.config.in_channels = in_channels
        logger.info(f"  ✓ UNet conv_in expanded: {old_in}ch → {in_channels}ch "
                    f"(zero-init for extra {in_channels - old_in} channels)")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode reference image to conditioning tokens c_i (Eq. 6)."""
        return self.image_encoder(image)

    def encode_text_clip(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text to conditioning tokens c_t (Eq. 7)."""
        return self.clip_text_encoder(text_tokens)

    def _set_ddca_contexts(
        self,
        image_context: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor],
        lambda1: float,
        lambda2: float,
    ):
        """Store contexts for DDCA layers to use during UNet forward pass."""
        for layer in self.ddca_layers:
            layer._current_image_context = image_context
            layer._current_text_context  = text_context
            layer._current_lambda1       = lambda1
            layer._current_lambda2       = lambda2

    def forward(
        self,
        noisy_latents: torch.Tensor,       # [B, 4, h, w]
        masked_image_latents: torch.Tensor, # [B, 4, h, w]
        mask_latents: torch.Tensor,         # [B, 1, h, w]
        timesteps: torch.Tensor,
        sd_text_embeddings: torch.Tensor,  # from frozen SD text encoder
        image_context: Optional[torch.Tensor] = None,   # c_i [B, 257, 1024]
        text_context: Optional[torch.Tensor] = None,    # c_t [B, 1, 1024]
        lambda1: Optional[float] = None,
        lambda2: Optional[float] = None,
        drop_image_cond: bool = False,
        drop_text_cond: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass implementing Eq. (11) loss computation.

        Input to UNet (concatenated): [noisy_latents, masked_image_latents, mask]
        = 4 + 4 + 1 = 9 channels (SD inpainting format)
        """
        lam1 = lambda1 if lambda1 is not None else self.lambda1
        lam2 = lambda2 if lambda2 is not None else self.lambda2

        # Classifier-free guidance dropout during training
        if drop_image_cond:
            image_context = torch.zeros_like(image_context) if image_context is not None else None
        if drop_text_cond:
            text_context = torch.zeros_like(text_context) if text_context is not None else None

        # Set DDCA contexts (accessed during UNet forward)
        self._set_ddca_contexts(image_context, text_context, lam1, lam2)

        # Concatenate latent input (9-channel SD inpainting format)
        latent_input = torch.cat(
            [noisy_latents, masked_image_latents, mask_latents], dim=1
        )  # [B, 9, h, w]

        # UNet forward (with DDCA layers active)
        noise_pred = self.unet(
            latent_input,
            timesteps,
            encoder_hidden_states=sd_text_embeddings,
        ).sample

        return noise_pred

    @torch.no_grad()
    def encode_image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """VAE encode image to latent space."""
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode latent to image."""
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample

    @torch.no_grad()
    def encode_sd_text(self, text_list: List[str], device: torch.device) -> torch.Tensor:
        """Encode text using frozen SD text encoder (for UNet conditioning)."""
        tokens = self.tokenizer(
            text_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(tokens)[0]

    def get_trainable_parameters(self):
        """Return only trainable parameters (DDCA + MLP heads)."""
        return [p for p in self.parameters() if p.requires_grad]
