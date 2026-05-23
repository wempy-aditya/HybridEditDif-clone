"""
Image & Text Encoders for HybridEditDif
=========================================
Reconstructed from paper Section 3.2 (Model Designs) and Section 3.3.

Image Encoder (Eq. 6):
    c_i = MLP(CLIP_ViT-H/14(A(X_r)))
    - Uses OpenCLIP ViT-H/14
    - Encodes to 257 tokens (1 class + 256 patch tokens)
    - Projects through MLP to match UNet cross-attention dim

Text Encoder (Eq. 7):
    c_t = MLP(CLIP_ViT-gopt-16-SigLIP2-384(T))
    - Uses OpenCLIP ViT-g/14 (closest publicly available to paper's SigLIP2)
    - Text features extracted and projected via MLP

Reference Image Augmentation (Section 3.2):
    - Rotation: ±45°
    - Scale: 0.8–1.2
    - Crop: 80–100% of original
    - Color: brightness/contrast/saturation ±10%
"""

import torch
import torch.nn as nn
import open_clip
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math
from PIL import Image
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# MLP Projection Head (used by both encoders)
# ══════════════════════════════════════════════════════════════════════════════

class MLPProjection(nn.Module):
    """
    2-layer MLP to project CLIP features into UNet cross-attention dimension.
    Paper: "image features output by the CLIP encoder are initially processed
    through a fully connected (FC) layer"
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or (in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Reference Image Augmentation (Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════

class ReferenceImageAugmentation:
    """
    Data augmentation for reference images during training.
    Simulates diverse real-world conditions per paper Section 3.2:
      - Rotation: -45° to +45°
      - Scale: 0.8 to 1.2
      - Crop: 80% to 100% of original
      - Color: brightness/contrast/saturation ±10%
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        # Random rotation: ±45°
        angle = random.uniform(-45, 45)
        image = TF.rotate(image, angle, expand=False, fill=0)

        # Random scale: 0.8–1.2
        scale = random.uniform(0.8, 1.2)
        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Random crop: 80–100% of resized image
        crop_factor = random.uniform(0.8, 1.0)
        cw = int(new_w * crop_factor)
        ch = int(new_h * crop_factor)
        x_offset = random.randint(0, max(0, new_w - cw))
        y_offset = random.randint(0, max(0, new_h - ch))
        image = image.crop((x_offset, y_offset, x_offset + cw, y_offset + ch))

        # Color adjustments: ±10%
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor   = random.uniform(0.9, 1.1)
        saturation_factor = random.uniform(0.9, 1.1)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        image = TF.adjust_saturation(image, saturation_factor)

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


# ══════════════════════════════════════════════════════════════════════════════
# Image Encoder: CLIP ViT-H/14 (Eq. 6)
# ══════════════════════════════════════════════════════════════════════════════

class CLIPImageEncoder(nn.Module):
    """
    Image encoder using OpenCLIP ViT-H/14.
    Per paper: encodes reference image X_r to 257 tokens (1 cls + 256 patch),
    then projects through MLP to match UNet cross-attention dim.

    Eq. (6): c_i = MLP(CLIP_ViT-H/14(A(X_r)))
    where A(·) is the reference image augmentation.
    """

    def __init__(
        self,
        output_dim: int = 1024,   # UNet cross-attention projection dim
        freeze: bool = True,
    ):
        super().__init__()

        # Load OpenCLIP ViT-H/14 (paper uses this specifically)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14',
            pretrained='laion2b_s32b_b79k'
        )
        self.clip_model = self.clip_model.visual  # visual encoder only

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # ViT-H/14 outputs: 1280-dim features, 257 tokens
        clip_hidden_dim = 1280

        # MLP projection per paper
        self.mlp = MLPProjection(
            in_dim=clip_hidden_dim,
            out_dim=output_dim,
        )

        self.augmentation = ReferenceImageAugmentation(image_size=224)

    def encode(
        self,
        image: torch.Tensor,   # [B, 3, 224, 224], normalized
    ) -> torch.Tensor:
        """
        Returns 257 tokens (cls + patch) projected to output_dim.
        Output shape: [B, 257, output_dim]
        """
        with torch.no_grad():
            # Get patch tokens + cls token from ViT
            # open_clip stores intermediate features via hooks
            features = self._extract_all_tokens(image)  # [B, 257, 1280]

        projected = self.mlp(features)  # [B, 257, output_dim]
        return projected

    def _extract_all_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Extract all 257 tokens (cls + 256 patch) from ViT-H/14."""
        # Patch embedding
        x = self.clip_model.conv1(image)           # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B, width, grid^2]
        x = x.permute(0, 2, 1)                     # [B, grid^2, width]

        # Prepend class token
        cls = self.clip_model.class_embedding.unsqueeze(0).unsqueeze(0)
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)             # [B, 257, width]

        # Add positional embedding
        x = x + self.clip_model.positional_embedding

        # Pre-norm
        x = self.clip_model.ln_pre(x)

        # Transformer blocks
        x = x.permute(1, 0, 2)                     # [seq, B, width]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                     # [B, seq, width]

        # Post-norm
        x = self.clip_model.ln_post(x)             # [B, 257, 1280]

        return x

    def forward(
        self,
        image: torch.Tensor,
        apply_augmentation: bool = False,
    ) -> torch.Tensor:
        if apply_augmentation:
            # Apply augmentation per-sample (training only)
            # image here is PIL; augmentation happens in dataset
            pass
        return self.encode(image)


# ══════════════════════════════════════════════════════════════════════════════
# Text Encoder: OpenCLIP (Eq. 7)
# ══════════════════════════════════════════════════════════════════════════════

class CLIPTextEncoder(nn.Module):
    """
    Text encoder using OpenCLIP.
    Paper uses ViT-g/14-SigLIP2-384; we use ViT-g-14 (closest public equivalent).
    Frozen during training per paper.

    Eq. (7): c_t = MLP(CLIP_ViT-gopt-16-SigLIP2-384(T))
    """

    def __init__(
        self,
        output_dim: int = 1024,
        freeze: bool = True,
    ):
        super().__init__()

        # ViT-g/14 — closest public model to paper's SigLIP2 variant
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-g-14',
            pretrained='laion2b_s12b_b42k'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # ViT-g outputs 1024-dim text features
        clip_text_dim = 1024

        self.mlp = MLPProjection(
            in_dim=clip_text_dim,
            out_dim=output_dim,
        )

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Encode list of text strings to projected features.
        Returns: [B, seq_t, output_dim]
        """
        tokens = self.tokenizer(texts).to(device)

        with torch.no_grad():
            # Get token-level features (not just pooled)
            text_features = self.clip_model.encode_text(tokens)  # [B, 1024]
            # Expand to sequence format for cross-attention
            text_features = text_features.unsqueeze(1)           # [B, 1, 1024]

        projected = self.mlp(text_features)  # [B, 1, output_dim]
        return projected

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip_model.encode_text(text_tokens)
        features = features.unsqueeze(1)
        return self.mlp(features)
