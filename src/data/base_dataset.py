"""
Base Dataset Interface for HybridEditDif
=========================================
Semua dataset class harus inherit dari BaseEditDataset ini
agar kompatibel dengan train.py secara universal.

Output format __getitem__ yang wajib:
    {
        "source":    Tensor [3, H, W]  — gambar asli ([-1,1])
        "reference": Tensor [3, 224, 224] — reference untuk CLIP ([CLIP norm])
        "mask":      Tensor [1, H, W]  — binary mask, 1=edit region
        "source_masked": Tensor [3, H, W] — (1-mask)*source
        "text":      str               — text prompt
        "text_tokens": Tensor [77]     — tokenized text
    }
"""

import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms as T
import open_clip

logger = logging.getLogger(__name__)


# ── Shared transforms ─────────────────────────────────────────────────────────

def make_source_transform(image_size: int = 512):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # [-1,1] for SD
    ])


def make_clip_transform(clip_size: int = 224):
    return T.Compose([
        T.Resize((clip_size, clip_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def make_ref_augment(clip_size: int = 224):
    """Reference image augmentation (paper Section 3.2)."""
    return T.Compose([
        T.RandomRotation(degrees=30),
        T.RandomResizedCrop(clip_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def bbox_to_mask(
    bbox_norm: Tuple[float, float, float, float],
    W: int, H: int,
    target_size: int = 512,
    min_area: float = 0.02,
    max_area: float = 0.5,
) -> Optional[Image.Image]:
    """
    Convert normalized bbox (x1,y1,x2,y2) to PIL mask image.
    Returns None if bbox area is too small or too large.
    """
    x1 = int(max(0, bbox_norm[0]) * W)
    y1 = int(max(0, bbox_norm[1]) * H)
    x2 = int(min(1, bbox_norm[2]) * W)
    y2 = int(min(1, bbox_norm[3]) * H)

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    area = (bw * bh) / (W * H)
    if area < min_area or area > max_area:
        return None

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
    return mask


def center_crop_mask(image_pil: Image.Image, ratio: float = 0.35) -> Image.Image:
    """Fallback mask: center box."""
    W, H = image_pil.size
    mask = Image.new("L", (W, H), 0)
    margin_x = int(W * (0.5 - ratio / 2))
    margin_y = int(H * (0.5 - ratio / 2))
    ImageDraw.Draw(mask).rectangle(
        [margin_x, margin_y, W - margin_x, H - margin_y], fill=255
    )
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Base
# ══════════════════════════════════════════════════════════════════════════════

class BaseEditDataset(Dataset, ABC):
    """
    Abstract base class for all HybridEditDif training datasets.

    Subclasses must implement:
        _load_samples()  → list of sample dicts
        _get_sample(idx) → (image_pil, ref_pil, mask_pil, text_str)
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 512,
        clip_image_size: int = 224,
        max_samples: Optional[int] = None,
        augment_reference: bool = True,
        seed: int = 42,
    ):
        self.split            = split
        self.image_size       = image_size
        self.clip_image_size  = clip_image_size
        self.augment_reference = augment_reference

        random.seed(seed)
        np.random.seed(seed)

        self.source_transform = make_source_transform(image_size)
        self.clip_transform   = make_clip_transform(clip_image_size)
        self.ref_aug          = make_ref_augment(clip_image_size)
        self.tokenizer        = open_clip.get_tokenizer('ViT-g-14')

        # Load samples (implemented by subclass)
        self.samples = self._load_samples()
        if max_samples and max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        logger.info(f"Dataset: {len(self.samples)} samples ({split})")

    @abstractmethod
    def _load_samples(self) -> List[Dict]:
        """Return list of sample metadata dicts."""
        ...

    @abstractmethod
    def _get_raw(self, idx: int) -> Tuple[
        Image.Image,            # source image
        Optional[Image.Image],  # reference image (or None → use crop from source)
        Optional[Image.Image],  # mask (or None → generate from bbox/random)
        str,                    # text prompt
    ]:
        """Return raw PIL inputs for sample at idx."""
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        try:
            source_pil, ref_pil, mask_pil, text = self._get_raw(idx)
        except Exception as e:
            logger.debug(f"Sample {idx} failed: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self.samples))

        W, H = source_pil.size

        # Fallback mask
        if mask_pil is None:
            mask_pil = center_crop_mask(source_pil)

        # Fallback reference → crop the masked region from source
        if ref_pil is None:
            mask_np = np.array(mask_pil) > 127
            ys, xs  = np.where(mask_np)
            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                ref_pil = source_pil.crop((x1, y1, x2, y2))
            else:
                ref_pil = source_pil

        # ── Resize to target size ─────────────────────────────────────────────
        source_pil = source_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
        mask_pil   = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        # ── Transforms ───────────────────────────────────────────────────────
        source_t  = self.source_transform(source_pil)                      # [-1,1]
        mask_np   = (np.array(mask_pil) > 127).astype(np.float32)
        mask_t    = torch.from_numpy(mask_np).unsqueeze(0)                 # [1,H,W]

        # Masked source: (1-m)*X_s
        masked_t  = source_t * (1.0 - mask_t)

        # Reference: augment during training, plain resize otherwise
        if self.augment_reference and self.split == "train":
            ref_t = self.ref_aug(ref_pil)
        else:
            ref_t = self.clip_transform(ref_pil)

        # Text tokens
        text_tokens = self.tokenizer([text])[0]   # [77]

        return {
            "source":        source_t,    # [3, H, W]
            "source_masked": masked_t,    # [3, H, W]
            "reference":     ref_t,       # [3, 224, 224]
            "mask":          mask_t,      # [1, H, W]
            "text":          text,
            "text_tokens":   text_tokens, # [77]
        }


# ══════════════════════════════════════════════════════════════════════════════
# Universal collate_fn  (same as before)
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "source":        torch.stack([b["source"]        for b in batch]),
        "source_masked": torch.stack([b["source_masked"] for b in batch]),
        "reference":     torch.stack([b["reference"]     for b in batch]),
        "mask":          torch.stack([b["mask"]          for b in batch]),
        "text":          [b["text"]         for b in batch],
        "text_tokens":   torch.stack([b["text_tokens"]   for b in batch]),
    }
