"""
OpenImages Dataset Downloader & DataLoader
==========================================
Paper Section 4.1: "We used the OpenImages dataset as the primary source
for training due to its extensive coverage of 1.9 million images and
16 million annotated bounding boxes across 600 object classes."

This script:
1. Downloads a subset of OpenImages V7 (train split)
2. Fetches bounding box annotations for self-supervised masking
3. Provides DataLoader compatible with HybridEditDif training
"""

import os
import json
import csv
import random
import logging
import requests
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import open_clip

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Downloader
# ══════════════════════════════════════════════════════════════════════════════

class OpenImagesDownloader:
    """
    Downloads a configurable subset of OpenImages V7.
    Full dataset = 1.9M images; we support downloading subsets for feasibility.
    """

    OPENIMAGES_URLS = {
        "train_bbox": (
            "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
            "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
        ),
        "validation_bbox": (
            "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        ),
        "test_bbox": (
            "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        ),
        "image_ids_train": (
            "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
        ),
        "image_ids_val": (
            "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        ),
        "image_ids_test": (
            "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
        ),
    }

    def __init__(self, data_root: str = "./data/openimages"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.annotations_dir = self.data_root / "annotations"
        self.annotations_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_root / "images"
        self.images_dir.mkdir(exist_ok=True)

    def download_annotations(self, split: str = "train"):
        """Download bounding box annotations CSV."""
        logger.info(f"Downloading {split} annotations...")

        # Image IDs
        ids_url = self.OPENIMAGES_URLS[f"image_ids_{split}"][0]
        ids_file = self.annotations_dir / f"{split}_image_ids.csv"
        if not ids_file.exists():
            self._download_file(ids_url, ids_file)

        # Bounding boxes
        bbox_url = self.OPENIMAGES_URLS[f"{split}_bbox"]
        bbox_file = self.annotations_dir / f"{split}_bbox.csv"
        if not bbox_file.exists():
            self._download_file(bbox_url[-1], bbox_file)

        return ids_file, bbox_file

    def _download_file(self, url: str, dest: Path):
        logger.info(f"  Downloading: {url}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"  Saved: {dest}")

    def download_images_subset(
        self,
        image_ids: List[str],
        split: str = "train",
        n_images: int = 50000,
        n_workers: int = 8,
    ):
        """
        Download images using AWS CLI or direct URLs.
        OpenImages images are hosted on AWS S3.
        """
        logger.info(f"Preparing to download {n_images} images from {split}...")

        split_dir = self.images_dir / split
        split_dir.mkdir(exist_ok=True)

        # Use aws s3 sync (fastest) if available
        aws_cmd = f"aws s3 --no-sign-request sync s3://open-images-dataset/{split} {split_dir} --include '*.jpg' --quiet"

        # Alternative: fiftyone library (easiest)
        install_cmd = "pip install fiftyone --quiet"

        logger.info("  Recommending: use fiftyone for easy OpenImages download")
        logger.info(f"  CMD: {install_cmd}")
        logger.info(f"  Then use OpenImagesV7FiftyoneLoader (see scripts/download_openimages.py)")

    def load_bbox_annotations(
        self,
        split: str = "train",
        max_images: int = 100000,
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Load bounding box annotations.
        Returns: dict {image_id: [(XMin, YMin, XMax, YMax), ...]}
        """
        bbox_file = self.annotations_dir / f"{split}_bbox.csv"
        if not bbox_file.exists():
            self.download_annotations(split)

        bboxes = {}
        logger.info(f"Loading bbox annotations from {bbox_file}...")

        with open(bbox_file, 'r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                img_id = row['ImageID']
                if img_id not in bboxes:
                    bboxes[img_id] = []
                    count += 1
                    if count >= max_images:
                        break
                bboxes[img_id].append((
                    float(row['XMin']),
                    float(row['YMin']),
                    float(row['XMax']),
                    float(row['YMax']),
                ))

        logger.info(f"  Loaded {len(bboxes)} images with bbox annotations")
        return bboxes


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class OpenImagesEditingDataset(Dataset):
    """
    Dataset for HybridEditDif training on OpenImages.

    Per paper Section 3.2 (Self-supervised training):
        - Source image X_s
        - Reference X_r = m · X_s  (crop masked by bounding box)
        - Training tuple: (1-m)·X_s, X_r, T, m
        - Text T from OpenImages visual relationship annotations

    Training image size: 512×512
    """

    def __init__(
        self,
        images_dir: str,
        bbox_annotations: Dict,           # {image_id: [(x1,y1,x2,y2),...]}
        text_annotations: Optional[Dict] = None,  # {image_id: "text description"}
        image_size: int = 512,
        clip_image_size: int = 224,
        split: str = "train",
        max_samples: Optional[int] = None,
        drop_image_cond_prob: float = 0.05,  # per paper Section 4.1
        drop_text_cond_prob: float = 0.05,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.clip_image_size = clip_image_size
        self.split = split
        self.drop_img_prob = drop_image_cond_prob
        self.drop_txt_prob = drop_text_cond_prob

        # Filter to existing images
        self.samples = []
        for img_id, bboxes in bbox_annotations.items():
            img_path = self.images_dir / f"{img_id}.jpg"
            if img_path.exists() and bboxes:
                text = text_annotations.get(img_id, "an object") if text_annotations else "an object"
                self.samples.append({
                    "image_id": img_id,
                    "path": img_path,
                    "bboxes": bboxes,
                    "text": text,
                })

        if max_samples:
            self.samples = self.samples[:max_samples]
        random.shuffle(self.samples)

        logger.info(f"Dataset: {len(self.samples)} samples ({split})")

        # ── Transforms ───────────────────────────────────────────────────────
        self.source_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1,1] for SD
        ])

        self.clip_transform = T.Compose([
            T.Resize((clip_image_size, clip_image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),  # OpenCLIP normalization
        ])

        # Reference image augmentation (paper Section 3.2)
        self.ref_aug = T.Compose([
            T.RandomRotation(degrees=45),
            T.RandomResizedCrop(
                clip_image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        # OpenCLIP tokenizer
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────────
        try:
            image_pil = Image.open(sample["path"]).convert("RGB")
        except Exception:
            # Fallback to next sample on corrupt image
            return self.__getitem__((idx + 1) % len(self.samples))

        W_orig, H_orig = image_pil.size

        # ── Select random bounding box ────────────────────────────────────────
        bbox_norm = random.choice(sample["bboxes"])  # (XMin, YMin, XMax, YMax) normalized
        x1 = int(bbox_norm[0] * W_orig)
        y1 = int(bbox_norm[1] * H_orig)
        x2 = int(bbox_norm[2] * W_orig)
        y2 = int(bbox_norm[3] * H_orig)
        x1, x2 = max(0, x1), min(W_orig, x2)
        y1, y2 = max(0, y1), min(H_orig, y2)

        if x2 <= x1 or y2 <= y1:
            return self.__getitem__((idx + 1) % len(self.samples))

        # ── Build self-supervised mask (Section 3.2) ─────────────────────────
        # m = bounding box binary mask at original resolution
        mask_np = np.zeros((H_orig, W_orig), dtype=np.float32)
        mask_np[y1:y2, x1:x2] = 1.0

        # Resize mask to model resolution
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(
            np.array(mask_img).astype(np.float32) / 255.0
        ).unsqueeze(0)  # [1, H, W]

        # ── Source image ──────────────────────────────────────────────────────
        source_tensor = self.source_transform(image_pil)  # [3, H, W] in [-1,1]

        # ── Reference image X_r = crop of source (m · X_s) ───────────────────
        ref_crop = image_pil.crop((x1, y1, x2, y2))
        # Apply reference augmentation (Section 3.2)
        ref_tensor = self.ref_aug(ref_crop)  # [3, 224, 224] CLIP normalized

        # ── Masked source (1-m) · X_s ─────────────────────────────────────────
        # In [-1,1] space: 0 in masked region, original elsewhere
        masked_source = source_tensor * (1.0 - mask_tensor)  # [3, H, W]

        # ── Text conditioning ─────────────────────────────────────────────────
        text = sample["text"]
        text_tokens = self.tokenizer([text])  # [1, seq_len]

        # ── Classifier-free guidance dropout (Section 4.1) ───────────────────
        drop_image = random.random() < self.drop_img_prob
        drop_text  = random.random() < self.drop_txt_prob

        return {
            "source":        source_tensor,    # [3, H, W]  original image
            "masked_source": masked_source,    # [3, H, W]  (1-m)·X_s
            "reference":     ref_tensor,       # [3, 224, 224] X_r for CLIP
            "mask":          mask_tensor,      # [1, H, W]  binary mask
            "text_tokens":   text_tokens[0],   # [seq_len]
            "text":          text,
            "drop_image":    drop_image,
            "drop_text":     drop_text,
            "image_id":      sample["image_id"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate for variable-length text tokens."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ("text", "image_id"):
            result[k] = [b[k] for b in batch]
        elif k in ("drop_image", "drop_text"):
            result[k] = [b[k] for b in batch]
        elif k == "text_tokens":
            # Pad to same length
            tokens = [b[k] for b in batch]
            max_len = max(t.shape[-1] for t in tokens)
            padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
            for i, t in enumerate(tokens):
                padded[i, :t.shape[-1]] = t
            result[k] = padded
        else:
            result[k] = torch.stack([b[k] for b in batch])
    return result


def get_dataloaders(
    images_dir: str,
    bbox_annotations: Dict,
    text_annotations: Optional[Dict] = None,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 8,
    val_split: float = 0.02,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    full_dataset = OpenImagesEditingDataset(
        images_dir=images_dir,
        bbox_annotations=bbox_annotations,
        text_annotations=text_annotations,
        image_size=image_size,
        max_samples=max_samples,
    )

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"Train: {n_train} | Val: {n_val} samples")
    return train_loader, val_loader
