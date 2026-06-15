"""
Dataset Factory — Universal Dataset Builder for HybridEditDif
=============================================================
Digunakan oleh train.py untuk membuat dataset berdasarkan config.

Config format:
    dataset:
      type: "openimages"       # openimages | coco | magicbrush | mixed
      data_root: "./data"
      image_size: 512
      max_samples: null        # null = all
      val_ratio: 0.02

      # Per-dataset options:
      openimages:
        split_dir: "openimages"
        annotations_file: "annotations.json"

      coco:
        split_dir: "coco"
        min_bbox_area: 0.02
        max_bbox_area: 0.60

      magicbrush:
        split_dir: "magicbrush"

      mixed:
        weights:               # sampling weight per dataset
          openimages: 0.6
          coco: 0.3
          magicbrush: 0.1
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from .base_dataset import collate_fn as base_collate_fn
from .openimages_dataset import OpenImagesEditingDataset
from .openimages_dataset import collate_fn as openimages_collate_fn
from .coco_dataset import COCOEditDataset
from .magicbrush_dataset import MagicBrushEditDataset

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

# openimages tidak masuk registry karena punya signature berbeda
DATASET_REGISTRY = {
    "coco":       COCOEditDataset,
    "magicbrush": MagicBrushEditDataset,
}


def build_openimages_dataset(
    data_root: str,
    split: str = "train",
    image_size: int = 512,
    max_samples: Optional[int] = None,
    annotations_file: str = None,
    **kwargs,
) -> Dataset:
    """
    Builder khusus OpenImages — load annotations JSON dulu,
    lalu pass ke OpenImagesEditingDataset dengan signature yang benar.
    """
    import json
    root = Path(data_root)

    # Cari bbox annotations
    ann_candidates = [
        root / "annotations" / f"{split}_bbox_annotations.json",
        root / "annotations" / "train_bbox_annotations.json",  # fallback
    ]
    if annotations_file:
        ann_candidates.insert(0, root / annotations_file)

    bbox_file = next((p for p in ann_candidates if p.exists()), None)
    if bbox_file is None:
        raise FileNotFoundError(
            f"OpenImages bbox annotations tidak ditemukan di {root}/annotations/\n"
            f"  Jalankan dulu: python scripts/download_openimages.py "
            f"--data_root {root}"
        )

    logger.info(f"Loading bbox annotations: {bbox_file.name}")
    with open(bbox_file) as f:
        raw = json.load(f)
    bbox_annotations = {
        img_id: [tuple(b) for b in bboxes]
        for img_id, bboxes in raw.items()
    }
    logger.info(f"  {len(bbox_annotations):,} images dengan bbox annotations")

    # Cari text annotations (optional)
    text_file = root / "annotations" / "text_annotations.json"
    text_annotations = None
    if text_file.exists():
        with open(text_file) as f:
            text_annotations = json.load(f)
        logger.info(f"  Text annotations: {len(text_annotations):,} entries")

    images_dir = root / "images" / split
    if not images_dir.exists():
        # Fallback: coba tanpa split subfolder
        images_dir = root / "images"

    logger.info(f"Building OpenImages dataset | split={split} | root={images_dir}")
    return OpenImagesEditingDataset(
        images_dir=str(images_dir),
        bbox_annotations=bbox_annotations,
        text_annotations=text_annotations,
        image_size=image_size,
        split=split,
        max_samples=max_samples,
    )


def build_single_dataset(
    dataset_type: str,
    data_root: str,
    split: str = "train",
    image_size: int = 512,
    max_samples: Optional[int] = None,
    extra_kwargs: Optional[Dict] = None,
) -> Dataset:
    """Build a single dataset by type name."""
    # OpenImages punya signature berbeda — dispatch ke builder khusus
    if dataset_type == "openimages":
        return build_openimages_dataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            max_samples=max_samples,
            **(extra_kwargs or {}),
        )

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Available: ['openimages'] + {list(DATASET_REGISTRY.keys())}"
        )

    cls = DATASET_REGISTRY[dataset_type]
    kwargs = dict(
        data_root=data_root,
        split=split,
        image_size=image_size,
        max_samples=max_samples,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.info(f"Building {dataset_type} dataset | split={split} | root={data_root}")
    return cls(**kwargs)


def build_mixed_dataset(
    cfg,                       # OmegaConf DictConfig
    split: str = "train",
) -> Tuple[Dataset, Optional[WeightedRandomSampler]]:
    """
    Build a weighted mix of multiple datasets.

    Returns (ConcatDataset, WeightedRandomSampler).
    The sampler ensures that each dataset is sampled according to its weight.
    """
    mixed_cfg = cfg.dataset.get("mixed", {})
    weights   = mixed_cfg.get("weights", {})
    data_root = cfg.dataset.data_root
    image_size = cfg.dataset.get("image_size", 512)
    max_samples = cfg.dataset.get("max_samples", None)

    datasets   = []
    ds_weights = []

    for ds_type, weight in weights.items():
        ds_root = str(Path(data_root) / cfg.dataset.get(ds_type, {}).get("split_dir", ds_type))
        extra   = dict(cfg.dataset.get(ds_type, {}))
        extra.pop("split_dir", None)

        try:
            ds = build_single_dataset(
                dataset_type=ds_type,
                data_root=ds_root,
                split=split,
                image_size=image_size,
                max_samples=max_samples,
                extra_kwargs=extra if extra else None,
            )
            datasets.append(ds)
            ds_weights.append((weight, len(ds)))
            logger.info(f"  ✓ {ds_type}: {len(ds)} samples (weight={weight})")
        except Exception as e:
            logger.warning(f"  ✗ {ds_type} FAILED: {e}. Skipping.")

    if not datasets:
        raise RuntimeError("No datasets could be loaded for mixed training.")

    combined = ConcatDataset(datasets)

    # Build per-sample weights for WeightedRandomSampler
    sample_weights = []
    total_norm = sum(w for w, _ in ds_weights)
    for (w, n) in ds_weights:
        per_sample = (w / total_norm) / n
        sample_weights.extend([per_sample] * n)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined),
        replacement=True,
    )

    logger.info(f"Mixed dataset: {len(combined)} total samples from {len(datasets)} sources")
    return combined, sampler


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry: build_dataloaders_from_config
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders_from_config(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train + val DataLoaders from OmegaConf config.

    Supports:
        dataset.type: openimages | coco | magicbrush | mixed
    """
    ds_type   = cfg.dataset.get("type", "openimages")
    data_root = cfg.dataset.data_root
    image_size = cfg.dataset.get("image_size", 512)
    max_samples = cfg.dataset.get("max_samples", None)
    val_ratio   = cfg.dataset.get("val_ratio", 0.02)
    batch_size  = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 4)

    sampler = None

    if ds_type == "mixed":
        train_ds, sampler = build_mixed_dataset(cfg, split="train")
        val_max = max(10, int(len(train_ds) * val_ratio))
        try:
            val_ds, _ = build_mixed_dataset(cfg, split="val")
        except Exception:
            # Fallback: use subset of train dataset
            indices = list(range(min(val_max, len(train_ds))))
            val_ds  = torch.utils.data.Subset(train_ds, indices)

    else:
        # Single dataset type
        extra_key = ds_type
        extra_cfg = dict(cfg.dataset.get(extra_key, {}))
        ds_root   = str(Path(data_root) / extra_cfg.pop("split_dir", ds_type))

        # Build train dataset
        train_ds = build_single_dataset(
            dataset_type=ds_type,
            data_root=ds_root,
            split="train",
            image_size=image_size,
            max_samples=max_samples,
            extra_kwargs=extra_cfg if extra_cfg else None,
        )

        # Build val dataset (use val split or subset)
        val_max = max(10, int(len(train_ds) * val_ratio))
        try:
            val_ds = build_single_dataset(
                dataset_type=ds_type,
                data_root=ds_root,
                split="val",
                image_size=image_size,
                max_samples=val_max,
                extra_kwargs=extra_cfg if extra_cfg else None,
            )
        except Exception:
            # Fallback: subset of train
            indices = random.sample(range(len(train_ds)), min(val_max, len(train_ds)))
            val_ds  = torch.utils.data.Subset(train_ds, indices)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} samples")

    # Pilih collate_fn: OpenImages punya format batch sendiri (masked_source, drop_image, dll)
    collate = openimages_collate_fn if ds_type == "openimages" else base_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=min(2, num_workers),
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader
