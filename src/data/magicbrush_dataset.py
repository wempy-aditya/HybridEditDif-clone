"""
MagicBrush Dataset Loader for HybridEditDif Training
======================================================
MagicBrush is an instruction-based editing dataset with
real human edits + masks, making it excellent for training.

Paper: Zhang et al. "MagicBrush: A Manually Annotated Dataset
for Instruction-Guided Image Editing" (NeurIPS 2023)

Download:
    python scripts/download_magicbrush.py --data_root ./data/magicbrush

    Or via HuggingFace:
    from huggingface_hub import snapshot_download
    snapshot_download("osunlp/MagicBrush", repo_type="dataset",
                      local_dir="data/magicbrush")

Structure expected:
    data/magicbrush/
    ├── train_dataset.json    ← annotation file
    ├── test_dataset.json
    ├── images/
    │   ├── {id}_input.png
    │   ├── {id}_output.png
    │   └── {id}_mask.png
    └── (or structured as HuggingFace parquet)
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base_dataset import BaseEditDataset, center_crop_mask

logger = logging.getLogger(__name__)


class MagicBrushEditDataset(BaseEditDataset):
    """
    MagicBrush dataset for inpainting training.

    Key differences from OpenImages/COCO:
    - Has REAL human instruction texts (not templates)
    - Has ground-truth edited output images (used as reference)
    - Has precise edit masks

    This is the highest-quality training data for editing tasks.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        clip_image_size: int = 224,
        max_samples: Optional[int] = None,
        augment_reference: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self._split    = split  # Store before super() calls _load_samples

        super().__init__(
            split=split,
            image_size=image_size,
            clip_image_size=clip_image_size,
            max_samples=max_samples,
            augment_reference=augment_reference,
            seed=seed,
        )

    def _find_annotation_file(self) -> Optional[Path]:
        """Try multiple common annotation file naming conventions."""
        candidates = [
            self.data_root / f"{self._split}_dataset.json",
            self.data_root / f"{self._split}.json",
            self.data_root / "annotations" / f"{self._split}.json",
            # HuggingFace parquet format (via json export)
            self.data_root / f"data/{self._split}-*.json",
        ]
        for c in candidates:
            if "*" in str(c):
                matches = list(self.data_root.glob(str(c.relative_to(self.data_root))))
                if matches:
                    return matches[0]
            elif c.exists():
                return c
        return None

    def _find_image(self, img_name: str) -> Optional[Path]:
        """Search for image across common folder structures."""
        candidates = [
            self.data_root / "images" / img_name,
            self.data_root / img_name,
            self.data_root / "data" / img_name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_samples(self) -> List[Dict]:
        ann_file = self._find_annotation_file()
        if ann_file is None:
            raise FileNotFoundError(
                f"MagicBrush annotation not found in {self.data_root}\n"
                f"  Run: python scripts/download_magicbrush.py "
                f"--data_root {self.data_root}"
            )

        logger.info(f"Loading MagicBrush annotations dari {ann_file}...")
        with open(ann_file) as f:
            data = json.load(f)

        # Support both list and dict formats
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data

        samples = []
        for item in items:
            # Flexible key mapping
            input_file  = item.get("input")  or item.get("source") or item.get("image")
            output_file = item.get("output") or item.get("target") or item.get("edited_image")
            mask_file   = item.get("mask")   or item.get("edit_mask")
            instruction = (item.get("instruction") or item.get("text")
                           or item.get("caption") or "edit the image")

            if not input_file:
                continue

            input_path  = self._find_image(input_file)
            output_path = self._find_image(output_file) if output_file else None
            mask_path   = self._find_image(mask_file)   if mask_file   else None

            if input_path is None:
                continue

            samples.append({
                "input_path":  input_path,
                "output_path": output_path,
                "mask_path":   mask_path,
                "instruction": instruction,
            })

        logger.info(f"MagicBrush ({self.split}): {len(samples)} samples loaded")
        return samples

    def _get_raw(self, idx: int):
        sample    = self.samples[idx]
        src_pil   = Image.open(sample["input_path"]).convert("RGB")
        text      = sample["instruction"]

        # Reference = output image (real edited result)
        ref_pil = None
        if sample["output_path"] and sample["output_path"].exists():
            ref_pil = Image.open(sample["output_path"]).convert("RGB")

        # Mask
        mask_pil = None
        if sample["mask_path"] and sample["mask_path"].exists():
            mask_pil = Image.open(sample["mask_path"]).convert("L")
        elif ref_pil is not None:
            # Infer mask from pixel diff between input and output
            src_np = np.array(src_pil.resize(ref_pil.size))
            ref_np = np.array(ref_pil)
            diff   = np.abs(src_np.astype(int) - ref_np.astype(int)).max(axis=-1)
            mask_np = (diff > 10).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np, mode="L")

        return src_pil, ref_pil, mask_pil, text
