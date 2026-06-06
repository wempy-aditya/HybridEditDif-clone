"""
COCO Dataset Loader for HybridEditDif Training
================================================
Uses COCO 2017 train/val splits with instance segmentation masks
as the inpainting region (instead of bounding boxes).

Download:
    python scripts/download_coco.py --data_root ./data/coco --split train

Structure expected:
    data/coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    └── images/
        ├── train2017/
        └── val2017/
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base_dataset import BaseEditDataset, bbox_to_mask, center_crop_mask

logger = logging.getLogger(__name__)

# Generic COCO captions — used when panoptic/caption annotations not available
COCO_CATEGORY_TEMPLATES = [
    "a photo of a {category}",
    "replace the {category} with a similar one",
    "edit the {category} in the scene",
    "a {category} in the image",
]


class COCOEditDataset(BaseEditDataset):
    """
    COCO 2017 dataset for inpainting training.

    Masking strategy:
      - Primary:  segmentation mask of a random instance (best quality)
      - Fallback: bounding box of a random instance
      - Last:     center crop

    Reference:
      - The masked region crop is used as reference (self-supervised).
      - Augmented with ref_aug during training.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        clip_image_size: int = 224,
        max_samples: Optional[int] = None,
        min_bbox_area: float = 0.02,   # min fraction of image area
        max_bbox_area: float = 0.60,   # max fraction of image area
        augment_reference: bool = True,
        seed: int = 42,
    ):
        self.data_root      = Path(data_root)
        self.min_bbox_area  = min_bbox_area
        self.max_bbox_area  = max_bbox_area

        # Determine annotation file and image folder
        ann_split = "train2017" if split == "train" else "val2017"
        self.images_dir = self.data_root / "images" / ann_split
        self.ann_file   = self.data_root / "annotations" / f"instances_{ann_split}.json"

        super().__init__(
            split=split,
            image_size=image_size,
            clip_image_size=clip_image_size,
            max_samples=max_samples,
            augment_reference=augment_reference,
            seed=seed,
        )

    def _load_samples(self) -> List[Dict]:
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotation not found: {self.ann_file}\n"
                f"  Run: python scripts/download_coco.py --data_root {self.data_root}"
            )

        logger.info(f"Loading COCO annotations dari {self.ann_file}...")
        with open(self.ann_file) as f:
            coco_data = json.load(f)

        # Build category id → name map
        cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

        # Build image id → image info map
        img_map = {img["id"]: img for img in coco_data["images"]}

        # Group annotations by image
        ann_by_img: Dict[int, List] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)

        samples = []
        for img_id, anns in ann_by_img.items():
            img_info = img_map.get(img_id)
            if img_info is None:
                continue

            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            W, H = img_info["width"], img_info["height"]

            # Filter annotations by area
            valid_anns = []
            for ann in anns:
                area_frac = ann.get("area", 0) / (W * H)
                if self.min_bbox_area <= area_frac <= self.max_bbox_area:
                    valid_anns.append(ann)

            if not valid_anns:
                continue

            category_name = cat_map.get(valid_anns[0]["category_id"], "object")
            samples.append({
                "img_id":    img_id,
                "path":      img_path,
                "anns":      valid_anns,
                "W": W, "H": H,
                "category":  category_name,
            })

        logger.info(f"COCO ({self.split}): {len(samples)} images dengan valid instances")
        return samples

    def _get_raw(self, idx: int):
        sample = self.samples[idx]
        image_pil = Image.open(sample["path"]).convert("RGB")
        W, H = image_pil.size

        # Pick random annotation
        ann = random.choice(sample["anns"])
        category = sample["category"]

        # Text prompt
        template = random.choice(COCO_CATEGORY_TEMPLATES)
        text = template.format(category=category)

        # ── Build mask ─────────────────────────────────────────────────────
        mask_pil = None

        # Try segmentation mask first (polygon)
        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and len(seg) > 0:
            try:
                from PIL import ImageDraw
                mask_pil = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(mask_pil)
                for poly in seg:
                    if len(poly) >= 6:
                        xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                        draw.polygon(xy, fill=255)
            except Exception:
                mask_pil = None

        # Fallback: bounding box
        if mask_pil is None:
            bbox = ann.get("bbox")   # [x, y, w, h] format in COCO
            if bbox:
                x, y, bw, bh = bbox
                bbox_norm = (x/W, y/H, (x+bw)/W, (y+bh)/H)
                mask_pil = bbox_to_mask(bbox_norm, W, H,
                                        min_area=self.min_bbox_area,
                                        max_area=self.max_bbox_area)

        # Reference = crop of masked region
        ref_pil = None
        if mask_pil is not None:
            mask_np = np.array(mask_pil) > 127
            ys, xs  = np.where(mask_np)
            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                ref_pil = image_pil.crop((x1, y1, max(x1+1, x2), max(y1+1, y2)))

        return image_pil, ref_pil, mask_pil, text


class COCOValDataset(COCOEditDataset):
    """Alias for COCO validation split."""
    def __init__(self, data_root: str, **kwargs):
        super().__init__(data_root=data_root, split="val", **kwargs)
