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

    Auto-detects multiple folder structures:
      A) Standard COCO:
           annotations/instances_train2017.json + images/train2017/
      B) FiftyOne export:
           labels.json + data/
      C) Flat:
           labels.json (or *.json) + images/ (atau data/)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        clip_image_size: int = 224,
        max_samples: Optional[int] = None,
        min_bbox_area: float = 0.02,
        max_bbox_area: float = 0.60,
        augment_reference: bool = True,
        seed: int = 42,
    ):
        self.data_root     = Path(data_root)
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area

        # Auto-detect annotation file + images dir
        self.ann_file, self.images_dir = self._detect_paths(split)

        super().__init__(
            split=split,
            image_size=image_size,
            clip_image_size=clip_image_size,
            max_samples=max_samples,
            augment_reference=augment_reference,
            seed=seed,
        )

    def _detect_paths(self, split: str):
        """
        Try multiple path conventions and return (ann_file, images_dir).
        Priority:
          1. Standard COCO (annotations/instances_train2017.json)
          2. FiftyOne export (labels.json + data/)
          3. Any .json in root + images/ or data/
        """
        root = self.data_root
        ann_split = "train2017" if split == "train" else "val2017"

        # ── 1. Standard COCO format ──────────────────────────────────────────
        std_ann = root / "annotations" / f"instances_{ann_split}.json"
        std_img = root / "images" / ann_split
        if std_ann.exists() and std_img.exists():
            logger.info(f"COCO format: standard ({std_ann.name})")
            return std_ann, std_img

        # Also try without split suffix in images dir
        std_img2 = root / "images"
        if std_ann.exists() and std_img2.exists():
            logger.info(f"COCO format: standard ann + flat images/")
            return std_ann, std_img2

        # ── 2. FiftyOne export format (labels.json + data/) ──────────────────
        fo_ann = root / "labels.json"
        fo_img = root / "data"
        if fo_ann.exists() and fo_img.exists() and any(fo_img.iterdir()):
            logger.info(f"COCO format: FiftyOne export (labels.json + data/)")
            return fo_ann, fo_img

        # ── 3. Any JSON file + images/data folder ────────────────────────────
        img_candidates = [root / "data", root / "images", root / f"images/{ann_split}"]
        json_candidates = (
            list(root.glob("*.json")) +
            list((root / "annotations").glob("*.json") if (root / "annotations").exists() else [])
        )

        img_dir = next((d for d in img_candidates if d.exists() and d.is_dir()), None)
        ann_file = next((f for f in json_candidates if f.exists()), None)

        if ann_file and img_dir:
            logger.info(f"COCO format: auto-detected ({ann_file.name} + {img_dir.name}/)")
            return ann_file, img_dir

        # ── Not found — raise informative error ──────────────────────────────
        raise FileNotFoundError(
            f"\nCOCO data tidak ditemukan di: {root}\n"
            f"  Struktur yang ada:\n"
            f"    {list(root.iterdir()) if root.exists() else '(folder tidak ada)'}\n\n"
            f"  Solusi:\n"
            f"    python scripts/download_coco.py --data_root {root} --n_samples 5000\n\n"
            f"  Atau jika FiftyOne sudah download:\n"
            f"    Cek apakah ada labels.json dan folder data/ di {root}"
        )

    def _load_samples(self) -> List[Dict]:
        logger.info(f"Loading COCO annotations dari {self.ann_file.name}...")
        with open(self.ann_file) as f:
            coco_data = json.load(f)

        # ── Validate JSON structure ───────────────────────────────────────────
        if "images" not in coco_data or "annotations" not in coco_data:
            raise ValueError(
                f"File {self.ann_file.name} bukan format COCO yang valid.\n"
                f"  Butuh keys: 'images', 'annotations', 'categories'\n"
                f"  Keys yang ada: {list(coco_data.keys())}"
            )

        # Build lookup maps
        cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        img_map = {img["id"]: img for img in coco_data["images"]}

        # Group annotations by image
        ann_by_img: Dict[int, List] = {}
        for ann in coco_data["annotations"]:
            iid = ann["image_id"]
            ann_by_img.setdefault(iid, []).append(ann)

        samples = []
        missing = 0
        for img_id, anns in ann_by_img.items():
            img_info = img_map.get(img_id)
            if img_info is None:
                continue

            # FiftyOne stores just the filename, standard COCO may store full path
            file_name = Path(img_info["file_name"]).name
            img_path  = self.images_dir / file_name

            if not img_path.exists():
                missing += 1
                continue

            W = img_info.get("width",  0)
            H = img_info.get("height", 0)

            # If dimensions not in annotation, read from file
            if W == 0 or H == 0:
                try:
                    from PIL import Image as _Image
                    W, H = _Image.open(img_path).size
                except Exception:
                    continue

            # Filter by area
            valid_anns = [
                ann for ann in anns
                if self.min_bbox_area <= ann.get("area", 0) / (W * H) <= self.max_bbox_area
            ]
            if not valid_anns:
                continue

            category_name = cat_map.get(valid_anns[0]["category_id"], "object")
            samples.append({
                "img_id":   img_id,
                "path":     img_path,
                "anns":     valid_anns,
                "W": W, "H": H,
                "category": category_name,
            })

        if missing > 0:
            logger.warning(f"COCO: {missing} images tidak ditemukan di {self.images_dir}")
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
