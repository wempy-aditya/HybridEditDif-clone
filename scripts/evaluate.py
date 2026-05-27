"""
HybridEditDif — Benchmark Evaluation Script
=============================================
Reproduces Table 1 from paper on:
  1. COCOEE dataset
  2. MagicBrush Test dataset
  3. Emu Edit Test dataset

Metrics computed: FID ↓, QS ↑, CLIP Score ↑

Usage:
    python evaluate.py \
        --checkpoint checkpoints/final_model/hybrid_edit_dif_weights.pt \
        --dataset cocoee \
        --data_root ./data/COCOEE \
        --output_dir experiments/ \
        --steps 30 \
        --max_samples 100
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.hybrid_edit_dif import HybridEditDif
from src.models.inference import HybridEditDifInferencePipeline
from src.utils.metrics import HybridEditDifEvaluator

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loaders for Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

class COCOEELoader:
    """
    COCOEE dataset loader (Paint-by-Example format).

    Mendukung naming convention:
      images/{id}_GT.png + masks/{id}_mask.png + references/{id}_ref.png
      → matching by numeric ID prefix ("000000000283")

    Juga mendukung:
      test_bench/GT_xxx/ + Ref_xxx/ + Mask_xxx/  (folder-based)
      images/{id}.png    + masks/{id}.png + references/{id}.png  (exact match)
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def _find_dirs(self):
        """Returns (fmt, img_dir, ref_dir, mask_dir)."""
        # Format test_bench/
        tb = self.data_root / "test_bench"
        if tb.exists():
            gts = sorted(tb.glob("GT_*"))
            if gts:
                refs  = sorted(tb.glob("Ref_*"))
                masks = sorted(tb.glob("Mask_*"))
                return "triplet", gts[0], refs[0] if refs else None, masks[0] if masks else None

        # Format GT_xxx/ in root
        gts = sorted(self.data_root.glob("GT_*"))
        if gts:
            refs  = sorted(self.data_root.glob("Ref_*"))
            masks = sorted(self.data_root.glob("Mask_*"))
            return "triplet", gts[0], refs[0] if refs else None, masks[0] if masks else None

        # Format images/ + masks/ + references/ (Paint-by-Example standard)
        img_dir = self.data_root / "images"
        if img_dir.exists():
            ref_dir  = self.data_root / "references"
            mask_dir = self.data_root / "masks"
            return ("imd", img_dir,
                    ref_dir  if ref_dir.exists()  else None,
                    mask_dir if mask_dir.exists() else None)

        return None, None, None, None

    @staticmethod
    def _build_index(folder):
        """
        Build {id_key -> Path} dict for cross-folder matching.
        Supports both exact stem and prefix-before-last-underscore:
          "000000000283_mask.png" -> keys: {"000000000283_mask", "000000000283"}
          "000001.png"            -> keys: {"000001"}
        """
        if folder is None or not Path(folder).exists():
            return {}
        idx = {}
        for f in Path(folder).iterdir():
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = f.stem
            idx[stem] = f
            if "_" in stem:
                idx[stem.rsplit("_", 1)[0]] = f
        return idx

    @staticmethod
    def _match(img_path, index):
        """Find matching file in index by stem or prefix."""
        stem = img_path.stem
        for key in [stem, stem.rsplit("_", 1)[0] if "_" in stem else stem]:
            if key in index:
                return index[key]
        return None

    def load(self, max_samples=None):
        """Load COCOEE samples with source/reference/mask triplets."""
        fmt, img_dir, ref_dir, mask_dir = self._find_dirs()

        if fmt in ("triplet", "imd") and img_dir and img_dir.exists():
            logger.info(f"COCOEE [{fmt}]: loading dari {img_dir}")
            img_files = sorted(
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if max_samples:
                img_files = img_files[:max_samples]

            ref_idx  = self._build_index(ref_dir)
            mask_idx = self._build_index(mask_dir)
            samples, miss_ref, miss_mask = [], 0, 0

            for img_path in img_files:
                sample = {
                    "source": Image.open(img_path).convert("RGB"),
                    "text":   "replace the object with a similar one",
                }

                rp = self._match(img_path, ref_idx)
                if rp:
                    sample["reference"] = Image.open(rp).convert("RGB")
                else:
                    miss_ref += 1

                mp = self._match(img_path, mask_idx)
                if mp:
                    sample["mask"] = Image.open(mp).convert("L")
                else:
                    miss_mask += 1
                    src = sample["source"]
                    W, H = src.size
                    fb = Image.new("L", src.size, 0)
                    from PIL import ImageDraw
                    ImageDraw.Draw(fb).rectangle(
                        [W // 4, H // 4, 3 * W // 4, 3 * H // 4], fill=255
                    )
                    sample["mask"] = fb

                samples.append(sample)

            logger.info(f"COCOEE: {len(samples)} samples loaded "
                        f"(miss ref={miss_ref}, miss mask={miss_mask})")
            return samples

        # Fallback: annotations.json
        ann_file = self.data_root / "annotations.json"
        if not ann_file.exists():
            logger.warning(f"COCOEE: tidak ada folder gambar di {self.data_root}")
            return []

        with open(ann_file) as f:
            annotations = json.load(f)

        samples = []
        for ann in annotations[:max_samples]:
            src_p  = self.data_root / "images"     / ann["image_file"]
            ref_p  = self.data_root / "references" / ann.get("reference_file", "")
            mask_p = self.data_root / "masks"      / ann.get("mask_file", "")
            if not src_p.exists():
                continue
            sample = {
                "source": Image.open(src_p).convert("RGB"),
                "text":   ann.get("caption", "an object in the scene"),
            }
            if ref_p.exists():
                sample["reference"] = Image.open(ref_p).convert("RGB")
            if mask_p.exists():
                sample["mask"] = Image.open(mask_p).convert("L")
            else:
                src = sample["source"]
                fb  = Image.new("L", src.size, 0)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(fb)
                if "bbox" in ann:
                    x1, y1, x2, y2 = ann["bbox"]
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                sample["mask"] = fb
            samples.append(sample)

        logger.info(f"COCOEE: loaded {len(samples)} samples")
        return samples


class MagicBrushLoader:
    """
    MagicBrush Test dataset loader.
    Source: Zhang et al., MagicBrush (NeurIPS 2023)
    Download: https://github.com/OSU-NLP-Group/MagicBrush
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def load(self, max_samples: Optional[int] = None) -> List[Dict]:
        test_file = self.data_root / "test_dataset.json"
        if not test_file.exists():
            logger.warning(f"MagicBrush test data not found at {test_file}")
            logger.warning("Download: https://github.com/OSU-NLP-Group/MagicBrush")
            return []

        with open(test_file) as f:
            data = json.load(f)

        samples = []
        for item in data[:max_samples]:
            source_path = self.data_root / "images" / item["input"]
            target_path = self.data_root / "images" / item.get("output", item["input"])
            mask_path   = self.data_root / "masks" / item.get("mask", "")

            if not source_path.exists():
                continue

            sample = {
                "source":    Image.open(source_path).convert("RGB"),
                "reference": Image.open(target_path).convert("RGB") if target_path.exists() else None,
                "text":      item.get("instruction", "edit the image"),
            }

            if mask_path.exists():
                sample["mask"] = Image.open(mask_path).convert("L")
            else:
                src = sample["source"]
                sample["mask"] = Image.new("L", src.size, 255)

            samples.append(sample)

        logger.info(f"MagicBrush: loaded {len(samples)} samples")
        return samples


class EmuEditLoader:
    """
    Emu Edit Test dataset loader.
    Source: Sheynin et al., Emu Edit (CVPR 2024)
    Download: https://emu-edit.metademolab.com/
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def load(self, max_samples: Optional[int] = None) -> List[Dict]:
        test_file = self.data_root / "emu_edit_test.json"
        if not test_file.exists():
            logger.warning(f"Emu Edit test data not found at {test_file}")
            logger.warning("Download: https://emu-edit.metademolab.com/")
            return []

        with open(test_file) as f:
            data = json.load(f)

        samples = []
        for item in list(data.items())[:max_samples]:
            item_id, ann = item
            source_path = self.data_root / "images" / ann["input_image"]
            ref_path    = self.data_root / "images" / ann.get("output_image", "")

            if not source_path.exists():
                continue

            sample = {
                "source":    Image.open(source_path).convert("RGB"),
                "reference": Image.open(ref_path).convert("RGB") if ref_path.exists() else None,
                "text":      ann.get("instruction", ""),
            }

            mask_path = self.data_root / "masks" / ann.get("mask", "")
            if mask_path.exists():
                sample["mask"] = Image.open(mask_path).convert("L")
            else:
                src = sample["source"]
                sample["mask"] = Image.new("L", src.size, 255)

            samples.append(sample)

        logger.info(f"Emu Edit: loaded {len(samples)} samples")
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Download Helper
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOAD_INSTRUCTIONS = {
    "cocoee": """
COCOEE Dataset Download Instructions:
  1. git clone https://github.com/Fantasy-Studio/Paint-by-Example
  2. Follow dataset download instructions in that repo
  3. Place in: data/COCOEE/
  Required structure:
    data/COCOEE/
    ├── images/       (contains *_GT.png files)
    ├── masks/        (contains *_mask.png files)
    └── references/   (contains *_ref.png files)
""",
    "magicbrush": """
MagicBrush Test Dataset Download Instructions:
  1. pip install huggingface_hub
  2. python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='osunlp/MagicBrush', repo_type='dataset', local_dir='data/magicbrush')"
  Required structure:
    data/magicbrush/
    ├── test_dataset.json
    ├── images/
    └── masks/
""",
    "emu_edit": """
Emu Edit Test Dataset Download Instructions:
  1. Visit: https://emu-edit.metademolab.com/
  2. Request access and download test split
  3. Place in: data/emu_edit/
  Required structure:
    data/emu_edit/
    ├── emu_edit_test.json
    ├── images/
    └── masks/
""",
}


# ══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model: HybridEditDif,
    samples: List[Dict],
    output_dir: str,
    dataset_name: str,
    w1: float = 7.5,
    w2: float = 7.5,
    num_inference_steps: int = 50,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run full evaluation pipeline."""

    if max_samples:
        samples = samples[:max_samples]

    output_path   = Path(output_dir) / dataset_name
    generated_dir = output_path / "generated"
    real_dir      = output_path / "real"
    generated_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    pipeline = HybridEditDifInferencePipeline(
        model=model,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    generated_images = []
    source_images    = []
    reference_images = []
    masks            = []
    texts            = []

    logger.info(f"\nRunning inference on {len(samples)} {dataset_name} samples...")

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
        src  = sample["source"]
        mask = sample.get("mask", Image.new("L", src.size, 255))
        ref  = sample.get("reference")
        text = sample.get("text", "")

        output = pipeline(
            source_image=src,
            mask_image=mask,
            reference_image=ref,
            text_prompt=text,
            w1=w1,
            w2=w2,
        )

        output.save(generated_dir / f"{i:05d}.jpg", quality=95)
        src.save(real_dir / f"{i:05d}.jpg", quality=95)

        generated_images.append(output)
        source_images.append(src)
        if ref:
            reference_images.append(ref)
        masks.append(mask)
        texts.append(text)

    # Compute metrics
    evaluator = HybridEditDifEvaluator(device=str(device))

    results = evaluator.evaluate(
        generated_images=generated_images,
        source_images=source_images,
        reference_images=reference_images if reference_images else None,
        texts=texts if texts else None,
        masks=masks,
        generated_dir=str(generated_dir),
        real_dir=str(real_dir),
        compute_fid=True,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Results on {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"  FID   ↓ : {results.get('FID', 'N/A'):.3f}")
    logger.info(f"  QS    ↑ : {results.get('QS', 'N/A'):.2f}")
    logger.info(f"  CLIP  ↑ : {results.get('CLIP_Score_img', results.get('CLIP_Score_txt', 'N/A')):.2f}")
    logger.info(f"  SSIM  ↑ : {results.get('SSIM', 'N/A'):.4f}")
    logger.info(f"  LPIPS ↓ : {results.get('LPIPS', 'N/A'):.4f}")
    logger.info(f"  BG MSE↓ : {results.get('BG_MSE', 'N/A'):.6f}")
    logger.info(f"{'='*60}")

    results_file = output_path / "metrics.json"
    with open(results_file, "w") as f:
        json.dump({"dataset": dataset_name, "metrics": results,
                   "n_samples": len(samples)}, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HybridEditDif Evaluation")
    parser.add_argument("--checkpoint",  required=True,  help="Path to .pt weights file")
    parser.add_argument("--sd_model",   default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--dataset",    choices=["cocoee", "magicbrush", "emu_edit", "all"], default="all")
    parser.add_argument("--data_root",  default="./data")
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps",      type=int, default=50)
    parser.add_argument("--w1",         type=float, default=7.5)
    parser.add_argument("--w2",         type=float, default=7.5)
    parser.add_argument("--lambda1",    type=float, default=1.0)
    parser.add_argument("--lambda2",    type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading HybridEditDif model...")
    model = HybridEditDif(sd_model_path=args.sd_model)

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.ddca_layers.load_state_dict(ckpt["ddca_layers"])
        model.image_encoder.mlp.load_state_dict(ckpt["image_encoder_mlp"])
        model.clip_text_encoder.mlp.load_state_dict(ckpt["clip_text_encoder_mlp"])
        if "unet_conv_in" in ckpt:
            model.unet.conv_in.load_state_dict(ckpt["unet_conv_in"])
        logger.info(f"✓ Loaded weights from: {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}. Using random weights.")

    model.to(device).eval()

    datasets_to_eval = (
        ["cocoee", "magicbrush", "emu_edit"] if args.dataset == "all" else [args.dataset]
    )

    all_results = {}
    for ds_name in datasets_to_eval:
        # Try multiple path candidates (case-insensitive, direct path)
        candidates = [
            Path(args.data_root) / ds_name,
            Path(args.data_root) / ds_name.upper(),
            Path(args.data_root),
        ]
        data_path = next((p for p in candidates if p.exists()), None)

        if data_path is None:
            logger.warning(f"\nData not found untuk dataset '{ds_name}'.")
            logger.warning(f"  Dicari di: {[str(c) for c in candidates]}")
            logger.warning(DOWNLOAD_INSTRUCTIONS.get(ds_name, ""))
            continue

        loader_map = {
            "cocoee":     COCOEELoader,
            "magicbrush": MagicBrushLoader,
            "emu_edit":   EmuEditLoader,
        }
        loader  = loader_map[ds_name](str(data_path))
        samples = loader.load(max_samples=args.max_samples)

        if not samples:
            logger.warning(f"No samples loaded for {ds_name}")
            continue

        results = run_evaluation(
            model=model,
            samples=samples,
            output_dir=args.output_dir,
            dataset_name=ds_name,
            w1=args.w1,
            w2=args.w2,
            num_inference_steps=args.steps,
            max_samples=args.max_samples,
        )
        all_results[ds_name] = results

    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY TABLE (matches paper Table 1 format)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Method':<20} {'COCOEE FID':>12} {'COCOEE QS':>10} {'COCOEE CLIP':>12}")
        logger.info(f"{'HybridEditDif':>20} "
                    f"{all_results.get('cocoee', {}).get('FID', 'N/A'):>12} "
                    f"{all_results.get('cocoee', {}).get('QS', 'N/A'):>10} "
                    f"{all_results.get('cocoee', {}).get('CLIP_Score_img', 'N/A'):>12}")
        logger.info(f"{'='*80}")

    combined_file = Path(args.output_dir) / "evaluate" / "all_results.json"
    combined_file.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to: {combined_file}")


if __name__ == "__main__":
    main()
