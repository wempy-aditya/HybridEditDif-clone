"""
COCO 2017 Downloader for HybridEditDif
========================================
Downloads COCO 2017 train/val splits via FiftyOne.

Usage:
    python scripts/download_coco.py --data_root ./data/coco --split train
    python scripts/download_coco.py --data_root ./data/coco --split val --n_samples 2000
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download COCO 2017 for HybridEditDif")
    parser.add_argument("--data_root",  default="./data/coco",
                        help="Target directory for COCO data")
    parser.add_argument("--split",      default="train",
                        choices=["train", "val", "both"])
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Max images to download (None = all)")
    parser.add_argument("--label_types", nargs="+",
                        default=["detections"],
                        help="FiftyOne label types: detections, segmentations")
    return parser.parse_args()


def check_disk_space(data_root: Path, required_gb: float):
    import shutil
    total, used, free = shutil.disk_usage(data_root.parent)
    free_gb  = free / (1024**3)
    logger.info(f"Disk space tersedia: {free_gb:.1f} GB (butuh ~{required_gb:.0f} GB)")
    if free_gb < required_gb * 1.2:
        logger.warning(f"Disk mungkin tidak cukup! Free: {free_gb:.1f} GB, perlu: {required_gb:.0f} GB")


def download_coco_via_fiftyone(
    data_root: Path,
    split: str,
    n_samples: int = None,
    label_types: list = None,
):
    """Download COCO via FiftyOne."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("FiftyOne tidak terinstall. Jalankan: pip install fiftyone")
        sys.exit(1)

    fo_split = "train" if split == "train" else "validation"

    kwargs = {
        "dataset_name": f"coco-2017-{split}-hybridedif",
        "label_types":  label_types or ["detections"],
        "split":        fo_split,
    }
    if n_samples:
        kwargs["max_samples"] = n_samples

    logger.info(f"Downloading COCO 2017 ({split}, {n_samples or 'all'} samples)...")
    dataset = foz.load_zoo_dataset("coco-2017", **kwargs)

    # ── Export ke struktur yang diharapkan ─────────────────────────────────
    images_dir = data_root / "images" / f"{split}2017"
    ann_dir    = data_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {len(dataset)} images ke {images_dir}...")
    dataset.export(
        export_dir=str(data_root),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        split=fo_split,
    )

    # Cek annotation file
    possible_ann_files = list(ann_dir.glob("*.json"))
    if possible_ann_files:
        target_ann = ann_dir / f"instances_{split}2017.json"
        if not target_ann.exists():
            shutil.copy(possible_ann_files[0], target_ann)
            logger.info(f"Annotation file: {target_ann}")
    else:
        logger.warning("Annotation JSON tidak ditemukan. Export manual mungkin diperlukan.")

    return len(dataset)


def download_coco_direct(
    data_root: Path,
    split: str,
    n_samples: int = None,
):
    """
    Alternatif: Download COCO langsung tanpa FiftyOne.
    Menggunakan pycocotools dan wget.
    """
    import subprocess
    import zipfile

    ann_dir    = data_root / "annotations"
    images_dir = data_root / "images" / f"{split}2017"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = data_root / "annotations_trainval2017.zip"
    if not (ann_dir / f"instances_{split}2017.json").exists():
        logger.info(f"Downloading annotations dari {ann_url}...")
        subprocess.run(["wget", "-c", ann_url, "-O", str(ann_zip)], check=True)
        logger.info("Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(str(data_root))
        ann_zip.unlink(missing_ok=True)

    # Download images
    img_url  = f"http://images.cocodataset.org/zips/{split}2017.zip"
    img_zip  = data_root / f"{split}2017.zip"
    if not any(images_dir.glob("*.jpg")):
        if n_samples and n_samples < 5000:
            # Gunakan COCO API untuk download selective
            logger.info(f"Downloading {n_samples} COCO {split} images via API...")
            _download_coco_selective(ann_dir / f"instances_{split}2017.json",
                                     images_dir, n_samples, split)
        else:
            logger.info(f"Downloading COCO {split} images (~18GB)...")
            subprocess.run(["wget", "-c", img_url, "-O", str(img_zip)], check=True)
            logger.info("Extracting images...")
            with zipfile.ZipFile(img_zip) as z:
                z.extractall(str(data_root / "images"))
            img_zip.unlink(missing_ok=True)

    imgs = list(images_dir.glob("*.jpg"))
    logger.info(f"Total: {len(imgs)} images di {images_dir}")
    return len(imgs)


def _download_coco_selective(ann_file: Path, images_dir: Path, n: int, split: str):
    """Download hanya n gambar dari COCO menggunakan pycocotools."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        logger.error("pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return

    import requests
    from tqdm import tqdm

    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()[:n]
    imgs    = coco.loadImgs(img_ids)

    base_url = f"http://images.cocodataset.org/{split}2017"
    for img_info in tqdm(imgs, desc=f"Downloading COCO {split}"):
        dest = images_dir / img_info["file_name"]
        if dest.exists():
            continue
        try:
            r = requests.get(f"{base_url}/{img_info['file_name']}", timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:
            logger.debug(f"Failed {img_info['file_name']}: {e}")


def main():
    args   = parse_args()
    splits = ["train", "val"] if args.split == "both" else [args.split]
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Estimate disk space
    est_gb = 20.0 if args.n_samples is None else (args.n_samples * 0.003)
    check_disk_space(data_root, est_gb)

    logger.info("=" * 60)
    logger.info("COCO 2017 Downloader — HybridEditDif")
    logger.info("=" * 60)
    logger.info(f"  Target    : {data_root.resolve()}")
    logger.info(f"  Split     : {args.split}")
    logger.info(f"  N samples : {args.n_samples or 'all'}")
    logger.info("=" * 60)

    total = 0
    for split in splits:
        try:
            # Coba FiftyOne dulu (lebih mudah)
            n = download_coco_via_fiftyone(
                data_root=data_root,
                split=split,
                n_samples=args.n_samples,
                label_types=args.label_types,
            )
        except Exception as e:
            logger.warning(f"FiftyOne gagal ({e}). Coba direct download...")
            n = download_coco_direct(
                data_root=data_root,
                split=split,
                n_samples=args.n_samples,
            )
        total += n
        logger.info(f"✓ {split}: {n} images downloaded")

    logger.info(f"\n✅ Done! Total: {total} COCO images di {data_root}")
    logger.info(f"   Struktur:")
    logger.info(f"   {data_root}/images/train2017/ — images")
    logger.info(f"   {data_root}/annotations/instances_train2017.json")
    logger.info(f"\nTrain command:")
    logger.info(f"   python scripts/train.py --config configs/train_config_coco.yaml")


if __name__ == "__main__":
    main()
