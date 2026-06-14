"""
Download OpenImages V7 Subset untuk HybridEditDif Training
===========================================================
Paper Section 4.1: "We used the OpenImages dataset as the primary source
for training due to its extensive coverage of 1.9 million images and
16 million annotated bounding boxes across 600 object classes."

Script ini mendownload subset yang manageable untuk training:
  - Default: 50,000 gambar train + annotations bbox
  - Bisa disesuaikan dengan --n_samples

Usage:
    # Download 50k gambar (recommended untuk mulai)
    python scripts/download_openimages.py --n_samples 50000

    # Pakai cache FiftyOne dari HDD (tanpa download ulang)
    python scripts/download_openimages.py --n_samples 50000 \\
        --fiftyone_dir /mnt/storage/fiftyone

    # Download lebih sedikit untuk testing
    python scripts/download_openimages.py --n_samples 1000 --split train

    # Download full (1.9M, butuh ~500GB disk)
    python scripts/download_openimages.py --n_samples 0 --split train
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def check_disk_space(path: str, required_gb: float):
    """Cek apakah ada cukup disk space."""
    import shutil
    free = shutil.disk_usage(path).free / 1024**3
    logger.info(f"Disk space tersedia: {free:.1f} GB (butuh ~{required_gb:.0f} GB)")
    if free < required_gb:
        logger.warning(f"⚠ Disk space mungkin tidak cukup! Tersedia: {free:.1f}GB")
    return free >= required_gb


def download_via_fiftyone(
    data_root: str,
    n_samples: int,
    split: str = "train",
    fiftyone_dir: str = None,
):
    """
    Download menggunakan FiftyOne (paling mudah).
    FiftyOne otomatis handle download + bbox annotations.

    fiftyone_dir: override lokasi FiftyOne database/cache.
                  Gunakan ini jika cache sudah dipindah ke HDD.
    """
    # ── Set FiftyOne dir SEBELUM import fo ─────────────────────────────────
    # FiftyOne membaca env var saat pertama kali diimport, jadi harus diset dulu.
    if fiftyone_dir:
        fo_dir = str(Path(fiftyone_dir).resolve())
        os.environ["FIFTYONE_DATABASE_DIR"]    = fo_dir
        os.environ["FIFTYONE_DATASET_ZOO_DIR"] = fo_dir
        logger.info(f"FiftyOne dir → {fo_dir}")
    elif "FIFTYONE_DATABASE_DIR" in os.environ:
        logger.info(f"FiftyOne dir (env) → {os.environ['FIFTYONE_DATABASE_DIR']}")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("FiftyOne tidak terinstall. Jalankan: pip install fiftyone")
        return False

    images_dir = Path(data_root) / "images" / split
    ann_dir    = Path(data_root) / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mendownload OpenImages V7 ({split}) via FiftyOne...")
    logger.info(f"  Jumlah gambar : {n_samples if n_samples > 0 else 'ALL (~1.9M)'}")
    logger.info(f"  Destinasi     : {images_dir}")

    # NOTE: Jangan pass 'dataset_dir' ke load_zoo_dataset() — konflik di FiftyOne baru
    # FiftyOne menyimpan di ~/fiftyone/ secara default
    kwargs = {
        "split": split,
        "label_types": ["detections"],
    }
    if n_samples > 0:
        kwargs["max_samples"] = n_samples

    # Cek apakah dataset sudah ada di FiftyOne persistent store
    dataset_name = f"open-images-v7-{split}-{n_samples}"
    if fo.dataset_exists(dataset_name):
        logger.info(f"Dataset '{dataset_name}' sudah ada di FiftyOne cache, loading...")
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = foz.load_zoo_dataset("open-images-v7", **kwargs)
        dataset.name = dataset_name
        dataset.persistent = True
        dataset.save()

    # Export ke format yang dibutuhkan openimages_dataset.py
    logger.info("Mengexport gambar dan annotations...")
    import json, shutil

    bbox_annotations = {}
    text_annotations = {}
    copied = 0
    skipped = 0

    logger.info(f"Mengexport {len(dataset)} gambar dan annotations ke {images_dir}...")

    for i, sample in enumerate(dataset):
        if i % 500 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)} ({copied} copied, {skipped} skipped)")

        img_id = Path(sample.filepath).stem
        src    = sample.filepath
        dst    = images_dir / f"{img_id}.jpg"

        # Copy gambar (skip kalau sudah ada)
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                logger.warning(f"  ⚠ Gagal copy {img_id}: {e}")
                skipped += 1
                continue

        # Ambil bbox — FiftyOne menyimpan dalam format normalized [x, y, w, h] (sudah 0-1)
        # TIDAK perlu W, H — langsung convert ke (XMin, YMin, XMax, YMax)
        bboxes = []
        label  = "an object"
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                x, y, w, h = det.bounding_box   # semua sudah normalized [0,1]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(1.0, x + w), min(1.0, y + h)
                if x2 > x1 and y2 > y1:         # pastikan bbox valid
                    bboxes.append((x1, y1, x2, y2))
            label = sample.ground_truth.detections[0].label

        if bboxes:
            bbox_annotations[img_id] = bboxes
            text_annotations[img_id] = label

    # Simpan annotations
    bbox_file = ann_dir / f"{split}_bbox_annotations.json"
    text_file  = ann_dir / "text_annotations.json"

    with open(bbox_file, "w") as f:
        json.dump(bbox_annotations, f)
    with open(text_file, "w") as f:
        json.dump(text_annotations, f)

    logger.info(f"✅ Export selesai!")
    logger.info(f"   Gambar di-copy  : {copied}")
    logger.info(f"   Gambar di-skip  : {skipped} (sudah ada)")
    logger.info(f"   BBox annotations: {len(bbox_annotations)} gambar → {bbox_file}")
    logger.info(f"   Text annotations: {len(text_annotations)} gambar → {text_file}")
    return True



def download_via_csv(
    data_root: str,
    n_samples: int,
    split: str = "train",
):
    """
    Download annotations CSV dari Google Storage,
    lalu gambar via openimages-downloader (oi_download_dataset).
    Fallback kalau FiftyOne tidak tersedia.
    """
    try:
        import oidv6
    except ImportError:
        logger.info("Menginstall openimages downloader...")
        os.system("pip install openimages -q")

    ann_dir    = Path(data_root) / "annotations"
    images_dir = Path(data_root) / "images" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Download annotations CSV dari Google Storage...")

    # Download bbox annotations
    import requests

    urls = {
        "train_bbox": "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
        "train_bbox_ann": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
        "train_image_ids": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
        "val_bbox_ann": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "val_image_ids": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    }

    for name, url in urls.items():
        fname = ann_dir / f"{name}.csv"
        if fname.exists():
            logger.info(f"  ✓ {fname.name} sudah ada, skip.")
            continue
        logger.info(f"  Downloading {fname.name}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info("✅ Annotations CSV selesai didownload!")
    logger.info("")
    logger.info("Untuk download gambar, jalankan:")
    logger.info(f"  python -m oidv6 downloader --dataset {data_root} \\")
    logger.info(f"    --type_data {split} --classes 'all' \\")
    logger.info(f"    --limit {n_samples if n_samples > 0 else 0} --yes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download OpenImages V7 untuk HybridEditDif")
    parser.add_argument("--data_root", default="./data/openimages",
                        help="Folder destinasi dataset (default: ./data/openimages)")
    parser.add_argument("--n_samples", type=int, default=50000,
                        help="Jumlah gambar yang didownload (0=semua, default: 50000)")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="train", help="Split dataset (default: train)")
    parser.add_argument("--method", choices=["fiftyone", "csv"],
                        default="fiftyone", help="Metode download (default: fiftyone)")
    parser.add_argument("--fiftyone_dir", default=None,
                        help="Override lokasi FiftyOne database/cache. "
                             "Gunakan jika cache sudah dipindah ke HDD, "
                             "contoh: --fiftyone_dir /mnt/storage/fiftyone")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    # Estimasi disk space
    gb_needed = max(1, args.n_samples // 1000)  # ~1GB per 1000 gambar
    logger.info("=" * 60)
    logger.info("OpenImages V7 Downloader — HybridEditDif")
    logger.info("=" * 60)
    logger.info(f"Target folder : {data_root}")
    logger.info(f"Split         : {args.split}")
    logger.info(f"Jumlah target : {args.n_samples if args.n_samples > 0 else 'semua'} gambar")
    logger.info(f"Est. disk     : ~{gb_needed} GB")
    check_disk_space(str(data_root.parent), gb_needed * 1.2)
    logger.info("=" * 60)

    if args.method == "fiftyone":
        success = download_via_fiftyone(
            str(data_root), args.n_samples, args.split,
            fiftyone_dir=args.fiftyone_dir,
        )
        if not success:
            logger.info("Fallback ke metode CSV...")
            download_via_csv(str(data_root), args.n_samples, args.split)
    else:
        download_via_csv(str(data_root), args.n_samples, args.split)


if __name__ == "__main__":
    main()
