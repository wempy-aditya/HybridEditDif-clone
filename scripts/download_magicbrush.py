"""
MagicBrush Dataset Downloader for HybridEditDif
=================================================
MagicBrush adalah dataset editing berbasis instruksi manusia real.
Total: ~10k edit sessions, ~5k gambar unik.
Size: ~3GB (sangat ringan dibanding OpenImages/COCO)

Source: https://huggingface.co/datasets/osunlp/MagicBrush
Paper : Zhang et al. NeurIPS 2023

Usage:
    # Download semua data (~3GB)
    python scripts/download_magicbrush.py --data_root ./data/magicbrush

    # Download hanya N samples train
    python scripts/download_magicbrush.py --data_root ./data/magicbrush --n_samples 2000

    # Download split tertentu
    python scripts/download_magicbrush.py --data_root ./data/magicbrush --split train
    python scripts/download_magicbrush.py --data_root ./data/magicbrush --split test

Output structure:
    data/magicbrush/
    ├── train_dataset.json
    ├── test_dataset.json
    └── images/
        ├── {session_id}_input.png
        ├── {session_id}_output.png
        └── {session_id}_mask.png
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HUGGINGFACE_REPO = "osunlp/MagicBrush"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MagicBrush dataset untuk HybridEditDif"
    )
    parser.add_argument(
        "--data_root", default="./data/magicbrush",
        help="Folder tujuan download (default: ./data/magicbrush)"
    )
    parser.add_argument(
        "--split", default="both",
        choices=["train", "test", "both"],
        help="Split yang didownload (default: both)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Jumlah sample yang didownload. "
             "None = semua (~10k). Rekomendasi: 3000 untuk training cepat."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed untuk sampling"
    )
    return parser.parse_args()


def check_disk_space(data_root: Path, required_gb: float):
    total, used, free = shutil.disk_usage(str(data_root.parent))
    free_gb = free / (1024 ** 3)
    logger.info(f"Disk tersedia: {free_gb:.1f} GB | Dibutuhkan: ~{required_gb:.1f} GB")
    if free_gb < required_gb * 1.2:
        logger.warning(f"Disk mungkin tidak cukup! Tersedia {free_gb:.1f} GB")


def download_via_huggingface_hub(
    data_root: Path,
    split: str,
    n_samples: int = None,
    seed: int = 42,
):
    """
    Download MagicBrush dari HuggingFace Hub menggunakan datasets library.
    Ini adalah cara paling mudah dan reliable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Library 'datasets' belum terinstall.\n"
            "  Jalankan: pip install datasets"
        )

    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    splits_to_dl = ["train", "test"] if split == "both" else [split]
    all_metadata = {}

    for sp in splits_to_dl:
        logger.info(f"Loading MagicBrush '{sp}' dari HuggingFace ({HUGGINGFACE_REPO})...")
        ds = load_dataset(HUGGINGFACE_REPO, split=sp)

        # Sample jika diminta
        if n_samples and n_samples < len(ds):
            random.seed(seed)
            indices = random.sample(range(len(ds)), n_samples)
            ds = ds.select(indices)
            logger.info(f"  Sampled {n_samples} dari {len(ds)} total '{sp}' samples")
        else:
            logger.info(f"  Downloading semua {len(ds)} '{sp}' samples")

        metadata = []
        for i, item in enumerate(ds):
            # MagicBrush HuggingFace format fields:
            # image, mask, source_img, instruction, tgt_img, etc.
            session_id = f"{sp}_{i:06d}"

            # Save source/input image
            input_img   = item.get("source_img") or item.get("image")
            output_img  = item.get("tgt_img") or item.get("edited_image")
            mask_img    = item.get("mask")
            instruction = (item.get("instruction") or item.get("caption") or
                           "edit the image")

            if input_img is None:
                continue

            input_path  = images_dir / f"{session_id}_input.png"
            output_path = images_dir / f"{session_id}_output.png" if output_img else None
            mask_path   = images_dir / f"{session_id}_mask.png"   if mask_img  else None

            input_img.save(str(input_path))
            if output_img:
                output_img.save(str(output_path))
            if mask_img:
                mask_img.save(str(mask_path))

            metadata.append({
                "input":       str(input_path.name),
                "output":      str(output_path.name) if output_path else None,
                "mask":        str(mask_path.name)   if mask_path   else None,
                "instruction": instruction,
            })

            if (i + 1) % 500 == 0:
                logger.info(f"  Progress: {i+1}/{len(ds)} images saved...")

        all_metadata[sp] = metadata

        # Save annotation JSON
        ann_file = data_root / f"{sp}_dataset.json"
        with open(ann_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✅ {sp}: {len(metadata)} samples | annotation: {ann_file}")

    return all_metadata


def download_via_snapshot(
    data_root: Path,
    split: str,
    n_samples: int = None,
    seed: int = 42,
):
    """
    Alternatif: Download seluruh repo via snapshot_download lalu parse manual.
    Gunakan ini jika 'datasets' library gagal.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "Library 'huggingface_hub' belum terinstall.\n"
            "  Jalankan: pip install huggingface_hub"
        )

    logger.info(f"Downloading MagicBrush repo snapshot dari {HUGGINGFACE_REPO}...")
    local_dir = data_root / "hf_cache"
    snapshot_download(
        repo_id=HUGGINGFACE_REPO,
        repo_type="dataset",
        local_dir=str(local_dir),
        ignore_patterns=["*.parquet"],  # skip parquet, kita mau JSON + images
    )
    logger.info(f"Snapshot downloaded ke: {local_dir}")

    # Cari annotation files di dalam snapshot
    ann_files = list(local_dir.rglob("*.json"))
    images_found = list(local_dir.rglob("*.png")) + list(local_dir.rglob("*.jpg"))

    logger.info(f"  Found {len(ann_files)} JSON files, {len(images_found)} images")

    # Copy ke struktur yang diharapkan
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for img in images_found:
        dest = images_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)

    for af in ann_files:
        dest = data_root / af.name
        if not dest.exists():
            shutil.copy2(af, dest)

    logger.info(f"  ✅ Files copied ke {data_root}")
    return len(images_found)


def main():
    args      = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Estimasi ukuran
    if args.n_samples:
        est_gb = args.n_samples * 0.0003 * 3   # ~0.3MB per triplet (input+output+mask)
    else:
        est_gb = 3.5  # MagicBrush full = ~3-4GB

    logger.info("=" * 60)
    logger.info("MagicBrush Downloader — HybridEditDif")
    logger.info("=" * 60)
    logger.info(f"  Target    : {data_root.resolve()}")
    logger.info(f"  Split     : {args.split}")
    logger.info(f"  N samples : {args.n_samples or 'semua (~10k)'}")
    logger.info(f"  Est. size : ~{est_gb:.1f} GB")
    logger.info("=" * 60)

    check_disk_space(data_root, est_gb)

    # Coba download via 'datasets' library dulu
    try:
        result = download_via_huggingface_hub(
            data_root=data_root,
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        total = sum(len(v) for v in result.values()) if isinstance(result, dict) else 0
    except ImportError as e:
        logger.warning(f"{e}")
        logger.info("Mencoba via snapshot_download...")
        try:
            total = download_via_snapshot(
                data_root=data_root,
                split=args.split,
                n_samples=args.n_samples,
                seed=args.seed,
            )
        except Exception as e2:
            logger.error(f"Download gagal: {e2}")
            logger.info("\nManual download option:")
            logger.info("  pip install datasets huggingface_hub")
            logger.info("  python -c \"from datasets import load_dataset; "
                        f"ds = load_dataset('{HUGGINGFACE_REPO}', split='train'); "
                        f"ds.save_to_disk('{data_root}')\"")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Download via datasets gagal: {e}")
        logger.info("Mencoba via snapshot_download...")
        total = download_via_snapshot(
            data_root=data_root,
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    logger.info(f"\n✅ Done! {total} MagicBrush samples di {data_root}")
    logger.info(f"\nStruktur:")
    logger.info(f"  {data_root}/train_dataset.json")
    logger.info(f"  {data_root}/images/{{id}}_input.png")
    logger.info(f"  {data_root}/images/{{id}}_output.png")
    logger.info(f"  {data_root}/images/{{id}}_mask.png")
    logger.info(f"\nTrain command:")
    logger.info(f"  CUDA_VISIBLE_DEVICES=1 python scripts/train.py "
                f"--config configs/train_config_magicbrush.yaml")


if __name__ == "__main__":
    main()
