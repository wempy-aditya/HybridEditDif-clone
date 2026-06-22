"""
HybridEditDif — Quick Inference Test
======================================
Menguji model yang sudah di-train dengan gambar dari dataset.

Mode 1: Single image (--source wajib diisi)
    python scripts/run_inference.py \\
        --weights checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \\
        --source path/gambar.jpg \\
        --reference path/ref.jpg \\
        --mask path/mask.png \\
        --prompt "replace with batik pattern"

Mode 2: Batch dari folder (--images_dir)
    python scripts/run_inference.py \\
        --weights checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \\
        --images_dir data/openimages/images/train \\
        --reference path/ref.jpg \\
        --output_dir experiments/inference \\
        --n_images 5
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.hybrid_edit_dif import HybridEditDif
from src.models.inference import HybridEditDifInferencePipeline


def load_model(weights_path: str, sd_model_path: str, device: str):
    print(f"Loading HybridEditDif dari SD: {sd_model_path}")
    model = HybridEditDif(sd_model_path=sd_model_path)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {weights_path}\n"
            f"Pastikan path benar, contoh:\n"
            f"  checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt"
        )

    print(f"Loading weights: {weights_path}")
    weights = torch.load(str(weights_path), map_location="cpu")

    model.ddca_layers.load_state_dict(weights["ddca_layers"])
    model.image_encoder.mlp.load_state_dict(weights["image_encoder_mlp"])
    model.clip_text_encoder.mlp.load_state_dict(weights["clip_text_encoder_mlp"])
    if "unet_conv_in" in weights:
        model.unet.conv_in.load_state_dict(weights["unet_conv_in"])

    print(f"✅ Weights loaded ({len(model.ddca_layers)} DDCA layers)")
    return model


def make_center_mask(size=(512, 512), ratio=0.4):
    """Buat mask kotak di tengah gambar (default jika --mask tidak diisi)."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    W, H = size
    margin_x = int(W * (0.5 - ratio / 2))
    margin_y  = int(H * (0.5 - ratio / 2))
    draw.rectangle([margin_x, margin_y, W - margin_x, H - margin_y], fill=255)
    return mask


def load_image(path: str, size=(512, 512), mode="RGB") -> Image.Image:
    img = Image.open(path).convert(mode)
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_comparison(src, mask, ref, result, out_path: Path):
    """Simpan grid: [Source | Mask | Reference | Result]."""
    W = 512
    panels = [src, mask.convert("RGB"), ref if ref else Image.new("RGB", (W, W), (100, 100, 100)), result]
    grid = Image.new("RGB", (W * len(panels), W))
    for i, p in enumerate(panels):
        grid.paste(p.resize((W, W), Image.LANCZOS), (i * W, 0))

    # Label
    draw = ImageDraw.Draw(grid)
    for i, label in enumerate(["Source", "Mask", "Reference", "Generated"]):
        draw.text((i * W + 8, 8), label, fill=(255, 255, 0))

    grid.save(str(out_path), quality=95)


def run_single(pipe, src_img, mask_img, ref_img, prompt, w1, w2, out_path, label=""):
    """Jalankan inference untuk satu gambar."""
    result = pipe(
        source_image    = src_img,
        mask_image      = mask_img,
        reference_image = ref_img,
        text_prompt     = prompt,
        w1              = w1,
        w2              = w2,
        seed            = 42,
    )
    save_comparison(src_img, mask_img, ref_img, result, out_path)
    print(f"   ✅ Saved: {out_path}  [source | mask | reference | result]")
    return result


def main():
    parser = argparse.ArgumentParser(description="HybridEditDif Inference")

    # Model
    parser.add_argument("--weights",    default="checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt",
                        help="Path ke file .pt hasil training")
    parser.add_argument("--sd_model",   default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device",     default="cuda:1")

    # Input — Mode 1: Single image
    parser.add_argument("--source",     default=None,
                        help="[Mode 1] Path ke satu gambar source")
    parser.add_argument("--mask",       default=None,
                        help="[Mode 1 & 2] Path ke mask image (putih=area edit, hitam=jaga). "
                             "Jika tidak diisi, pakai mask kotak tengah otomatis")
    parser.add_argument("--reference",  default=None,
                        help="[Mode 1 & 2] Path ke reference image. "
                             "Jika tidak diisi, reference = gambar source sendiri")

    # Input — Mode 2: Batch folder
    parser.add_argument("--images_dir", default=None,
                        help="[Mode 2] Folder berisi banyak gambar (jpg/png)")
    parser.add_argument("--masks_dir",  default=None,
                        help="[Mode 2] Folder berisi mask per gambar (nama harus sama). "
                             "Jika tidak diisi, semua pakai center mask")
    parser.add_argument("--refs_dir",   default=None,
                        help="[Mode 2] Folder berisi reference per gambar (nama harus sama). "
                             "Jika tidak diisi, semua pakai self-reference")

    # Output & inference params
    parser.add_argument("--output_dir", default="experiments/inference")
    parser.add_argument("--n_images",   type=int, default=5,
                        help="Jumlah gambar yang diproses (Mode 2)")
    parser.add_argument("--steps",      type=int, default=30)
    parser.add_argument("--w1",         type=float, default=2.0,  help="Text guidance scale")
    parser.add_argument("--w2",         type=float, default=2.0,  help="Image guidance scale")
    parser.add_argument("--prompt",     default="a batik pattern object in the scene")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.weights, args.sd_model, args.device)
    pipe  = HybridEditDifInferencePipeline(
        model=model,
        scheduler_type="ddim",
        num_inference_steps=args.steps,
        device=args.device,
    )
    print(f"✅ Pipeline ready on {args.device}\n")

    # ── Mode 1: Single image ──────────────────────────────────────────────────
    if args.source:
        print(f"[Mode 1] Single image inference")
        print(f"  source    : {args.source}")
        print(f"  mask      : {args.mask or 'auto (center box)'}")
        print(f"  reference : {args.reference or 'self (same as source)'}")
        print(f"  prompt    : {args.prompt}")
        print(f"  w1={args.w1}  w2={args.w2}  steps={args.steps}\n")

        src_img = load_image(args.source)
        mask_img = load_image(args.mask, mode="L") if args.mask else make_center_mask()
        ref_img  = load_image(args.reference) if args.reference else src_img.copy()

        out_path = out_dir / f"result_{Path(args.source).stem}.jpg"
        run_single(pipe, src_img, mask_img, ref_img, args.prompt,
                   args.w1, args.w2, out_path)

    # ── Mode 2: Batch folder ──────────────────────────────────────────────────
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        test_images = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )[:args.n_images]

        if not test_images:
            print(f"⚠ Tidak ada gambar di: {images_dir}")
            return

        print(f"[Mode 2] Batch inference: {len(test_images)} gambar")
        print(f"  source dir : {images_dir}")
        print(f"  masks_dir  : {args.masks_dir or 'auto (center box)'}")
        print(f"  refs_dir   : {args.refs_dir or 'self-reference'}")
        print(f"  reference  : {args.reference or 'per-gambar (refs_dir) / self'}")
        print(f"  prompt     : {args.prompt}")
        print(f"  w1={args.w1}  w2={args.w2}  steps={args.steps}\n")

        # Global reference (satu file untuk semua gambar)
        global_ref = load_image(args.reference) if args.reference else None

        for i, img_path in enumerate(test_images, 1):
            stem = img_path.stem
            print(f"[{i}/{len(test_images)}] {img_path.name}")

            src_img = load_image(str(img_path))

            # Mask: per-file dari masks_dir → global --mask → auto center
            if args.masks_dir:
                for ext in [".png", ".jpg", ".jpeg"]:
                    m = Path(args.masks_dir) / (stem + ext)
                    if m.exists():
                        mask_img = load_image(str(m), mode="L")
                        break
                else:
                    mask_img = make_center_mask()
            elif args.mask:
                mask_img = load_image(args.mask, mode="L")
            else:
                mask_img = make_center_mask()

            # Reference: per-file dari refs_dir → global --reference → self
            if args.refs_dir:
                for ext in [".png", ".jpg", ".jpeg"]:
                    r = Path(args.refs_dir) / (stem + ext)
                    if r.exists():
                        ref_img = load_image(str(r))
                        break
                else:
                    ref_img = global_ref or src_img.copy()
            else:
                ref_img = global_ref or src_img.copy()

            out_path = out_dir / f"result_{stem}.jpg"
            run_single(pipe, src_img, mask_img, ref_img, args.prompt,
                       args.w1, args.w2, out_path, label=stem)

    else:
        print("⚠ Tentukan --source (single) atau --images_dir (batch).")
        print("Contoh:")
        print("  python scripts/run_inference.py --source foto.jpg --reference ref.jpg")
        print("  python scripts/run_inference.py --images_dir data/openimages/images/train --n_images 5")

    print(f"\nDone! Semua hasil ada di: {out_dir}/")


if __name__ == "__main__":
    main()
