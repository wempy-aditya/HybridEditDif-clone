"""
HybridEditDif — Quick Inference Test
======================================
Menguji model yang sudah di-train dengan gambar dari dataset.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --device cuda:1 --steps 30 --n_images 3
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
    print(f"Loading HybridEditDif...")
    model = HybridEditDif(sd_model_path=sd_model_path)

    print(f"Loading trained weights dari: {weights_path}")
    weights = torch.load(weights_path, map_location="cpu")

    model.ddca_layers.load_state_dict(weights["ddca_layers"])
    model.image_encoder.mlp.load_state_dict(weights["image_encoder_mlp"])
    model.clip_text_encoder.mlp.load_state_dict(weights["clip_text_encoder_mlp"])
    model.unet.conv_in.load_state_dict(weights["unet_conv_in"])

    print(f"✅ Weights loaded ({len(model.ddca_layers)} DDCA layers)")
    return model


def make_center_mask(size=(512, 512), ratio=0.4):
    """Buat mask kotak di tengah gambar."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    W, H  = size
    margin_x = int(W * (0.5 - ratio / 2))
    margin_y  = int(H * (0.5 - ratio / 2))
    draw.rectangle(
        [margin_x, margin_y, W - margin_x, H - margin_y],
        fill=255
    )
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default="checkpoints/final_model/hybrid_edit_dif_weights.pt")
    parser.add_argument("--sd_model",   default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--images_dir", default="data/openimages/images/train")
    parser.add_argument("--output_dir", default="experiments/inference")
    parser.add_argument("--device",     default="cuda:1")
    parser.add_argument("--steps",      type=int, default=30)
    parser.add_argument("--n_images",   type=int, default=3)
    parser.add_argument("--w1",         type=float, default=7.5,  help="text guidance scale")
    parser.add_argument("--w2",         type=float, default=5.0,  help="image guidance scale")
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

    # Pilih gambar test
    images_dir  = Path(args.images_dir)
    test_images = sorted(images_dir.glob("*.jpg"))[:args.n_images]
    print(f"Menjalankan inference pada {len(test_images)} gambar...")
    print(f"  prompt   : {args.prompt}")
    print(f"  steps    : {args.steps}")
    print(f"  w1 (text): {args.w1}  |  w2 (image): {args.w2}\n")

    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] {img_path.name}")

        src_img = Image.open(img_path).convert("RGB").resize((512, 512))
        mask    = make_center_mask((512, 512), ratio=0.35)

        result  = pipe(
            source_image    = src_img,
            mask_image      = mask,
            reference_image = src_img,   # self-reference (gunakan gambar sendiri)
            text_prompt     = args.prompt,
            w1              = args.w1,
            w2              = args.w2,
            seed            = 42,
        )

        # Simpan: source | mask | result side-by-side
        comparison = Image.new("RGB", (512 * 3, 512))
        comparison.paste(src_img, (0, 0))
        comparison.paste(mask.convert("RGB"), (512, 0))
        comparison.paste(result, (1024, 0))

        out_path = out_dir / f"result_{img_path.stem}.jpg"
        comparison.save(out_path, quality=95)
        print(f"   ✅ Saved: {out_path}  [source | mask | result]\n")

    print(f"Done! Semua hasil ada di: {out_dir}/")


if __name__ == "__main__":
    main()
