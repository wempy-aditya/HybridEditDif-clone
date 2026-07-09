# HybridEditDif

Implementasi ulang (rekonstruksi) dari paper:

> **"HybridEditDif: Hybrid Guided Image Editing via Diffusion Model"**  
> Liu et al., *Pattern Recognition*, 2026  
> DOI: `10.1016/j.patcog.2025.111732`

Model ini menggabungkan **text conditioning** dan **image conditioning** secara bersamaan untuk melakukan inpainting dan editing gambar yang dipandu oleh referensi visual dan deskripsi teks.

---

## Arsitektur

```
Source Image (X_s) ──┐
                     ├──► VAE Encoder ──► Masked Latent
Mask (m)        ──── ┘

Reference (X_r) ──► OpenCLIP ViT-H/14 ──► MLP ──► c_i (image context)
Text Prompt (T) ──► OpenCLIP ViT-g/14 ──► MLP ──► c_t (text context)

                    ┌─────────────────────────────────┐
                    │  Stable Diffusion UNet (frozen)  │
                    │    + 16 DDCA Layers (trainable)  │
                    └─────────────────────────────────┘
                                    │
                              VAE Decoder
                                    │
                           Output Image (Y)
```

**Dynamic Decoupled Cross-Attention (DDCA)** — komponen utama yang diinjeksi ke dalam 16 cross-attention layer UNet:
- **Text Branch**: menggunakan `c_t` dari OpenCLIP ViT-g/14
- **Image Branch**: menggunakan `c_i` dari OpenCLIP ViT-H/14
- Kedua branch digabung secara adaptif dengan bobot `λ1` dan `λ2`

**Training Loss (Eq. 11):**
```
L = E_{t,y₀,ε} ‖ ε_θ(y_t, m̄⊙X_s, c_i, c_t, t) - ε ‖²₂
```

**Inference dengan Classifier-Free Guidance (Eq. 12):**
```
ε̂ = w1·ε_θ(..., c_t) + w2·ε_θ(..., c_i) + (1-w1-w2)·ε_θ(...)
```

---

## Struktur Project

```
HybridEditDif/
├── src/
│   ├── models/
│   │   ├── hybrid_edit_dif.py   # Model utama (DDCA injection ke SD UNet)
│   │   ├── attention.py         # Dynamic Decoupled Cross-Attention (DDCA)
│   │   ├── encoders.py          # CLIP image & text encoders + MLP heads
│   │   └── inference.py         # Inference pipeline dengan CFG
│   ├── data/
│   │   ├── dataset_factory.py   # Universal dataset builder & DataLoader
│   │   ├── openimages_dataset.py
│   │   ├── coco_dataset.py
│   │   └── magicbrush_dataset.py
│   └── utils/
│       └── metrics.py           # FID, SSIM, LPIPS, CLIP Score, BG MSE
│
├── scripts/
│   ├── train.py                 # Training script (single/multi-GPU)
│   ├── evaluate.py              # Benchmark evaluation (COCOEE/MagicBrush)
│   ├── run_inference.py         # Inference manual (single image / batch)
│   ├── download_openimages.py   # Download OpenImages via FiftyOne
│   ├── download_coco.py         # Download COCO dataset
│   └── download_magicbrush.py  # Download MagicBrush dataset
│
├── configs/
│   ├── train_config.yaml        # OpenImages (default / paper replication)
│   ├── train_config_coco.yaml   # COCO fine-tuning
│   ├── train_config_magicbrush.yaml  # MagicBrush fine-tuning
│   └── train_config_mixed.yaml  # Mixed dataset training
│
├── data/                        # Dataset (tidak di-commit ke git)
├── checkpoints/                 # Model checkpoints
├── experiments/                 # Hasil inference & evaluasi
├── requirements.txt
└── SETUP_UBUNTU.md              # Panduan setup environment di Ubuntu
```

---

## Instalasi

### Prasyarat
- Python 3.10+
- CUDA 11.8+ / CUDA 12.x
- GPU VRAM ≥ 16 GB (training), ≥ 8 GB (inference)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/USERNAME/HybridEditDif.git
cd HybridEditDif

# Buat conda environment
conda create -n hybridedif python=3.10
conda activate hybridedif

# Install PyTorch (sesuaikan dengan versi CUDA)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

> Untuk panduan lengkap setup di Ubuntu (termasuk driver NVIDIA, CUDA, dan konfigurasi multi-GPU), lihat [`SETUP_UBUNTU.md`](SETUP_UBUNTU.md).

---

## Dataset

### OpenImages V7 (Paper Replication — 1.9M images)

```bash
# Download semua data (~500 GB) ke HDD eksternal
python scripts/download_openimages.py \
    --n_samples 0 \
    --split train \
    --data_root ./data/openimages \
    --fiftyone_dir /mnt/storage/fiftyone

# Jika cache FiftyOne sudah ada di HDD, gunakan langsung:
python scripts/download_openimages.py \
    --n_samples 0 \
    --split train \
    --fiftyone_dir /mnt/storage/fiftyone
```

### COCO (Untuk fine-tuning / eksperimen cepat)

```bash
python scripts/download_coco.py \
    --output_dir data/coco \
    --n_samples 10000   # 0 = semua
```

### MagicBrush (Untuk fine-tuning)

```bash
python scripts/download_magicbrush.py \
    --output_dir data/magicbrush
```

---

## Training

### 1. Replikasi Paper (OpenImages, 1 epoch, 1.9M images)

```bash
# Setup accelerate untuk multi-GPU
accelerate config

# Training dengan 2 GPU
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    scripts/train.py --config configs/train_config.yaml

# Training dengan 1 GPU
CUDA_VISIBLE_DEVICES=1 python \
    scripts/train.py --config configs/train_config.yaml
```

### 2. Fine-tuning pada COCO

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --config configs/train_config_coco.yaml
```

### 3. Fine-tuning pada MagicBrush

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --config configs/train_config_magicbrush.yaml
```

### Resume dari Checkpoint

Edit `configs/train_config.yaml`:
```yaml
training:
  resume_from_checkpoint: "latest"   # atau path spesifik: "checkpoints/.../checkpoint-15000"
```

Lalu jalankan kembali perintah training yang sama.

### Konfigurasi Penting (`configs/train_config.yaml`)

```yaml
dataset:
  type: "openimages"
  max_samples: null          # null = semua data, atau angka: 100000

training:
  num_epochs: 1
  batch_size: 4              # Per GPU
  gradient_accumulation_steps: 8   # Effective batch = 4 × GPU × 8 = 64
  learning_rate: 1.0e-4
  mixed_precision: "bf16"
  num_workers: 8
```

---

## Inference

### Single Image

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_inference.py \
    --weights checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \
    --source path/to/source.jpg \
    --mask path/to/mask.png \
    --reference path/to/reference.jpg \
    --prompt "replace the object with a similar one" \
    --device cuda:1 \
    --steps 30 \
    --w1 2.0 --w2 2.0
```

### Batch dari Folder

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_inference.py \
    --weights checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \
    --images_dir data/COCOEE/images \
    --masks_dir data/COCOEE/masks \
    --refs_dir data/COCOEE/references \
    --output_dir experiments/inference_cocoee \
    --n_images 20 \
    --steps 30 \
    --w1 2.0 --w2 2.0
```

**Output:** Grid 4-panel per gambar: `[Source | Mask | Reference | Generated]`

> **Catatan Guidance Scale:** Gunakan `w1=w2=2.0`–`3.0` untuk hasil stabil.  
> Nilai tinggi (default paper `w1=w2=7.5`) membutuhkan model yang sudah terlatih penuh.

---

## Evaluasi

### Benchmark COCOEE

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --checkpoint checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \
    --dataset cocoee \
    --data_root ./data/COCOEE \
    --output_dir ./experiments \
    --steps 30 \
    --max_samples 100 \
    --w1 2.0 --w2 2.0
```

### Semua Benchmark Sekaligus

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --checkpoint checkpoints/openimages_paper/final_model/hybrid_edit_dif_weights.pt \
    --dataset all \
    --data_root ./data \
    --output_dir ./experiments
```

### Metrik yang Dihitung

| Metrik | Keterangan | Target |
|--------|-----------|--------|
| **FID** ↓ | Frechet Inception Distance | Semakin kecil semakin baik |
| **SSIM** ↑ | Structural Similarity | Semakin besar semakin baik |
| **LPIPS** ↓ | Perceptual similarity | Semakin kecil semakin baik |
| **CLIP Score** ↑ | Text-image alignment | Semakin besar semakin baik |
| **BG MSE** ↓ | Background preservation | Semakin kecil semakin baik |

### Output Evaluasi

```
experiments/cocoee/
├── generated/          # Hasil generated (00000.jpg, 00001.jpg, ...)
├── real/               # Source images (aligned dengan generated)
├── references/         # Reference images (aligned, nama sama)
├── visualization/      # Grid [source|mask|ref|generated] per sample
└── metrics.json        # Hasil metrik
```

---

## Struktur Checkpoint

```
checkpoints/
├── openimages_paper/
│   ├── checkpoints/
│   │   ├── checkpoint-5000/    # Accelerate checkpoint (resume)
│   │   ├── checkpoint-10000/
│   │   └── checkpoint-15000/
│   └── final_model/
│       └── hybrid_edit_dif_weights.pt   # Weights final untuk inference
├── coco/
│   └── final_model/hybrid_edit_dif_weights.pt
└── magicbrush/
    └── final_model/hybrid_edit_dif_weights.pt
```

Format file `.pt`:
```python
{
    "ddca_layers":            state_dict,   # 16 DDCA attention layers
    "image_encoder_mlp":      state_dict,   # CLIP image MLP head
    "clip_text_encoder_mlp":  state_dict,   # CLIP text MLP head
    "unet_conv_in":           state_dict,   # Modified conv_in (4ch→9ch)
    "global_step":            int,
    "config":                 dict,
}
```

---

## Referensi

```bibtex
@article{liu2026hybridEditDif,
  title   = {HybridEditDif: Hybrid Guided Image Editing via Diffusion Model},
  author  = {Liu et al.},
  journal = {Pattern Recognition},
  year    = {2026},
  doi     = {10.1016/j.patcog.2025.111732}
}
```

**Libraries yang digunakan:**
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) — Base diffusion model
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Vision-Language encoders (ViT-H/14 & ViT-g/14)
- [🤗 Diffusers](https://github.com/huggingface/diffusers) — Diffusion pipeline & scheduler
- [🤗 Accelerate](https://github.com/huggingface/accelerate) — Multi-GPU training
- [FiftyOne](https://voxel51.com/fiftyone/) — OpenImages dataset management

---

## Catatan Reproduksi

| Aspek | Paper Asli | Implementasi Ini |
|-------|-----------|-----------------|
| Dataset | OpenImages 1.9M | ✅ Sama |
| Base Model | Stable Diffusion | ✅ SD v1.5 |
| DDCA Layers | 16 | ✅ 16 |
| Image Encoder | CLIP ViT-H/14 | ✅ OpenCLIP ViT-H/14 |
| Text Encoder | CLIP ViT-g/14 | ✅ OpenCLIP ViT-g/14 |
| Image Size | 512×512 | ✅ 512×512 |
| Optimizer | AdamW, lr=1e-4 | ✅ Sama |
| Training | ~30 epoch, V100 | ~1 epoch, RTX 4080/PRO 4000 |
