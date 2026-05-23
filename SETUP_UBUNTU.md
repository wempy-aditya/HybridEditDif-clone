# 🚀 Setup HybridEditDif di Ubuntu + Conda

## Spesifikasi Device (if24-desktop)

| Komponen | Detail |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU 0 | NVIDIA GeForce RTX 4080 — 15 GB VRAM (sm_89) |
| GPU 1 | NVIDIA RTX PRO 4000 Blackwell — 23 GB VRAM (sm_120) |
| CUDA Driver | 13.1 (Driver 590.48.01) |
| Python | 3.10 (via Conda) |
| Conda Env | `hybridedif` |
| PyTorch | 2.11.0+cu128 (terinstall aktual) |

> ⚠️ **RTX PRO 4000 Blackwell (sm_120)** membutuhkan **PyTorch ≥ 2.7** dengan **CUDA 12.8+**.
> PyTorch < 2.7 hanya bisa pakai RTX 4080 (sm_89).

---

## STEP 1 — Install Miniconda (kalau belum ada)

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Jalankan installer
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Tambahkan ke PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verifikasi
conda --version
```

---

## STEP 2 — Clone / Transfer Project ke Ubuntu

**Opsi A — Clone dari GitHub:**
```bash
git clone <URL_REPO_KAMU> ~/code/HybridEditDif-clone
cd ~/code/HybridEditDif-clone
```

**Opsi B — Transfer dari Windows via SCP:**
```bash
# Jalankan dari PowerShell Windows
scp -r "D:\Documents\TUGAS KULIAH\PROJECT-BATIK\HybridEditDif" if24@<IP>:~/code/HybridEditDif-clone
```

---

## STEP 3 — Buat Conda Environment

```bash
# Hapus env lama kalau ada
conda deactivate
conda env remove -n hybridedif -y

# Buat environment baru dengan Python 3.10
conda create -n hybridedif python=3.10 -y
conda activate hybridedif
```

---

## STEP 4 — Install PyTorch (Support Dual GPU + Blackwell)

> ✅ **Wajib install via `pip`**, bukan `conda install pytorch` — karena conda PyTorch
> menyebabkan konflik MKL (`iJIT_NotifyEvent` error) di setup ini.
>
> Versi yang terinstall: **PyTorch 2.11.0+cu128** (latest dari index cu128)

```bash
conda activate hybridedif

# PyTorch latest + CUDA 12.8 — support RTX 4080 (sm_89) DAN RTX PRO 4000 Blackwell (sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verifikasi kedua GPU terdeteksi:**
```bash
python -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
print('GPUs    :', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} — {p.total_memory // 1024**3} GB — sm_{p.major}{p.minor}')
"
```

Output yang diharapkan:
```
PyTorch : 2.7.x+cu128
CUDA    : True
GPUs    : 2
  GPU 0: NVIDIA GeForce RTX 4080 — 16 GB — sm_89
  GPU 1: NVIDIA RTX PRO 4000 Blackwell — 24 GB — sm_120
```

---

## STEP 5 — Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-dev \
    git \
    wget \
    curl
```

---

## STEP 6 — Install Python Dependencies

```bash
conda activate hybridedif

# Upgrade pip
pip install --upgrade pip

# ── Core diffusers ecosystem ──────────────────────────────────────────────────
# PENTING: diffusers>=0.30.0 agar kompatibel dengan huggingface_hub terbaru
# (diffusers==0.27.2 menyebabkan: ImportError: cannot import 'cached_download')
pip install \
    "diffusers>=0.30.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.2"

# ── CLIP / Vision encoders ────────────────────────────────────────────────────
pip install "open_clip_torch>=2.24.0"

# ── Image processing ──────────────────────────────────────────────────────────
pip install \
    "Pillow>=10.0.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "scipy>=1.11.0"
# Catatan: NumPy 2.x kompatibel dengan PyTorch 2.11, tidak perlu pin <2

# ── Dataset & data loading ────────────────────────────────────────────────────
pip install \
    "datasets>=2.18.0" \
    "pandas>=2.0.0" \
    "tqdm>=4.66.0" \
    "pycocotools>=2.0.7"

# ── Metrics ───────────────────────────────────────────────────────────────────
pip install \
    "lpips>=0.1.4" \
    "pytorch-fid>=0.3.0" \
    "torch-fidelity>=0.3.0" \
    "scikit-image>=0.21.0" \
    "clean-fid>=0.1.35"

# Install OpenAI CLIP (dari GitHub)
pip install git+https://github.com/openai/CLIP.git

# ── Training utilities ────────────────────────────────────────────────────────
pip install \
    "wandb>=0.16.0" \
    "tensorboard>=2.14.0" \
    "einops>=0.7.0"

# ── xformers (opsional, mempercepat attention) ────────────────────────────────
pip install xformers || echo "⚠ xformers skip — pakai attention standar"

# ── Utilities ─────────────────────────────────────────────────────────────────
pip install \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0" \
    "pyyaml>=6.0" \
    "requests>=2.31.0" \
    "huggingface-hub>=0.21.0"

# ── NIMA (opsional) ───────────────────────────────────────────────────────────
pip install git+https://github.com/yunxiaoshi/neural-image-assessment.git \
    || echo "⚠ NIMA skip — QS metric akan pakai fallback VGG"
```

---

## STEP 7 — Setup Hugging Face Token

```bash
# Login via CLI
huggingface-cli login
# → Masukkan token dari: https://huggingface.co/settings/tokens

# Atau via environment variable
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

> 📌 Buat token di: **https://huggingface.co/settings/tokens** (tipe: Read)

---

## STEP 8 — Setup Accelerate (Dual GPU)

```bash
# Setup interaktif
accelerate config
```

Jawab pertanyaannya untuk **dual GPU**:
- `compute environment` → **This machine**
- `multi-GPU` → **Yes**
- `num processes` → **2** (untuk 2 GPU)
- `mixed precision` → **bf16** (Blackwell support bf16 native)

**Atau buat config langsung untuk 2 GPU:**
```bash
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 0,1
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
use_cpu: false
EOF
```

**Untuk single GPU (RTX 4080 saja):**
```bash
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
gpu_ids: '0'
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
```

---

## STEP 9 — Buat Config Training

```bash
cd ~/code/HybridEditDif-clone

cat > configs/train_config.yaml << 'EOF'
model:
  sd_model_path: "runwayml/stable-diffusion-v1-5"
  image_context_dim: 1024
  text_context_dim: 1024
  lambda1: 1.0
  lambda2: 1.0

data:
  data_root: "./data/openimages"
  image_size: 512
  num_workers: 8
  max_images: 200000      # Kurangi ke 1000 untuk dry run
  max_samples: null

training:
  num_epochs: 30
  # RTX 4080 (16GB): batch 4 aman dengan fp16
  # RTX PRO 4000 Blackwell (24GB): bisa batch 6-8
  train_batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  gradient_accumulation_steps: 4
  mixed_precision: "bf16"   # bf16 lebih stabil di Blackwell
  warmup_steps: 500
  log_every: 50
  save_every: 2000
  eval_every: 5000
  seed: 42
  resume_from_checkpoint: null

output_dir: "./checkpoints"
use_wandb: false
EOF

echo "✓ Config training dibuat!"
```

---

## STEP 10 — Verifikasi Semua Dependency

```bash
cd ~/code/HybridEditDif-clone
conda activate hybridedif

python - << 'EOF'
import torch, numpy as np

print("=" * 55)
print(f"PyTorch      : {torch.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPU count    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} — {p.total_memory // 1024**3}GB — sm_{p.major}{p.minor}")

import diffusers, transformers, accelerate, open_clip, cv2, omegaconf
print(f"Diffusers    : {diffusers.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"Accelerate   : {accelerate.__version__}")
print(f"OpenCLIP     : {open_clip.__version__}")
print(f"OpenCV       : {cv2.__version__}")
print("=" * 55)
print("✅ Semua dependency siap!")
EOF
```

Output yang diharapkan:
```
=======================================================
PyTorch      : 2.11.0+cu128
NumPy        : 2.2.6
CUDA         : True
GPU count    : 2
  GPU 0: NVIDIA GeForce RTX 4080 — 15GB — sm_89
  GPU 1: NVIDIA RTX PRO 4000 Blackwell — 23GB — sm_120
Diffusers    : 0.30.x
Transformers : 4.40.x
Accelerate   : 0.30.x
OpenCLIP     : 2.24.x
OpenCV       : 4.x.x
=======================================================
✅ Semua dependency siap!
```

---

## STEP 11 — Download Dataset OpenImages V7

Dataset yang dipakai paper: **OpenImages V7** — 1.9M gambar, 16M bounding box.
Untuk training awal, cukup download **50.000 gambar** (~50GB).

### Opsi A — Via FiftyOne (Paling Mudah ✅ Direkomendasikan)

```bash
conda activate hybridedif

# Install FiftyOne
pip install fiftyone

# Jalankan script download (50k gambar, ~50GB, bisa makan waktu beberapa jam)
cd ~/code/HybridEditDif-clone
python scripts/download_openimages.py \
    --data_root ./data/openimages \
    --n_samples 50000 \
    --split train
```

> ⏱ Estimasi waktu: **2–6 jam** tergantung koneksi internet.
> 💾 Estimasi disk: **~50 GB** untuk 50k gambar.

**Untuk testing cepat (1000 gambar saja):**
```bash
python scripts/download_openimages.py \
    --data_root ./data/openimages \
    --n_samples 1000 \
    --split train
```

### Opsi B — Via AWS CLI (Lebih Cepat untuk Dataset Besar)

```bash
# Install AWS CLI
sudo apt install awscli -y

# Download images langsung dari S3 (no sign-in required)
mkdir -p ./data/openimages/images/train

# Download subset tertentu (ganti limit sesuai kebutuhan)
aws s3 --no-sign-request sync \
    s3://open-images-dataset/train \
    ./data/openimages/images/train \
    --quiet

# Download annotations CSV
python scripts/download_openimages.py \
    --data_root ./data/openimages \
    --method csv
```

### Verifikasi Dataset

```bash
python - << 'EOF'
from pathlib import Path
import json

data_root = Path("./data/openimages")
images_dir = data_root / "images" / "train"
ann_dir    = data_root / "annotations"

n_images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
print(f"✓ Gambar tersedia : {n_images:,}")

bbox_file = ann_dir / "train_bbox_annotations.json"
if bbox_file.exists():
    with open(bbox_file) as f:
        bboxes = json.load(f)
    print(f"✓ BBox annotations: {len(bboxes):,} gambar")
else:
    print("⚠ BBox annotations belum ada")

text_file = ann_dir / "text_annotations.json"
if text_file.exists():
    with open(text_file) as f:
        texts = json.load(f)
    print(f"✓ Text annotations: {len(texts):,} gambar")
else:
    print("⚠ Text annotations belum ada (opsional)")
EOF
```

### Update Config Training Sesuai Dataset

```bash
# Sesuaikan data_root dan max_images di config
# Kalau download 50k gambar, set max_images: 50000
sed -i 's/max_images: 200000/max_images: 50000/' configs/train_config.yaml

# Verifikasi
grep max_images configs/train_config.yaml
```

---

## STEP 12 — Jalankan Training

**Single GPU (RTX 4080):**
```bash
cd ~/code/HybridEditDif-clone
conda activate hybridedif

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/train_config.yaml
```

**Dual GPU (RTX 4080 + RTX PRO 4000 Blackwell):**
```bash
accelerate launch --num_processes 2 scripts/train.py \
    --config configs/train_config.yaml
```

**Pakai GPU Blackwell saja (24GB VRAM, batch lebih besar):**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/train_config.yaml
```

**Monitor GPU real-time:**
```bash
# Buka terminal baru
watch -n 1 nvidia-smi
```

**Lihat log TensorBoard:**
```bash
tensorboard --logdir ./checkpoints/logs --port 6006
# Buka browser: http://localhost:6006
```

---

## STEP 12 — Jalankan Evaluasi

```bash
cd ~/code/HybridEditDif-clone
conda activate hybridedif

python scripts/evaluate.py \
    --checkpoint checkpoints/final_model/hybrid_edit_dif_weights.pt \
    --dataset cocoee \
    --data_root ./data \
    --output_dir ./experiments \
    --steps 50 \
    --w1 7.5 \
    --w2 7.5
```

---

## 🔧 Troubleshooting

| Error | Solusi |
|---|---|
| `iJIT_NotifyEvent` / MKL error | **Jangan** `conda install pytorch` — wajib via `pip install torch --index-url .../cu128` |
| `cannot import 'cached_download'` | `pip install "diffusers>=0.30.0"` — diffusers lama tidak kompatibel dengan huggingface_hub baru |
| RTX PRO 4000 Blackwell not supported | Install PyTorch via `--index-url https://download.pytorch.org/whl/cu128` |
| `CUDA out of memory` | Kurangi `train_batch_size` ke 2, naikkan `gradient_accumulation_steps` ke 8 |
| `xformers` install gagal | Abaikan, tidak wajib |
| `pycocotools` build error | `sudo apt install python3-dev` lalu install ulang |
| `ImportError: src.models` | Jalankan dari root: `cd ~/code/HybridEditDif-clone` |
| SD model download lambat | `huggingface-cli login` lalu set `HF_HUB_OFFLINE=1` setelah download pertama |

---

## 💾 Struktur Folder

```
~/code/HybridEditDif-clone/
├── configs/
│   └── train_config.yaml    ← ✅ Dibuat di Step 9
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── models/  (attention, encoders, hybrid_edit_dif, inference)
│   ├── data/    (openimages_dataset, mask_augmentation)
│   └── utils/   (metrics)
├── data/
│   └── openimages/          ← Download otomatis saat training pertama
├── checkpoints/             ← Checkpoint tersimpan di sini
├── experiments/             ← Hasil evaluasi
└── requirements.txt
```
