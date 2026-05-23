# 🚀 Setup HybridEditDif di Ubuntu + Conda

## Prasyarat
- Ubuntu 20.04 / 22.04 LTS
- GPU NVIDIA dengan VRAM ≥ 16 GB (disarankan V100/A100/RTX 3090+)
- Driver NVIDIA sudah terinstall (`nvidia-smi` bisa jalan)

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
git clone <URL_REPO_KAMU> ~/HybridEditDif
cd ~/HybridEditDif
```

**Opsi B — Transfer dari Windows via SCP:**
```bash
# Jalankan dari PowerShell Windows
scp -r "D:\Documents\TUGAS KULIAH\PROJECT-BATIK\HybridEditDif" user@<IP_SERVER>:~/HybridEditDif
```

**Opsi C — Via USB / shared folder:**
```bash
cp -r /path/ke/HybridEditDif ~/HybridEditDif
cd ~/HybridEditDif
```

---

## STEP 3 — Buat Conda Environment

```bash
# Buat environment baru dengan Python 3.10
conda create -n hybridedif python=3.10 -y

# Aktifkan environment
conda activate hybridedif

# Verifikasi Python
python --version   # harus Python 3.10.x
```

---

## STEP 4 — Install PyTorch dengan CUDA

> ⚠️ Sesuaikan versi CUDA dengan output `nvidia-smi` kamu!

```bash
# Cek versi CUDA driver dulu
nvidia-smi
```

**Untuk CUDA 12.1 (GPU modern — RTX 30xx/40xx, A100):**
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Untuk CUDA 11.8:**
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Verifikasi CUDA bisa dideteksi:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```
Output yang diharapkan: `CUDA: True` dan nama GPU kamu.

---

## STEP 5 — Install System Dependencies

```bash
# Library sistem yang dibutuhkan OpenCV dan pycocotools
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
# Pastikan environment aktif
conda activate hybridedif

# Upgrade pip dulu
pip install --upgrade pip

# ── Core diffusers ecosystem ──────────────────────────────────────────────────
pip install \
    "diffusers==0.27.2" \
    "transformers==4.38.2" \
    "accelerate==0.27.2" \
    "safetensors>=0.4.2"

# ── CLIP / Vision encoders ────────────────────────────────────────────────────
pip install "open_clip_torch>=2.24.0"

# ── Image processing ──────────────────────────────────────────────────────────
pip install \
    "Pillow>=10.0.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "scipy>=1.11.0"

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
pip install "xformers>=0.0.24" || echo "⚠ xformers skip — pakai attention standar"

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

Model Stable Diffusion v1.5 di-download otomatis dari HuggingFace saat pertama kali dijalankan.

```bash
# Login via CLI
huggingface-cli login
# → Masukkan token dari: https://huggingface.co/settings/tokens

# Atau via environment variable (lebih praktis di server)
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

> 📌 Buat token di: **https://huggingface.co/settings/tokens** (pilih tipe: Read)

---

## STEP 8 — Setup Accelerate

```bash
# Setup interaktif (direkomendasikan)
accelerate config
```

Jawab pertanyaannya:
- `compute environment` → **This machine**
- `multi-GPU` → **No** (single GPU) atau **Yes** (multi)
- `num processes` → jumlah GPU kamu (1, 2, 4, dst.)
- `mixed precision` → **fp16**

**Atau langsung buat config untuk single GPU:**
```bash
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
```

---

## STEP 9 — Buat Config Training

```bash
cd ~/HybridEditDif

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
  train_batch_size: 4     # Kurangi ke 2 kalau VRAM < 24GB
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  gradient_accumulation_steps: 4
  mixed_precision: "fp16"
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
cd ~/HybridEditDif
conda activate hybridedif

python - << 'EOF'
import torch, diffusers, transformers, open_clip, accelerate, lpips, omegaconf, cv2
print("=" * 50)
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}", end=" ")
print(f"— {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
print(f"Diffusers    : {diffusers.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"Accelerate   : {accelerate.__version__}")
print(f"OpenCLIP     : {open_clip.__version__}")
print(f"OpenCV       : {cv2.__version__}")
print("=" * 50)
print("✅ Semua dependency siap!")
EOF
```

---

## STEP 11 — Jalankan Training

**Single GPU:**
```bash
cd ~/HybridEditDif
conda activate hybridedif

python scripts/train.py --config configs/train_config.yaml
```

**Multi-GPU (contoh 4 GPU):**
```bash
accelerate launch --num_processes 4 scripts/train.py \
    --config configs/train_config.yaml
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
# (atau http://<IP_SERVER>:6006 dari laptop kamu)
```

---

## STEP 12 — Jalankan Evaluasi

```bash
cd ~/HybridEditDif
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
| `CUDA out of memory` | Kurangi `train_batch_size` ke 2, naikkan `gradient_accumulation_steps` ke 8 |
| `xformers` install gagal | Abaikan saja, tidak wajib |
| `pycocotools` build error | `sudo apt install python3-dev` lalu install ulang |
| `ImportError: src.models` | Pastikan jalankan dari root folder `~/HybridEditDif/` |
| SD model download lambat | Set `HF_TOKEN` dan gunakan `HF_HUB_OFFLINE=1` setelah download pertama |
| `nvidia-smi` tidak ada | Install NVIDIA driver: `sudo ubuntu-drivers install` |

---

## 💾 Struktur Setelah Setup

```
HybridEditDif/
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
