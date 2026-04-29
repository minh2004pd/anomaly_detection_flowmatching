# Anomaly Detection with Flow Matching on BraTS2021

Flow Matching generative model applied to brain tumor anomaly detection on BraTS2021 MRI data. The model is trained on healthy brain scans, then used to reconstruct diseased scans — the reconstruction error localizes tumor regions.

## Method

- Train a conditional flow matching UNet on **healthy** brain MRI slices (class 0)
- At inference, partially noise a diseased scan to timestep `t`, then denoise toward the healthy distribution using classifier-free guidance
- The pixel-wise reconstruction difference (MAD) is thresholded with Otsu to produce a tumor mask
- Evaluated with DICE, IoU, and AUROC

## Setup

```bash
git clone https://github.com/minh2004pd/anomaly_detection_flowmatching.git
cd anomaly_detection_flowmatching
pip install -r requirements.txt
```

## Data Preparation

### Option 1 — Download preprocessed data from Kaggle (recommended)

```bash
pip install kaggle
kaggle datasets download minhdon/brats2021-preprocessed
unzip brats2021-preprocessed.zip -d data/
```

The dataset contains:
- `data/brats2021/healthy/` — healthy brain .npy slices (4 modalities, 256×256)
- `data/brats2021/unhealthy/` — tumor brain .npy slices + ground truth masks
- `data/brats2021/preprocessed_split.json` — case-level 80/20 train/val split

### Option 2 — Preprocess from raw BraTS2021 NIfTI

Download the raw dataset from [Kaggle BraTS2021](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification), then:

```bash
python preprocess_brats.py --data_dir /path/to/BraTS2021_Training_Data --output_dir ./data/brats2021
```

Each `.npy` slice has shape `(4, 256, 256)` — channels are `[FLAIR, T1, T1ce, T2]`, normalized to `[0, 1]`.

## Training

```bash
# Quick validation (1 step, no data required)
python train.py --dataset=cifar10 --test_run

# Train on BraTS2021 (healthy only — unsupervised anomaly detection)
python train.py \
  --dataset=brats \
  --data_path=./data/brats2021 \
  --healthy_only \
  --batch_size=16 \
  --epochs=200 \
  --output_dir=./output_brats

# With EMA (recommended for best results)
python train.py \
  --dataset=brats \
  --data_path=./data/brats2021 \
  --healthy_only \
  --batch_size=16 \
  --epochs=200 \
  --use_ema \
  --output_dir=./output_brats
```

Checkpoints and logs are saved to `--output_dir`. Resume training with `--resume ./output_brats/checkpoint.pth`.

## Inference & Anomaly Detection

```bash
python infer_anomaly.py \
  --checkpoint ./output_brats/checkpoint.pth \
  --data_path ./data/brats2021 \
  --cfg_scale 8.0 \
  --t 0.8 \
  --step_size 0.02
```

Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `--t` | Noise level (0=clean, 1=pure noise). Higher → more reconstruction, slower | 0.8 |
| `--cfg_scale` | Classifier-free guidance strength. Higher → stronger healthy prior | 8.0 |
| `--step_size` | ODE integration step size. Smaller → more accurate, slower | 0.02 |

Output: per-case DICE, IoU scores + visualization grid saved to `--output_dir`.

## Hyperparameter Search

Run the 3-phase sweep to find optimal `t`, `cfg_scale`, `step_size`:

```bash
python run_experiments.py 1                    # Phase 1: sweep t
python run_experiments.py 2 0.6               # Phase 2: sweep cfg_scale (best_t=0.6)
python run_experiments.py 3 0.6 20.0          # Phase 3: sweep step_size
```

## Project Structure

```
flow-matching-main/
├── train.py                  # Main training script
├── infer_anomaly.py          # Anomaly detection inference + metrics
├── run_experiments.py        # Hyperparameter sweep runner
├── preprocess_brats.py       # Raw NIfTI → .npy preprocessing
├── flow_matching/            # Core flow matching library
│   ├── path/                 # Probability paths (OT, Affine, Geodesic)
│   ├── loss/                 # Flow matching loss
│   └── solver/               # ODE solvers (dopri5, Euler)
├── models/
│   ├── unet.py               # UNet velocity field model
│   └── model_configs.py      # Dataset-specific configs
├── training/
│   ├── train_loop.py         # Training loop
│   └── eval_loop.py          # Evaluation + FID
└── data/brats2021/
    ├── healthy/              # Healthy slices (train + val)
    ├── unhealthy/            # Tumor slices (val only)
    └── preprocessed_split.json
```

## Acknowledgements

- [Flow Matching](https://github.com/facebookresearch/flow_matching) — Meta AI
- [Guided Diffusion](https://github.com/openai/guided-diffusion) — OpenAI
- BraTS2021 dataset — RSNA-MICCAI Brain Tumor Challenge

## License

CC-BY-NC. UNet model and distributed computing code are under MIT license.
