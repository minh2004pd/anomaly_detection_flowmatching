# Anomaly Detection with Flow Matching on BraTS2021

Flow Matching generative model applied to unsupervised brain tumor anomaly detection on BraTS2021 MRI data. The model learns the healthy brain distribution, then detects anomalies by measuring reconstruction error on unseen scans.

## Method

1. Train a conditional flow matching UNet on **healthy** brain MRI slices (label=0) with classifier-free guidance
2. At inference, partially encode a test scan to timestep `t` (data → noisy latent), then decode back toward the healthy distribution using CFG
3. The pixel-wise reconstruction difference (MAD) is thresholded with Otsu + post-processing to produce a tumor mask
4. Evaluated with DICE, IoU, and AUROC on the BraTS2021 test split

## Setup

```bash
git clone https://github.com/minh2004pd/anomaly_detection_flowmatching_medical.git
cd anomaly_detection_flowmatching_medical
pip install -r requirements.txt
```

## Data Preparation

### Option 1 — Download preprocessed data from Kaggle (recommended)

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download minhdon/brats2021-preprocessed
unzip brats2021-preprocessed.zip -d data/
```

The dataset contains:
- `data/brats2021/healthy/` — healthy brain .npy slices, shape `(4, 256, 256)`, modalities `[T1, T1ce, T2, FLAIR]`
- `data/brats2021/unhealthy/` — tumor brain .npy slices + `_seg.npy` ground truth masks
- `data/brats2021/preprocessed_split_train_val_test.json` — case-level 80/10/10 train/val/test split (included in this repo)

### Option 2 — Preprocess from raw BraTS2021 NIfTI

```bash
python process_brats.py --data_dir /path/to/BraTS2021_Training_Data --output_dir ./data/brats2021
python create_brats_split.py --data_path ./data/brats2021
```

## Training

```bash
# Quick smoke test (1 step, no data required)
python train.py --dataset=cifar10 --test_run

# Train on BraTS2021 — healthy + unhealthy slices, class-conditional
python train.py \
  --dataset=brats \
  --data_path=./data/brats2021 \
  --split_file=./data/brats2021/preprocessed_split_train_val_test.json \
  --batch_size=16 \
  --epochs=200 \
  --use_ema \
  --output_dir=./output_brats

# Resume training
python train.py --resume ./output_brats/checkpoint.pth ...
```

Use `train_brats.sh` for a ready-made training script:

```bash
bash train_brats.sh
```

## Anomaly Detection Inference

```bash
python infer_anomaly.py \
  --checkpoint ./output_brats/checkpoint.pth \
  --arch bratsv2 \
  --data_path ./data/brats2021 \
  --split_file ./data/brats2021/preprocessed_split_train_val_test.json \
  --split test \
  --cfg_scale 0.5 \
  --t 0.2 \
  --step_size 0.02 \
  --combined_modalities T2,FLAIR \
  --output_dir ./anomaly_results
```

Key parameters:

| Parameter | Description | Best value |
|---|---|---|
| `--t` | Encode endpoint (0=pure noise, 1=clean). Lower → stronger erasure | 0.2 |
| `--cfg_scale` | Classifier-free guidance scale | 0.5 |
| `--step_size` | Euler ODE step size. Smaller → more accurate | 0.02 |
| `--combined_modalities` | Modalities to union for the combined mask | T2,FLAIR |
| `--arch` | Model architecture: `brats` or `bratsv2` | bratsv2 |

Use `execute_anomaly_detection.sh` for a full evaluation run:

```bash
bash execute_anomaly_detection.sh
```

## Hyperparameter Grid Search

Coordinate-descent sweep over `cfg_scale` → `t` → `step_size`:

```bash
# Phase 1: cfg_scale sweep (t=0.2, step=0.02)
SPLIT_FILE=./data/brats2021/preprocessed_split_train_val_test.json \
SPLIT=val \
CHECKPOINT_OVERRIDE=./output_brats/checkpoint.pth \
NUM_UNHEALTHY=1000 NUM_HEALTHY=1000 \
bash run_grid.sh auto --v2

# Or run phases manually:
bash run_grid.sh 1 --v2                  # Phase 1: cfg sweep
bash run_grid.sh 2 0.5 --v2             # Phase 2: t sweep (best cfg=0.5)
bash run_grid.sh 3 0.2 0.5 --v2         # Phase 3: step sweep (best t=0.2, cfg=0.5)
```

Results are uploaded to HuggingFace automatically if `HF_RESULTS_REPO` and `HF_TOKEN` are set in `.env`.

## Data Split

`preprocessed_split_train_val_test.json` — case-level 80/10/10 split (seed=42):

| Split | Cases | Slices (healthy + unhealthy) |
|-------|-------|------------------------------|
| train | ~800  | ~49,000 |
| val   | ~125  | ~6,125  |
| test  | ~126  | ~6,174  |

Generate a new split:

```bash
python create_brats_split.py \
  --data_path ./data/brats2021 \
  --train_ratio 0.8 \
  --seed 42
```

## Project Structure

```
├── train.py                     # Training script
├── infer_anomaly.py             # Anomaly detection inference + metrics
├── run_grid.sh                  # Coordinate-descent hyperparameter sweep
├── execute_anomaly_detection.sh # Full evaluation pipeline
├── create_brats_split.py        # Generate train/val/test splits
├── process_brats.py             # Raw NIfTI → .npy preprocessing
├── flow_matching/               # Core flow matching library
│   ├── path/                    # Probability paths (OT, Affine, Geodesic)
│   ├── loss/                    # Flow matching loss
│   └── solver/                  # ODE solvers (dopri5, Euler)
├── models/
│   ├── unet.py                  # UNet velocity field model
│   └── model_configs.py         # Dataset-specific configs (brats, bratsv2)
├── training/
│   ├── train_loop.py            # Training loop
│   └── classifier_guidance.py   # CFG implementation
└── preprocessed_split_train_val_test.json  # Official train/val/test split
```

## Results

Evaluated on BraTS2021 test split (126 cases, ~6174 slices), checkpoint epoch 11, arch `bratsv2`:

| Method | DICE | IoU | AUROC |
|--------|------|-----|-------|
| Flow Matching (combined T2+FLAIR) | **0.640** | **0.521** | 0.861 |
| Diffusion (DDIM, combined 4-mod) | 0.626 | 0.491 | **0.905** |

## Acknowledgements

- [Flow Matching](https://github.com/facebookresearch/flow_matching) — Meta AI Research
- BraTS2021 dataset — RSNA-MICCAI Brain Tumor Radiogenomic Classification Challenge
