# Anomaly Detection with Flow Matching on BraTS2021

Flow Matching generative model applied to unsupervised brain tumor anomaly detection on BraTS2021 MRI data. The model learns the healthy brain distribution, then detects anomalies by measuring reconstruction error on unseen scans.

## Method

1. Train a conditional flow matching UNet on brain MRI slices (healthy + unhealthy) with classifier-free guidance
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
mkdir -p data
unzip brats2021-preprocessed.zip -d data/
# The zip extracts as data/brats2021-preprocessed/brats2021/ — flatten it:
mv data/brats2021-preprocessed/brats2021 data/brats2021
rm -rf data/brats2021-preprocessed brats2021-preprocessed.zip
# Copy the official split file from the repo into the data directory:
cp preprocessed_split_train_val_test.json data/brats2021/
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

## Pretrained Checkpoint

Download the pretrained BraTS checkpoint (epoch 11) from HuggingFace:

```bash
huggingface-cli download minh2k4/brats-flow-matching-perbatch checkpoint_epoch0011.pth --local-dir ./output_brats
```

## Training

```bash
# Quick smoke test (1 step, no data required)
python train.py --dataset=cifar10 --test_run

# Train on BraTS2021
python train.py \
  --dataset=bratsv2 \
  --data_path=./data/brats2021 \
  --use_preprocessed \
  --batch_size=4 \
  --accum_iter=8 \
  --epochs=50 \
  --lr=1e-4 \
  --lr_scheduler=cosine \
  --precision=bf16 \
  --class_drop_prob=0.15 \
  --use_ema \
  --output_dir=./output_brats

# Resume training
python train.py --resume ./output_brats/checkpoint.pth ...
```

Or use the provided script:

```bash
bash train_brats.sh --v2
```

Key training parameters:

| Parameter | Description | Default |
|---|---|---|
| `--dataset` | Architecture: `brats` (4-level) or `bratsv2` (5-level) | `bratsv2` |
| `--use_preprocessed` | Load from `.npy` slices (required for BraTS) | — |
| `--batch_size` | Batch size per GPU | `4` |
| `--accum_iter` | Gradient accumulation steps (effective batch = batch × accum) | `8` |
| `--precision` | Training precision: `fp32`, `fp16`, `bf16` | `bf16` |
| `--class_drop_prob` | Label dropout probability for classifier-free guidance | `0.15` |
| `--use_ema` | Use EMA weights at evaluation | — |
| `--lr_scheduler` | LR schedule: `constant`, `linear`, `cosine` | `cosine` |

## Interactive Demo (Gradio)

A browser-based demo with step-by-step ODE visualisation:

```bash
bash run_demo.sh
# → http://localhost:7860
```

Features:
- **Folder picker** — select any case from `data/brats2021/unhealthy/` directly, or upload a custom `.npy` file
- **Step-by-step encoding gallery** — every Euler step of the reverse ODE (t = 1 → t_start), shown as a 4-modality strip with a red progress bar
- **Step-by-step decoding gallery** — every Euler step of the forward ODE (t_start → 1), shown with a green progress bar
- **Result grid** — original / reconstructed / anomaly map (jet colormap) / binary mask / GT mask for all 4 modalities + combined row
- **Metrics table** — DICE, IoU, AUROC per modality and combined (shown when a `_seg.npy` GT is available)

Tunable parameters in the UI: `t_start`, `step_size`, `cfg_scale`, `save_every` (snapshot interval).

## Anomaly Detection Inference

```bash
python infer_anomaly.py \
  --checkpoint ./output_brats/checkpoint_epoch0011.pth \
  --arch bratsv2 \
  --data_path ./data/brats2021 \
  --split_file ./data/brats2021/preprocessed_split_train_val_test.json \
  --split test \
  --cfg_scale 0.5 \
  --t 0.2 \
  --step_size 0.02 \
  --combined_modalities T2,FLAIR \
  --num_unhealthy 1000 \
  --num_healthy 1000 \
  --output_dir ./anomaly_results
```

Or use the provided script:

```bash
bash execute_anomaly_detection.sh --v2
```

Key inference parameters:

| Parameter | Description | Best value |
|---|---|---|
| `--t` | Encode endpoint (0=pure noise, 1=clean). Lower → stronger erasure | `0.2` |
| `--cfg_scale` | Classifier-free guidance scale | `0.5` |
| `--step_size` | Euler ODE step size. Smaller → more accurate | `0.02` |
| `--combined_modalities` | Modalities to union for the combined mask | `T2,FLAIR` |
| `--arch` | Model architecture: `brats` or `bratsv2` | `bratsv2` |
| `--split` | Evaluation split: `val` or `test` | `test` |
| `--num_unhealthy` | Number of unhealthy slices to evaluate (`-1` = all) | `-1` |
| `--num_healthy` | Number of healthy slices to evaluate (`-1` = all) | `-1` |
| `--min_component_size` | Drop connected components smaller than this (pixels) | `100` |
| `--border_erosion` | Brain rim thickness for edge-artefact suppression | `3` |

## Hyperparameter Grid Search

Coordinate-descent sweep over `cfg_scale` → `t` → `step_size`:

```bash
# Phase 1: cfg_scale sweep (t=0.2, step=0.02)
SPLIT_FILE=./data/brats2021/preprocessed_split_train_val_test.json \
SPLIT=val \
CHECKPOINT_OVERRIDE=./output_brats/checkpoint_epoch0011.pth \
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
├── train_brats.sh               # Ready-made BraTS training script
├── infer_anomaly.py             # Anomaly detection inference + metrics
├── execute_anomaly_detection.sh # Full evaluation pipeline
├── run_grid.sh                  # Coordinate-descent hyperparameter sweep
├── demo.py                      # Gradio interactive demo
├── run_demo.sh                  # Launch script for the demo
├── create_brats_split.py        # Generate train/val/test splits
├── process_brats.py             # Raw NIfTI → .npy preprocessing
├── preprocessed_split_train_val_test.json  # Official train/val/test split
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
└── datasets/
    └── brats.py                 # BraTS dataset loader
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
