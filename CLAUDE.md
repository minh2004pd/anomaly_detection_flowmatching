# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
# or with uv (pins CUDA 12.8 wheels):
uv sync
```

Environment variables used at runtime: `HF_REPO`, `HF_TOKEN` (auto-upload checkpoints to HuggingFace), `HF_RESULTS_REPO` (auto-upload grid search results).

## Commands

### Training

```bash
# Quick smoke test (1 step, no data required)
python train.py --dataset=cifar10 --test_run

# BraTS2021 (recommended settings)
python train.py \
  --dataset=bratsv2 \
  --data_path=./data/brats2021 \
  --use_preprocessed \
  --batch_size=4 --accum_iter=8 \
  --epochs=50 --lr=1e-4 \
  --lr_scheduler=cosine \
  --precision=bf16 \
  --class_drop_prob=0.15 \
  --use_ema \
  --output_dir=./output_brats

# Resume
python train.py --resume ./output_brats/checkpoint.pth ...

# Convenience script
bash train_brats.sh --v2

# SLURM
python submitit_train.py --nodes=8 --ngpus=8 --dataset=cifar10

# Eval-only (generates samples + optional FID, no training)
python train.py --eval_only --resume ./output_dir/checkpoint.pth --compute_fid
```

Key training flags: `--precision {fp32,fp16,bf16}`, `--lr_scheduler {constant,linear,cosine}`, `--warmup_epochs`, `--healthy_only` (unconditional, inpaints tumors), `--discrete_flow_matching`.

### Anomaly Detection Inference

```bash
python infer_anomaly.py \
  --checkpoint ./output_brats/checkpoint_epoch0011.pth \
  --arch bratsv2 \
  --data_path ./data/brats2021 \
  --split_file ./data/brats2021/preprocessed_split_train_val_test.json \
  --split test \
  --cfg_scale 0.5 --t 0.2 --step_size 0.02 \
  --combined_modalities T2,FLAIR \
  --num_unhealthy 1000 --num_healthy 1000 \
  --output_dir ./anomaly_results

# Convenience script
bash execute_anomaly_detection.sh --v2
```

Outputs: `metrics.csv`, `summary.txt`, per-sample PNG grids in `--output_dir`.

### Hyperparameter Grid Search

Coordinate-descent sweep over `cfg_scale → t → step_size` on the val split:

```bash
# Fully automated (reads best from each phase)
SPLIT_FILE=./data/brats2021/preprocessed_split_train_val_test.json \
SPLIT=val CHECKPOINT_OVERRIDE=./output_brats/checkpoint_epoch0011.pth \
NUM_UNHEALTHY=1000 NUM_HEALTHY=1000 \
bash run_grid.sh auto --v2

# Manual phases
bash run_grid.sh 1 --v2                  # Phase 1: cfg_scale sweep
bash run_grid.sh 2 0.5 --v2             # Phase 2: t sweep (best cfg=0.5)
bash run_grid.sh 3 0.2 0.5 --v2         # Phase 3: step_size sweep
```

### Data Preparation

```bash
# Preprocess raw NIfTI
python process_brats.py --data_dir /path/to/BraTS2021_Training_Data --output_dir ./data/brats2021
python create_brats_split.py --data_path ./data/brats2021 --train_ratio 0.8 --seed 42

# Or download preprocessed from Kaggle (see README)
```

## Architecture

### Flow Matching Library (`flow_matching/`)

Custom implementation of the flow matching framework:

- **`path/`** — probabilistic interpolation paths: `CondOTProbPath` (conditional optimal transport, default for BraTS), `AffineProbPath`, `GeodesicProbPath` (sphere/torus manifolds), `MixtureDiscreteProbPath` (categorical tokens). `CondOTScheduler` defines `alpha_t=t, sigma_t=1−t`, so `x_t = (1−t)·noise + t·data`.
- **`loss/`** — `GeneralizedFlowMatchingLoss`: dispatches to MSE (continuous) or cross-entropy (discrete) based on path type.
- **`solver/`** — ODE integration: `ODESolver` (wraps `torchdiffeq`; dopri5/Euler/midpoint), `DiscreteSolver` (Euler + Gumbel-max), `RiemannianODESolver`.
- **`utils/`** — manifold definitions (Sphere, Torus), `ModelWrapper` (adapts UNet signature for solver), samplers.

### Models (`models/`)

- **`UNetModel`** (`unet.py`) — continuous flow model; outputs velocity field `v(x_t, t, label)`.
- **`DiscreteUNetModel`** (`discrete_unet.py`) — outputs logits over vocab_size=257 tokens.
- **`MODEL_CONFIGS`** (`model_configs.py`) — all architecture hyperparameters keyed by dataset name. `instantiate_model(architecture, is_discrete, use_ema)` is the single entry point. Key configs: `brats` (4-level UNet, `channel_mult=[1,2,4,4]`), `bratsv2` (5-level, adds 16×16 bottleneck stage with attention, recommended), `brats_healthy` (unconditional, `num_classes=None`).
- **`EMA`** (`ema.py`) — wraps any model; `use_ema=True` passes EMA weights at eval time.

All models accept `(x, t, extra={"label": ...})` where `t ∈ [0,1]` (0=noise, 1=data) and `extra` is empty for unconditional.

### Training Infrastructure (`training/`)

`train_loop.py` core loop: sample `t ~ U[0,1]` → compute `x_t` via prob path → forward `(x_t, t, label)` → flow matching loss → mixed-precision backward with AdamW.

- `classifier_guidance.py` — `CFGScaledModel` wrapper: interpolates conditioned vs. unconditioned predictions at inference with `cfg_scale`. Label dropout during training (controlled by `--class_drop_prob`) enables CFG at inference.
- `eval_loop.py` — generates samples, optionally computes FID via `torchmetrics[image]`. `CFGScaledModel` is also imported by `infer_anomaly.py`.
- `distributed_mode.py` — DDP setup; `submitit_train.py` handles SLURM submission.
- `load_and_save.py` — checkpoint format: `{model, optimizer, lr_scheduler, loss_scaler, epoch}`. Always writes `args.json` alongside weights (required by `infer_anomaly.py` to reconstruct the model arch).
- `edm_time_discretization.py` — EDM-paper Heun2 integrator and skewed log-normal timestep schedule.

### BraTS Anomaly Detection Pipeline (`infer_anomaly.py`)

1. **Encode**: reverse ODE `t=1 → t_start` with unconditional velocity (class-agnostic; `--encode_label` can optionally condition this).
2. **Decode**: forward ODE `t_start → t=1` with CFG toward healthy label (class=0).
3. **Anomaly map**: `|input − recon|` per modality, brain-masked, Otsu-thresholded (`--hysteresis` for alternative).
4. **Post-processing**: remove small connected components (`--min_component_size`), suppress brain-border artefacts (`--border_erosion`, `--border_overlap_thr`).
5. **Combined mask**: union of per-modality binary masks over `--combined_modalities` (best: `T2,FLAIR`).
6. **Metrics**: DICE/IoU/AUROC for unhealthy slices; PSNR/SSIM for healthy slices.

`--cfg_zero_star` enables CFG-Zero* (optimized scale projection + zero-init steps). `--encode_label 1 --encode_cfg_scale >0` pushes the latent deeper into the unhealthy manifold before decoding.

## Key Conventions

- **Timestep direction**: `t=0` is pure noise, `t=1` is clean data. Sampling integrates `t=0 → t=1`. In `infer_anomaly.py`, `--t` is the encode endpoint: smaller `--t` ⇒ closer to pure noise ⇒ stronger structural erasure.
- **`args.json` dependency**: `infer_anomaly.py` reads `args.json` from the checkpoint directory to reconstruct the model architecture. Keep `args.json` alongside every checkpoint.
- **Discrete models**: `--discrete_flow_matching`; vocab_size fixed at 257 (256 pixel values + 1 mask token); requires `MixtureDiscreteProbPath` and `DiscreteUNetModel`.
- **Dataset key doubles as arch key**: the `--dataset` argument passed to `train.py` is the key into `MODEL_CONFIGS`; `--arch` in `infer_anomaly.py` overrides the dataset key read from `args.json`.
- **Output directory**: each run writes `checkpoint.pth`, `checkpoint-<epoch>.pth`, `args.json`, `log.txt`, and `snapshots/`.
- **W&B logging**: opt-in with `--wandb --wandb_project <name>`.
