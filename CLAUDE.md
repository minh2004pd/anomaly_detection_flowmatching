# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training

```bash
# Quick validation (one train step + one eval step)
python train.py --dataset=cifar10 --test_run

# Local single-GPU training
python train.py --dataset=cifar10 --batch_size=64 --epochs=20 --output_dir=./output_cifar10

# With EMA + EDM schedule (matches published CIFAR-10 results)
python train.py --dataset=cifar10 --use_ema --edm_schedule --skewed_timesteps --epochs=1800

# BraTS medical imaging (conditional on healthy/unhealthy)
python train.py --dataset=brats --data_path=./data/brats2021 --healthy_only

# Discrete flow matching (categorical tokens, vocab_size=257)
python train.py --dataset=cifar10 --discrete_flow_matching --epochs=2500

# SLURM cluster submission
python submitit_train.py --nodes=8 --ngpus=8 --dataset=cifar10
```

### Inference

```bash
# Generate images from a checkpoint
python infer.py --checkpoint ./output_mnist/checkpoint.pth --labels 0 1 2 3 4 5 6 7 8 9
python infer.py --checkpoint ./output_mnist/checkpoint.pth --labels 7 --num_per_label 16
python infer.py --checkpoint ./output_mnist/checkpoint.pth --labels 3 5 --cfg_scale 4.0

# BraTS anomaly detection (tumor segmentation via healthy→unhealthy flow reversal)
python infer_anomaly.py --checkpoint ./output_brats/checkpoint.pth \
  --data_path ./data/brats2021 --cfg_scale 8.0 --t 0.8 --step_size 0.02

# Evaluate a checkpoint (FID + sample snapshots, no training)
python train.py --eval_only --resume ./output_dir/checkpoint-899.pth --compute_fid
```

### Hyperparameter Sweeps (BraTS anomaly detection)

```bash
python run_experiments.py 1              # Phase 1: optimize interpolation t
python run_experiments.py 2 0.6          # Phase 2: optimize cfg_scale given best_t
python run_experiments.py 3 0.6 20.0     # Phase 3: optimize step_size given best_t & best_cfg
```

## Architecture

### Core Library (`flow_matching/`)

The `flow_matching/` package implements the generative modeling primitives:

- **`path/`** — probabilistic interpolation paths between noise and data:
  - `CondOTProbPath` (conditional optimal transport), `AffineProbPath`, `GeodesicProbPath` (sphere/torus manifolds), `MixtureDiscreteProbPath` (for discrete tokens)
  - `scheduler/` — time discretization schedules for discrete flows
- **`loss/`** — `GeneralizedFlowMatchingLoss`, dispatches to continuous MSE or discrete cross-entropy based on path type
- **`solver/`** — ODE integration: `ODESolver` (wraps `torchdiffeq` dopri5/RK45), `DiscreteSolver` (Euler + Gumbel-max sampling), `RiemannianODESolver`
- **`utils/`** — manifold definitions (Sphere, Torus), samplers, model wrappers

### Models (`models/`)

- **`UNetModel`** (`unet.py`) — continuous flow model; outputs velocity field
- **`DiscreteUNetModel`** (`discrete_unet.py`) — categorical prediction model; outputs logits over 257 tokens
- **`MODEL_CONFIGS`** (`model_configs.py`) — dataset-specific hyperparameters keyed by dataset name; instantiate models through this dict
- **`EMAModel`** (`ema.py`) — wraps any model to maintain an exponential moving average of weights
- **`ClassifierModel`** (`classifier.py`) — auxiliary classifier for guided generation

All models accept a continuous timestep `t ∈ [0, 1]` (0 = data, 1 = noise) and an optional class label.

### Training Infrastructure (`training/`)

The training loop in `train_loop.py` follows:
1. Sample `t ~ U[0,1]` (or EDM log-normal skew schedule)
2. Compute interpolated `x_t` via the chosen probabilistic path
3. Forward pass with `(x_t, t, label)` conditioning
4. Flow matching loss (MSE for continuous, cross-entropy for discrete)
5. Mixed-precision backward + AdamW optimizer + optional LR scheduler

Key modules:
- `eval_loop.py` — generates samples and optionally computes FID via `torchmetrics`
- `classifier_guidance.py` — classifier-free guidance: trains with random label dropout (`class_drop_prob`), inference interpolates conditioned vs. unconditioned predictions
- `distributed_mode.py` — DDP setup; `submitit_train.py` handles SLURM job submission
- `load_and_save.py` — checkpoint I/O; saves `args.json` alongside weights for reproducibility
- `edm_time_discretization.py` — EDM-paper Heun2 integrator and time schedules

### Data Flow for BraTS Anomaly Detection

The anomaly detection pipeline (`infer_anomaly.py`) trains a model conditioned on healthy (class 0) vs. unhealthy (class 1) brain MRI scans. At inference, a diseased image is partially noised to timestep `t` then denoised back toward the healthy distribution using negative classifier-free guidance. The reconstruction difference (MAD) is thresholded with Otsu to produce a tumor segmentation mask; DICE and IoU are reported.

## Key Conventions

- **Timestep direction**: `t=0` is clean data, `t=1` is pure noise — opposite of some diffusion paper conventions.
- **Checkpoint format**: dict with keys `model`, `optimizer`, `lr_scheduler`, `loss_scaler`, `epoch`. Resume with `--resume <path>`.
- **Discrete models**: use `--discrete_flow_matching`; vocab size is always 257 (256 pixel values + 1 mask token); requires `MixtureDiscreteProbPath` and `DiscreteUNetModel`.
- **FID computation**: requires `--compute_fid`; uses `torchmetrics[image]` and generates 50k samples against the training set.
- **Output directory**: each run writes `checkpoint.pth`, `checkpoint-<epoch>.pth`, `args.json`, `log.txt`, and `snapshots/` to `--output_dir`.
