#!/usr/bin/env python3
"""
Anomaly Detection Inference for BraTS2021.

Pipeline:
  1. Reverse ODE  : encode image from t=1 → t_start  (unconditional velocity)
  2. Forward ODE  : reconstruct with CFG label=0,    t_start → t=1
  3. Anomaly map  : |input - recon| per modality, Otsu only inside brain region
  4. Combined     : max-diff across 4 modalities → Otsu → final binary mask
  5. Unhealthy    : DICE / IOU / AUROC
     Healthy      : PSNR / SSIM
  6. Visualisation: 5 rows (T1 T1ce T2 FLAIR Combined) × 5 cols
                    (Original | Reconstruction | Anomaly-jet | Binary | GT Mask)
                    Metrics annotated on each row.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from flow_matching.solver.ode_solver import ODESolver
from models.model_configs import instantiate_model
from training.eval_loop import CFGScaledModel

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="BraTS2021 anomaly detection with reverse+forward ODE")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--data_path",     type=str, required=True,
                   help="BraTS2021 root containing healthy/ unhealthy/ and preprocessed_split.json")
    p.add_argument("--split_file",    type=str, default=None,
                   help="Path to preprocessed_split.json. "
                        "Defaults to <data_path>/preprocessed_split.json")
    p.add_argument("--t",             type=float, default=0.6,
                   help="Encoding depth: fraction along ODE from data (1) toward noise (0).")
    p.add_argument("--step_size",     type=float, default=0.02)
    p.add_argument("--cfg_scale",     type=float, default=3.0)
    p.add_argument("--num_unhealthy", type=int,   default=50)
    p.add_argument("--num_healthy",   type=int,   default=20)
    p.add_argument("--output_dir",    type=str,   default="anomaly_results")
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ─── Data loading from .npy ───────────────────────────────────────────────────

def load_sample(data_path: str, entry: dict):
    """
    Load one val entry from preprocessed_split.json.

    Returns:
        image_np  : (4, 256, 256) float32 [0,1]  — from .npy
        label     : int  0=healthy  1=unhealthy
        mask_np   : (1, 256, 256) float32         — from _seg.npy (zeros if missing)
    """
    rel_path = entry["path"]
    label    = int(entry["label"])

    image_np = np.load(os.path.join(data_path, rel_path)).astype(np.float32)

    seg_path = os.path.join(data_path, rel_path.replace(".npy", "_seg.npy"))
    if os.path.exists(seg_path):
        mask_np = np.load(seg_path).astype(np.float32)
    else:
        mask_np = np.zeros((1, 256, 256), dtype=np.float32)

    return image_np, label, mask_np          # (4,H,W), int, (1,H,W)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    checkpoint_path = Path(checkpoint_path)
    args_path = checkpoint_path.parent / "args.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_path}")

    with open(args_path) as f:
        train_args = json.load(f)

    arch = "brats_healthy" if train_args.get("healthy_only", False) else train_args["dataset"]
    model = instantiate_model(
        architechture=arch,
        is_discrete=train_args.get("discrete_flow_matching", False),
        use_ema=train_args.get("use_ema", False),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded [{arch}] epoch={ckpt.get('epoch', '?')}")

    cfg_model = CFGScaledModel(model=model)
    cfg_model.to(device).eval()
    return cfg_model, train_args


# ─── Velocity wrappers ────────────────────────────────────────────────────────

class _UncondVelocity(nn.Module):
    """Unconditional velocity (cfg_scale=0) for the reverse-ODE encoding step."""
    def __init__(self, cfg_model: CFGScaledModel):
        super().__init__()
        self.cfg = cfg_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        dummy_label = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.cfg(x=x, t=t, cfg_scale=0.0, label=dummy_label)


# ─── ODE encode / decode ──────────────────────────────────────────────────────

@torch.no_grad()
def encode(cfg_model: CFGScaledModel,
           x_image: torch.Tensor,
           t_start: float,
           step_size: float,
           device: str) -> torch.Tensor:
    """
    Reverse ODE: data (t=1) → latent at t_start.
    Uses unconditional velocity so encoding is class-neutral.
    Returns x_t in [-1, 1].
    """
    x = x_image.to(device)
    x_scaled = x * 2.0 - 1.0                          # [0,1] → [-1,1]
    wrapper = _UncondVelocity(cfg_model)
    solver  = ODESolver(velocity_model=wrapper)
    # descending time_grid → torchdiffeq integrates backward
    time_grid = torch.tensor([1.0, t_start], device=device)
    x_t = solver.sample(x_init=x_scaled, time_grid=time_grid,
                        step_size=step_size, method="euler")
    return x_t                                         # (B,4,H,W) in [-1,1]


@torch.no_grad()
def decode(cfg_model: CFGScaledModel,
           x_t: torch.Tensor,
           t_start: float,
           step_size: float,
           device: str,
           cfg_scale: float) -> torch.Tensor:
    """
    Forward ODE: latent at t_start → healthy reconstruction (t=1).
    Uses CFG with label=0 (healthy).
    Returns x_recon in [0, 1].
    """
    x_t = x_t.to(device)
    solver  = ODESolver(velocity_model=cfg_model)
    time_grid = torch.tensor([t_start, 1.0], device=device)
    healthy_label = torch.zeros(x_t.shape[0], dtype=torch.long, device=device)
    x_recon_scaled = solver.sample(
        x_init=x_t,
        time_grid=time_grid,
        step_size=step_size,
        method="euler",
        label=healthy_label,
        cfg_scale=cfg_scale,
    )
    return torch.clamp(x_recon_scaled * 0.5 + 0.5, 0.0, 1.0)   # (B,4,H,W)


# ─── Brain mask ───────────────────────────────────────────────────────────────

def brain_mask(image_np: np.ndarray) -> np.ndarray:
    """(H,W) bool: True where at least one modality is non-zero."""
    return np.any(image_np > 1e-4, axis=0)


# ─── Anomaly analysis ─────────────────────────────────────────────────────────

def _otsu_on_region(diff_2d: np.ndarray, region: np.ndarray):
    """Compute Otsu threshold using only pixels inside `region` (bool H×W).
    Returns (threshold_value, binary_mask_HW)."""
    pixels = diff_2d[region]
    if pixels.size == 0 or pixels.max() - pixels.min() < 1e-8:
        return 0.0, np.zeros_like(diff_2d, dtype=np.float32)
    pmax = pixels.max()
    pixels_u8 = (pixels / pmax * 255.0).astype(np.uint8)
    thresh_u8, _ = cv2.threshold(pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(thresh_u8) / 255.0 * pmax
    binary = np.zeros_like(diff_2d, dtype=np.float32)
    binary[region & (diff_2d > thresh)] = 1.0
    return thresh, binary


def anomaly_analysis(input_np: np.ndarray,
                     recon_np: np.ndarray,
                     b_mask: np.ndarray):
    """
    Args:
        input_np : (4,H,W) float32 [0,1]
        recon_np : (4,H,W) float32 [0,1]
        b_mask   : (H,W) bool brain mask
    Returns:
        diff_maps      : (4,H,W) |input - recon|
        binary_masks   : (4,H,W) per-modality binary anomaly maps
        combined_diff  : (H,W)   max over modalities
        combined_binary: (H,W)   Otsu on combined_diff within brain
    """
    diff_maps    = np.abs(input_np - recon_np)          # (4,H,W)
    binary_masks = np.zeros_like(diff_maps)
    for i in range(4):
        _, binary_masks[i] = _otsu_on_region(diff_maps[i], b_mask)
    combined_diff   = diff_maps.max(axis=0)             # (H,W)
    _, combined_binary = _otsu_on_region(combined_diff, b_mask)
    return diff_maps, binary_masks, combined_diff, combined_binary


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _dice_iou(pred: np.ndarray, gt: np.ndarray):
    inter = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    union = ((pred + gt) > 0).sum()
    dice  = float(2 * inter / total) if total > 0 else 1.0
    iou   = float(inter / union)     if union  > 0 else 1.0
    return dice, iou


def _auroc(score_2d: np.ndarray, gt_2d: np.ndarray, b_mask: np.ndarray):
    scores = score_2d[b_mask].ravel()
    labels = gt_2d[b_mask].ravel().astype(int)
    if labels.max() == 0 or labels.min() == 1:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_metrics_unhealthy(binary_masks: np.ndarray,
                              combined_binary: np.ndarray,
                              gt_mask_np: np.ndarray,
                              diff_maps: np.ndarray,
                              combined_diff: np.ndarray,
                              b_mask: np.ndarray) -> dict:
    gt = (gt_mask_np.squeeze(0) > 0).astype(np.float32)   # (H,W)
    results = {}
    for i, name in enumerate(MODALITY_NAMES):
        d, iou = _dice_iou(binary_masks[i], gt)
        auc    = _auroc(diff_maps[i], gt, b_mask)
        results[name] = {"dice": d, "iou": iou, "auroc": auc}
    d, iou = _dice_iou(combined_binary, gt)
    auc    = _auroc(combined_diff, gt, b_mask)
    results["combined"] = {"dice": d, "iou": iou, "auroc": auc}
    return results


def compute_metrics_healthy(input_np: np.ndarray, recon_np: np.ndarray) -> dict:
    results = {}
    for i, name in enumerate(MODALITY_NAMES):
        p = float(peak_signal_noise_ratio(input_np[i], recon_np[i], data_range=1.0))
        s = float(structural_similarity(input_np[i], recon_np[i], data_range=1.0))
        results[name] = {"psnr": p, "ssim": s}
    avg_in = input_np.mean(axis=0)
    avg_re = recon_np.mean(axis=0)
    p = float(peak_signal_noise_ratio(avg_in, avg_re, data_range=1.0))
    s = float(structural_similarity(avg_in, avg_re, data_range=1.0))
    results["combined"] = {"psnr": p, "ssim": s}
    return results


def _metric_str(label_int: int, mdict: dict, key: str) -> str:
    m = mdict.get(key, {})
    if label_int == 1:
        auroc_v = m.get("auroc", float("nan"))
        auroc_s = f"{auroc_v:.3f}" if not (isinstance(auroc_v, float) and np.isnan(auroc_v)) else "nan"
        return f"DICE={m.get('dice', 0):.3f}  IOU={m.get('iou', 0):.3f}  AUROC={auroc_s}"
    else:
        return f"PSNR={m.get('psnr', 0):.2f}dB  SSIM={m.get('ssim', 0):.3f}"


# ─── Visualisation ────────────────────────────────────────────────────────────

_COL_TITLES = ["Original", "Reconstruction", "Anomaly Map", "Binary (Otsu)", "GT Mask"]
_ROW_KEYS   = MODALITY_NAMES + ["combined"]


def visualize_sample(input_np, recon_np, diff_maps, binary_masks,
                     combined_diff, combined_binary, gt_mask_np,
                     label_int, mdict, idx, save_path):
    """
    Grid: 5 rows × 5 cols.
    Rows  : T1 | T1ce | T2 | FLAIR | Combined
    Cols  : Original | Reconstruction | Anomaly-jet | Binary | GT Mask
    Metrics annotated as text overlay on the first column of each row.
    """
    n_rows, n_cols = 5, 5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.2, n_rows * 4.0 + 0.6),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    gt_2d = gt_mask_np.squeeze(0).astype(np.float32)          # (H,W)

    # Build per-row data: (orig_2d, recon_2d, diff_2d, binary_2d)
    row_data = [(input_np[i], recon_np[i], diff_maps[i], binary_masks[i])
                for i in range(4)]
    row_data.append((input_np.mean(0), recon_np.mean(0), combined_diff, combined_binary))

    for r, (row_key, (orig, recon, diff, binary)) in enumerate(zip(_ROW_KEYS, row_data)):
        vmax = float(diff.max()) if diff.max() > 0 else 1.0

        axes[r, 0].imshow(orig,   cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(recon,  cmap="gray", vmin=0, vmax=1)
        im_diff = axes[r, 2].imshow(diff, cmap="jet", vmin=0, vmax=vmax)
        axes[r, 3].imshow(binary, cmap="gray", vmin=0, vmax=1)
        axes[r, 4].imshow(gt_2d,  cmap="gray", vmin=0, vmax=1)

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)

        # Row name
        axes[r, 0].set_ylabel(row_key.upper(), fontsize=9, fontweight="bold",
                               rotation=0, labelpad=40, va="center")
        axes[r, 0].yaxis.set_visible(True)

        # Metric overlay on the diff cell (col 2)
        ms = _metric_str(label_int, mdict, row_key)
        axes[r, 2].text(0.01, 0.01, ms, transform=axes[r, 2].transAxes,
                        fontsize=6.5, va="bottom", ha="left", color="white",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="black", alpha=0.55))

    # Column headers (only top row)
    for c, title in enumerate(_COL_TITLES):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold", pad=4)

    label_str = "Unhealthy" if label_int == 1 else "Healthy"
    fig.suptitle(f"Sample idx={idx}  |  {label_str}", fontsize=12, fontweight="bold")

    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device        = torch.device(args.device)
    split_file    = args.split_file    or os.path.join(args.data_path, "preprocessed_split.json")

    cfg_model, train_args = load_model(args.checkpoint, str(device))

    with open(split_file) as f:
        split = json.load(f)
    val_entries = split["val"]          # list of {"path": ..., "label": ...}
    print(f"Val set: {len(val_entries)} slices  (from {split_file})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    order           = np.random.permutation(len(val_entries))
    unhealthy_count = 0
    healthy_count   = 0
    all_uh_metrics  = []
    all_h_metrics   = []

    csv_file = open(output_dir / "metrics.csv", "w")
    csv_file.write("idx,label,modality,dice,iou,auroc,psnr,ssim,time_s\n")

    pbar = tqdm(total=args.num_unhealthy + args.num_healthy)

    for raw_idx in order:
        if unhealthy_count >= args.num_unhealthy and healthy_count >= args.num_healthy:
            break

        idx   = int(raw_idx)
        entry = val_entries[idx]
        try:
            image_np, label_int, gt_np = load_sample(
                args.data_path, entry)                  # (4,H,W), int, (1,H,W)

            if label_int == 1 and unhealthy_count >= args.num_unhealthy:
                continue
            if label_int == 0 and healthy_count >= args.num_healthy:
                continue

            t0 = time.time()

            x_in = torch.from_numpy(image_np).unsqueeze(0).to(device)  # (1,4,H,W)

            # 1. Reverse ODE: encode (t=1 → t_start)
            x_t    = encode(cfg_model, x_in, args.t, args.step_size, str(device))

            # 2. Forward ODE: decode healthy (t_start → t=1)
            x_recon = decode(cfg_model, x_t, args.t, args.step_size, str(device), args.cfg_scale)

            elapsed = time.time() - t0

            recon_np = x_recon[0].cpu().numpy()         # (4,H,W) [0,1]
            input_np = image_np                          # (4,H,W) [0,1]

            b_mask                                          = brain_mask(input_np)
            diff_maps, binary_masks, combined_diff, combined_binary \
                                                            = anomaly_analysis(input_np, recon_np, b_mask)

            if label_int == 1:
                mdict = compute_metrics_unhealthy(
                    binary_masks, combined_binary, gt_np,
                    diff_maps, combined_diff, b_mask)
                all_uh_metrics.append(mdict)
                for key in _ROW_KEYS:
                    m = mdict.get(key, {})
                    csv_file.write(
                        f"{idx},{label_int},{key},"
                        f"{m.get('dice','')},{m.get('iou','')},{m.get('auroc','')}"
                        f",,,{elapsed:.3f}\n")
                unhealthy_count += 1
            else:
                mdict = compute_metrics_healthy(input_np, recon_np)
                all_h_metrics.append(mdict)
                for key in _ROW_KEYS:
                    m = mdict.get(key, {})
                    csv_file.write(
                        f"{idx},{label_int},{key}"
                        f",,,,"
                        f"{m.get('psnr','')},{m.get('ssim','')},{elapsed:.3f}\n")
                healthy_count += 1

            csv_file.flush()

            tag = "unhealthy" if label_int == 1 else "healthy"
            visualize_sample(
                input_np, recon_np, diff_maps, binary_masks,
                combined_diff, combined_binary, gt_np,
                label_int, mdict, idx,
                output_dir / f"sample_{idx}_{tag}.png",
            )
            pbar.update(1)

        except Exception as e:
            import traceback
            print(f"\nError at idx={idx}: {e}")
            traceback.print_exc()
            continue

    pbar.close()
    csv_file.close()

    # ── Summary ──────────────────────────────────────────────────────────────

    def _nan_mean(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return np.mean(clean) if clean else float("nan")

    lines = [f"t={args.t}  step={args.step_size}  cfg_scale={args.cfg_scale}\n"]

    if all_uh_metrics:
        lines.append("\n=== Unhealthy (DICE / IOU / AUROC) ===")
        print(lines[-1])
        for key in _ROW_KEYS:
            dice_v  = _nan_mean([m[key]["dice"]  for m in all_uh_metrics if key in m])
            iou_v   = _nan_mean([m[key]["iou"]   for m in all_uh_metrics if key in m])
            auroc_v = _nan_mean([m[key]["auroc"] for m in all_uh_metrics if key in m])
            s = f"  {key:10s}  DICE={dice_v:.4f}  IOU={iou_v:.4f}  AUROC={auroc_v:.4f}"
            lines.append(s); print(s)

    if all_h_metrics:
        lines.append("\n=== Healthy (PSNR / SSIM) ===")
        print(lines[-1])
        for key in _ROW_KEYS:
            psnr_v = _nan_mean([m[key]["psnr"] for m in all_h_metrics if key in m])
            ssim_v = _nan_mean([m[key]["ssim"] for m in all_h_metrics if key in m])
            s = f"  {key:10s}  PSNR={psnr_v:.2f}dB  SSIM={ssim_v:.4f}"
            lines.append(s); print(s)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
