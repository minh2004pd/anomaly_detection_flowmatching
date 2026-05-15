#!/usr/bin/env python3
"""
Gradio demo — BraTS2021 Anomaly Detection via Flow Matching.

Upload a .npy MRI slice (4×256×256) and optionally its _seg.npy ground-truth mask.
The demo visualises every Euler step of the encode and decode ODE passes,
then shows the full anomaly result grid.

Run:
    /root/.venv_brats/bin/python demo.py
or:
    UV_PROJECT_ENVIRONMENT=/root/.venv_brats uv run python demo.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.model_configs import instantiate_model
from training.eval_loop import CFGScaledModel
from infer_anomaly import (
    MODALITY_NAMES,
    _UncondVelocity,
    anomaly_analysis,
    brain_mask,
    compute_metrics_unhealthy,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "output_brats" / "checkpoint_epoch0011.pth"
ARCH       = "bratsv2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Dark-theme palette
BG          = "#0d1117"
PANEL       = "#161b22"
BORDER      = "#30363d"
DIM         = "#8b949e"
BRIGHT      = "#e6edf3"
ACCENT_BLUE = "#79c0ff"
ACCENT_RED  = "#ff7b72"   # encode (noise)
ACCENT_GRN  = "#56d364"   # decode (reconstruct)
ACCENT_ORG  = "#ffa657"   # combined row

# ── Model singleton ────────────────────────────────────────────────────────────
_MODEL: CFGScaledModel | None = None


def get_model() -> CFGScaledModel:
    global _MODEL
    if _MODEL is None:
        args_path = CHECKPOINT.parent / "args.json"
        with open(args_path) as f:
            ta = json.load(f)
        raw = instantiate_model(
            architechture=ARCH,
            is_discrete=False,
            use_ema=ta.get("use_ema", False),
        )
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        raw.load_state_dict(ckpt["model"])
        _MODEL = CFGScaledModel(raw).to(DEVICE).eval()
        print(f"[demo] {ARCH}  epoch={ckpt.get('epoch', '?')}  device={DEVICE}")
    return _MODEL


# ── Step-by-step Euler ODE ─────────────────────────────────────────────────────

@torch.no_grad()
def euler_encode(model, x01_bchw, t_start: float, step_size: float, save_every: int = 1):
    """Reverse Euler  t=1 → t_start  (unconditional, adds noise).
    Returns list[(t_val, img_4hw_01)], x_t in [-1,1].
    """
    x   = x01_bchw.to(DEVICE) * 2.0 - 1.0
    unc = _UncondVelocity(model)
    snaps, n, t = [], 0, 1.0
    while t > t_start + 1e-7:
        dt = min(step_size, t - t_start)
        tb = torch.zeros(x.shape[0], device=DEVICE).fill_(t)
        x  = x - unc(x, tb) * dt
        t  = round(t - dt, 8)
        n += 1
        if n % save_every == 0:
            snaps.append((t, (x * 0.5 + 0.5).clamp(0, 1)[0].cpu().numpy().copy()))
    return snaps, x


@torch.no_grad()
def euler_decode(model, x_t_11, t_start: float, step_size: float,
                 cfg_scale: float, save_every: int = 1):
    """Forward Euler  t_start → t=1  (CFG healthy label=0).
    Returns list[(t_val, img_4hw_01)], x_final in [-1,1].
    """
    x   = x_t_11.to(DEVICE)
    lbl = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
    snaps, n, t = [], 0, t_start
    while t < 1.0 - 1e-7:
        dt = min(step_size, 1.0 - t)
        tb = torch.zeros(x.shape[0], device=DEVICE).fill_(t)
        v  = model(x=x, t=tb, cfg_scale=cfg_scale, label=lbl)
        x  = x + v * dt
        t  = round(t + dt, 8)
        n += 1
        if n % save_every == 0:
            snaps.append((t, (x * 0.5 + 0.5).clamp(0, 1)[0].cpu().numpy().copy()))
    return snaps, x


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _fig2arr(fig) -> np.ndarray:
    """Rasterise a matplotlib figure to HxWx3 uint8."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())   # HxWx4 RGBA
    arr = buf[:, :, :3].copy()                   # drop alpha
    plt.close(fig)
    return arr


def _ax_clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(BG)


def render_input_strip(img4: np.ndarray) -> np.ndarray:
    """4 modality panels side-by-side."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.patch.set_facecolor(BG)
    for ax, name, i in zip(axes, MODALITY_NAMES, range(4)):
        ax.imshow(img4[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        _ax_clean(ax)
        ax.set_title(name, color=BRIGHT, fontsize=12, fontweight="bold", pad=5)
    fig.suptitle("Input MRI — 4 Modalities", color=ACCENT_BLUE,
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.5)
    return _fig2arr(fig)


def render_step_strip(t_val: float, img4: np.ndarray, stage: str) -> np.ndarray:
    """Gallery thumbnail for one ODE step — 4 modalities in a row."""
    if stage == "encode":
        color = ACCENT_RED
        title = f"Encoding   t = {t_val:.3f}   ▼   (noise injection)"
    else:
        color = ACCENT_GRN
        title = f"Decoding   t = {t_val:.3f}   ▲   (healthy reconstruction)"

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    fig.patch.set_facecolor(BG)
    for ax, name, i in zip(axes, MODALITY_NAMES, range(4)):
        ax.imshow(img4[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        _ax_clean(ax)
        ax.set_title(name, color=DIM, fontsize=10, pad=3)

    # t-bar at top: filled proportion indicates progress
    if stage == "encode":
        prog = 1.0 - t_val          # 0→1 as t goes 1→t_start
    else:
        prog = (t_val - 0.2) / 0.8  # rough; visual only
    prog = max(0.0, min(1.0, prog))

    bar_ax = fig.add_axes([0.05, 0.93, 0.90, 0.025])
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
    bar_ax.barh(0.5, prog, height=1.0, color=color, alpha=0.7)
    bar_ax.barh(0.5, 1.0, height=1.0, color=BORDER, alpha=0.3, zorder=0)
    bar_ax.axis("off")
    bar_ax.set_facecolor(BG)

    fig.suptitle(title, color=color, fontsize=12, fontweight="bold", y=1.06)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.05)
    return _fig2arr(fig)


def render_result_grid(
    orig_m, recon_m, diff_maps, binary_masks,
    combined_diff, combined_binary, gt_2d, mdict
) -> np.ndarray:
    """5×5 publication-style result grid (5 rows × 5 cols)."""
    row_imgs = [
        (orig_m[i], recon_m[i], diff_maps[i], binary_masks[i]) for i in range(4)
    ]
    row_imgs.append((
        orig_m[2:4].mean(0),
        recon_m[2:4].mean(0),
        combined_diff,
        combined_binary,
    ))
    row_labels = list(MODALITY_NAMES) + ["Combined\n(T2+FLAIR)"]
    row_colors = [BRIGHT] * 4 + [ACCENT_ORG]
    col_titles = ["Original", "Reconstruction", "Anomaly Map", "Binary Mask", "Ground Truth"]

    fig, axes = plt.subplots(5, 5, figsize=(26, 26))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.06, wspace=0.04,
                        left=0.09, right=0.995, top=0.96, bottom=0.005)

    for r, (label, row_color, (o, rc, df, bm)) in enumerate(
        zip(row_labels, row_colors, row_imgs)
    ):
        vd = float(df.max()) or 1.0
        panels = [
            (o,      "gray",   0, 1.0),
            (rc,     "gray",   0, 1.0),
            (df,     "jet",    0, vd ),
            (bm,     "gray",   0, 1.0),
            (gt_2d,  "gray",   0, 1.0),
        ]
        for c, (img, cm_, vn, vx) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cm_, vmin=vn, vmax=vx, interpolation="nearest")
            _ax_clean(ax)

            # Column headers (top row only)
            if r == 0:
                ax.set_title(col_titles[c], color=BRIGHT,
                             fontsize=11, fontweight="bold", pad=6)

            # Row labels (leftmost col only)
            if c == 0:
                ax.set_ylabel(label, color=row_color, fontsize=11,
                              fontweight="bold", rotation=0,
                              labelpad=65, va="center")
                ax.yaxis.set_visible(True)

            # Metrics overlay on anomaly map column
            if c == 2 and mdict:
                key = MODALITY_NAMES[r] if r < 4 else "combined"
                m   = mdict.get(key, {})
                if m:
                    d   = m.get("dice",  float("nan"))
                    iou = m.get("iou",   float("nan"))
                    auc = m.get("auroc", float("nan"))
                    clr = ACCENT_GRN if (d >= 0.6 or auc >= 0.85) else BRIGHT
                    ax.text(
                        0.03, 0.03,
                        f"DICE  {d:.3f}\nIoU   {iou:.3f}\nAUROC {auc:.3f}",
                        transform=ax.transAxes, fontsize=8.5, va="bottom",
                        color=clr,
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor=BG, alpha=0.88),
                    )

    return _fig2arr(fig)


def render_metrics_table(mdict) -> np.ndarray:
    """Coloured metrics table."""
    rows = []
    for key in MODALITY_NAMES + ["combined"]:
        m = mdict.get(key, {})
        rows.append([
            key.upper(),
            f"{m.get('dice',  float('nan')):.4f}",
            f"{m.get('iou',   float('nan')):.4f}",
            f"{m.get('auroc', float('nan')):.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Modality", "DICE ↑", "IoU ↑", "AUROC ↑"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)

    # Find best DICE row for highlight
    dice_vals = [_safe_float(r[1]) for r in rows]
    best = int(np.argmax(dice_vals))

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.8)

        if r == 0:                             # header
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=BRIGHT, fontweight="bold")
        elif c == 0:                            # row labels
            is_combined = (r - 1 == len(MODALITY_NAMES))
            cell.set_facecolor(PANEL)
            cell.set_text_props(
                color=ACCENT_ORG if is_combined else ACCENT_BLUE
            )
        else:
            v = _safe_float(cell.get_text().get_text())
            if v >= 0.70:
                cell.set_facecolor("#0d2818")
                cell.set_text_props(color=ACCENT_GRN, fontweight="bold")
            elif v >= 0.50:
                cell.set_facecolor("#1f2d1a")
                cell.set_text_props(color="#e3b341")
            else:
                cell.set_facecolor(PANEL)
                cell.set_text_props(color=DIM)

        # Highlight best-DICE row with green border
        if r == best + 1:
            cell.set_linewidth(2.8)
            cell.set_edgecolor(ACCENT_GRN)

    plt.tight_layout(pad=0.2)
    return _fig2arr(fig)


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


# ── Inference entry point ──────────────────────────────────────────────────────

def run_inference(
    npy_file,
    seg_file,
    cfg_scale: float,
    t_val: float,
    step_size: float,
    n_viz: int,
    progress=gr.Progress(track_tqdm=False),
):
    """Called by the Run button. Returns all Gradio output values."""
    if npy_file is None:
        return (None, None, None, None, None, "⚠️  Please upload a .npy file first.")

    t0  = time.time()
    log = []

    def emit(msg: str) -> str:
        log.append(msg)
        return "\n".join(log[-7:])

    try:
        # ── Load data ──────────────────────────────────────────────────────────
        progress(0.02, desc="Loading .npy file…")
        image_np = np.load(npy_file).astype(np.float32)
        if image_np.ndim != 3 or image_np.shape[0] != 4:
            raise ValueError(f"Expected shape (4, H, W), got {image_np.shape}")
        s = emit(
            f"✓ MRI loaded  shape={tuple(image_np.shape)}"
            f"  range=[{image_np.min():.3f}, {image_np.max():.3f}]"
        )

        if seg_file is not None:
            gt_np = np.load(seg_file).astype(np.float32)
            s = emit(f"✓ GT mask loaded  tumor px={int(gt_np.sum())}")
        else:
            gt_np = np.zeros((1, *image_np.shape[1:]), dtype=np.float32)
            s = emit("ℹ  No GT provided — metrics tab will be empty.")

        # ── Model ──────────────────────────────────────────────────────────────
        progress(0.05, desc="Loading model…")
        model = get_model()
        s = emit(f"✓ Model [{ARCH}] ready on {DEVICE}")

        # ── ODE step budget ────────────────────────────────────────────────────
        n_enc = max(1, round((1.0 - t_val) / step_size))
        se    = max(1, n_enc // n_viz)
        sd    = se
        x_in  = torch.from_numpy(image_np).unsqueeze(0)   # (1,4,H,W)

        # ── Encoding ───────────────────────────────────────────────────────────
        progress(0.08, desc=f"Encoding  t=1.0 → {t_val}  ({n_enc} steps)…")
        enc_snaps, x_t = euler_encode(model, x_in, t_val, step_size, save_every=se)
        s = emit(f"✓ Encode done — {len(enc_snaps)} snapshots  (every {se} steps)")

        # ── Decoding ───────────────────────────────────────────────────────────
        progress(0.50, desc=f"Decoding  t={t_val} → 1.0  cfg={cfg_scale}…")
        dec_snaps, x_fin = euler_decode(
            model, x_t, t_val, step_size, cfg_scale, save_every=sd
        )
        s = emit(f"✓ Decode done — {len(dec_snaps)} snapshots")

        # ── Anomaly analysis ───────────────────────────────────────────────────
        progress(0.92, desc="Anomaly analysis…")
        x_recon  = (x_fin * 0.5 + 0.5).clamp(0, 1)[0].cpu().numpy()
        b_mask   = brain_mask(image_np)
        orig_m   = image_np * b_mask[None]
        recon_m  = x_recon  * b_mask[None]

        diff_maps, binary_masks, combined_diff, combined_binary = anomaly_analysis(
            orig_m, recon_m, b_mask,
            postprocess=True,
            min_component_size=100,
            border_erosion=3,
            border_overlap_thr=0.6,
            combined_idx=(2, 3),          # T2+FLAIR
        )

        has_gt = gt_np.max() > 0
        mdict  = (
            compute_metrics_unhealthy(
                binary_masks, combined_binary, gt_np,
                diff_maps, combined_diff, b_mask,
            )
            if has_gt
            else None
        )
        if mdict:
            c_m = mdict.get("combined", {})
            s = emit(
                f"✓ Combined  DICE={c_m.get('dice', 0):.3f}"
                f"  IoU={c_m.get('iou', 0):.3f}"
                f"  AUROC={c_m.get('auroc', 0):.3f}"
            )

        # ── Render ─────────────────────────────────────────────────────────────
        progress(0.95, desc="Rendering figures…")

        input_img = render_input_strip(image_np)

        enc_gallery = [
            (render_step_strip(t, img, "encode"), f"t = {t:.3f}")
            for t, img in enc_snaps
        ]
        dec_gallery = [
            (render_step_strip(t, img, "decode"), f"t = {t:.3f}")
            for t, img in dec_snaps
        ]

        gt_2d = gt_np.squeeze(0) if gt_np.ndim == 3 else gt_np
        result_img = render_result_grid(
            orig_m, recon_m, diff_maps, binary_masks,
            combined_diff, combined_binary, gt_2d, mdict,
        )

        metrics_img = render_metrics_table(mdict) if mdict else None

        elapsed = time.time() - t0
        s = emit(f"✅  Finished in {elapsed:.1f}s")
        progress(1.0)

        return input_img, enc_gallery, dec_gallery, result_img, metrics_img, s

    except Exception as exc:
        import traceback
        msg = f"❌  {exc}\n\n{traceback.format_exc()}"
        return None, None, None, None, None, msg


# ── Gradio UI ──────────────────────────────────────────────────────────────────

_CSS = """
/* ── global bg ──────────────────────────────── */
body, .gradio-container, .gradio-container > .main  { background:#0d1117 !important; }
.dark { --body-background-fill:#0d1117; }

/* ── panels / cards ─────────────────────────── */
.panel, .block         { background:#161b22 !important; border:1px solid #30363d !important; }

/* ── tab bar ─────────────────────────────────── */
.tab-nav button        { color:#8b949e !important; font-weight:600; }
.tab-nav button.selected { color:#e6edf3 !important; border-bottom:2px solid #79c0ff !important; }

/* ── primary button ─────────────────────────── */
.primary               { background:#238636 !important; border-color:#2ea043 !important; font-size:15px !important; }
.primary:hover         { background:#2ea043 !important; }

/* ── file upload ─────────────────────────────── */
.upload-container      { border:2px dashed #30363d !important; }

/* ── gallery thumbnails ──────────────────────── */
.grid-wrap img         { border-radius:4px; border:1px solid #30363d; }

/* ── footer ─────────────────────────────────── */
footer                 { display:none !important; }

/* ── scrollbar ───────────────────────────────── */
::-webkit-scrollbar          { width:6px; height:6px; }
::-webkit-scrollbar-track    { background:#0d1117; }
::-webkit-scrollbar-thumb    { background:#30363d; border-radius:3px; }
"""

_HEADER = """
# 🧠 Brain Tumor Anomaly Detection — Flow Matching Demo

Upload a **BraTS2021 preprocessed `.npy` slice** `(4 × 256 × 256, modalities: T1 · T1ce · T2 · FLAIR)`.
The model **encodes** the scan toward noise (t = 1 → t_start, unconditional ODE), then
**decodes** back to the healthy distribution using **Classifier-Free Guidance** (label = healthy).
The pixel-wise reconstruction difference is thresholded to produce the anomaly mask.
"""

_FOOTER = """
---
**Checkpoint**: `output_brats/checkpoint_epoch0011.pth`   ·   **Architecture**: `bratsv2` (5-level UNet, 4-ch MRI, 2-class CFG)
**Method**: Conditional Optimal Transport Flow Matching + CFG  ·  **Dataset**: BraTS2021
"""

_EXAMPLE_NPY = str(
    ROOT / "data" / "brats2021" / "unhealthy" / "BraTS2021_00000" / "slice_088.npy"
)
_EXAMPLE_SEG = str(
    ROOT / "data" / "brats2021" / "unhealthy" / "BraTS2021_00000" / "slice_088_seg.npy"
)


# ── Data-folder browser helpers ────────────────────────────────────────────────

def _scan_data_folder() -> list[tuple[str, str]]:
    """Return sorted list of (display_label, npy_abs_path) for all downloaded slices."""
    uh_root = ROOT / "data" / "brats2021" / "unhealthy"
    items: list[tuple[str, str]] = []
    if not uh_root.exists():
        return items
    for case_dir in sorted(uh_root.iterdir()):
        if not case_dir.is_dir():
            continue
        for f in sorted(case_dir.glob("*.npy")):
            if "_seg" in f.name:
                continue
            label = f"{case_dir.name}  /  {f.name}"
            items.append((label, str(f)))
    return items


_DATA_CHOICES: list[tuple[str, str]] = _scan_data_folder()
_CHOICE_LABELS = [lbl for lbl, _ in _DATA_CHOICES]
_CHOICE_MAP    = {lbl: path for lbl, path in _DATA_CHOICES}


def load_from_folder(label: str) -> tuple[str | None, str | None]:
    """Given a dropdown label → return (npy_path, seg_path | None)."""
    if not label:
        return None, None
    npy_path = _CHOICE_MAP.get(label)
    if npy_path is None:
        return None, None
    seg_path = npy_path.replace(".npy", "_seg.npy")
    return npy_path, (seg_path if Path(seg_path).exists() else None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Brain Tumor Anomaly Detection") as demo:

        gr.Markdown(_HEADER)

        with gr.Row(equal_height=False):

            # ── Left column: inputs + controls ────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                # ── Option A: pick from downloaded data ───────────────────────
                with gr.Accordion("📁  Pick from data folder", open=True):
                    folder_dd = gr.Dropdown(
                        choices=_CHOICE_LABELS,
                        value=_CHOICE_LABELS[0] if _CHOICE_LABELS else None,
                        label=f"Sample  ({len(_CHOICE_LABELS)} slices available)",
                        filterable=True,
                    )
                    load_btn = gr.Button("⬆  Load selected", size="sm")
                    gr.Markdown(
                        "<small>GT mask (_seg.npy) is loaded automatically when present.</small>"
                    )

                gr.Markdown("**— or upload your own —**")

                # ── Option B: upload ──────────────────────────────────────────
                npy_upload = gr.File(
                    label="📂  MRI Slice (.npy)  — required",
                    file_types=[".npy"],
                    type="filepath",
                )
                seg_upload = gr.File(
                    label="🏷️  GT Segmentation (_seg.npy)  — optional",
                    file_types=[".npy"],
                    type="filepath",
                )

                with gr.Accordion("⚙️  Parameters", open=True):
                    cfg_scale_sl = gr.Slider(
                        0.0, 3.0, value=0.5, step=0.05,
                        label="CFG Scale  (higher → stronger push toward healthy)",
                    )
                    t_sl = gr.Slider(
                        0.05, 0.50, value=0.20, step=0.05,
                        label="Encode endpoint  t  (smaller → deeper into noise)",
                        info="t=0 is pure noise; t=1 is clean data.",
                    )
                    step_sl = gr.Slider(
                        0.005, 0.10, value=0.02, step=0.005,
                        label="ODE step size  (smaller → more accurate, slower)",
                    )
                    n_viz_sl = gr.Slider(
                        4, 20, value=8, step=1,
                        label="Snapshots per phase  (encode + decode gallery size)",
                    )

                run_btn = gr.Button(
                    "▶  Run Inference", variant="primary", size="lg"
                )

                status_box = gr.Textbox(
                    label="📋  Log",
                    lines=7,
                    max_lines=9,
                    interactive=False,
                    placeholder="Waiting for input…",
                )

            # ── Right column: output tabs ──────────────────────────────────────
            with gr.Column(scale=4):
                with gr.Tabs() as tabs:

                    # ── Tab 1: Input ───────────────────────────────────────────
                    with gr.Tab("📷  Input"):
                        gr.Markdown(
                            "The four MRI modalities of the uploaded slice (normalised to [0,1])."
                        )
                        input_out = gr.Image(
                            label="Input MRI — T1 · T1ce · T2 · FLAIR",
                            type="numpy",
                            interactive=False,
                        )

                    # ── Tab 2: Encoding ────────────────────────────────────────
                    with gr.Tab("📉  Encoding  (t = 1 → t_start)"):
                        gr.Markdown(
                            "**Unconditional reverse ODE** — noise is injected, erasing pathological structure.  \n"
                            "Each frame shows the 4 modalities at one Euler step.  \n"
                            "Red progress bar ▼ fills as the scan approaches pure noise."
                        )
                        enc_out = gr.Gallery(
                            label="Encoding steps",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=580,
                            interactive=False,
                        )

                    # ── Tab 3: Decoding ────────────────────────────────────────
                    with gr.Tab("📈  Decoding  (t_start → 1)"):
                        gr.Markdown(
                            "**Classifier-Free Guidance forward ODE** — reconstructs a *healthy* version.  \n"
                            "Label = 0 (healthy) with CFG scale steers the trajectory away from tumors.  \n"
                            "Green progress bar ▲ fills as the reconstruction completes."
                        )
                        dec_out = gr.Gallery(
                            label="Decoding steps",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=580,
                            interactive=False,
                        )

                    # ── Tab 4: Results ─────────────────────────────────────────
                    with gr.Tab("🎯  Results"):
                        gr.Markdown(
                            "**Rows**: T1 · T1ce · T2 · FLAIR · Combined (T2+FLAIR union)  \n"
                            "**Cols**: Original · Reconstruction · Anomaly Map (jet) · Binary Mask (Otsu) · Ground Truth  \n"
                            "Metrics (DICE / IoU / AUROC) are annotated on the anomaly map column when GT is provided."
                        )
                        result_out = gr.Image(
                            label="Detection Results",
                            type="numpy",
                            interactive=False,
                            height=960,
                        )

                    # ── Tab 5: Metrics ─────────────────────────────────────────
                    with gr.Tab("📊  Metrics"):
                        gr.Markdown(
                            "Per-modality **DICE / IoU / AUROC** (requires GT `_seg.npy` upload).  \n"
                            "🟢 ≥ 0.70  ·  🟡 ≥ 0.50  ·  Highlighted row = best DICE modality."
                        )
                        metrics_out = gr.Image(
                            label="Metrics Table",
                            type="numpy",
                            interactive=False,
                        )

        gr.Markdown(_FOOTER)

        # ── Wiring ─────────────────────────────────────────────────────────────
        # "Load selected" fills the File components from the dropdown
        load_btn.click(
            fn=load_from_folder,
            inputs=[folder_dd],
            outputs=[npy_upload, seg_upload],
        )
        # Also auto-load when the dropdown value changes
        folder_dd.change(
            fn=load_from_folder,
            inputs=[folder_dd],
            outputs=[npy_upload, seg_upload],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[npy_upload, seg_upload, cfg_scale_sl, t_sl, step_sl, n_viz_sl],
            outputs=[input_out, enc_out, dec_out, result_out, metrics_out, status_box],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser()
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--share", action="store_true")
    a = p.parse_args()

    ui = build_ui()
    ui.launch(
        server_name=a.host,
        server_port=a.port,
        share=a.share,
        css=_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
