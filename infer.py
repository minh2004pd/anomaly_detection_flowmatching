#!/usr/bin/env python3
"""
General-purpose inference script for flow-matching models.

Loads a checkpoint (and its args.json), generates images for specified labels,
and saves a grid with labels displayed above each image.

Usage examples:
    # Generate 10 images per digit (0-9) from MNIST checkpoint:
    python infer.py --checkpoint ./output_mnist/checkpoint.pth --labels 0 1 2 3 4 5 6 7 8 9

    # Generate 8 images of digit "3" only:
    python infer.py --checkpoint ./output_mnist/checkpoint.pth --labels 3 --num_per_label 8

    # Generate random-label images with custom CFG scale:
    python infer.py --checkpoint ./output_mnist/checkpoint.pth --num_images 64 --cfg_scale 3.0

    # Use a CIFAR-10 checkpoint:
    python infer.py --checkpoint ./output_cifar10/checkpoint.pth --labels 0 1 2 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.model_configs import MODEL_CONFIGS, instantiate_model
from training.eval_loop import CFGScaledModel
from flow_matching.solver.ode_solver import ODESolver
from training.edm_time_discretization import get_time_discretization


# Dataset-specific metadata
DATASET_INFO = {
    "mnist": {"resolution": 28, "channels": 1, "num_classes": 10,
              "class_names": [str(i) for i in range(10)]},
    "cifar10": {"resolution": 32, "channels": 3, "num_classes": 10,
                "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                                "dog", "frog", "horse", "ship", "truck"]},
    "imagenet": {"resolution": 64, "channels": 3, "num_classes": 1000,
                 "class_names": None},
    "simple_shape": {"resolution": 64, "channels": 1, "num_classes": None,
                     "class_names": None},
}


def get_args():
    parser = argparse.ArgumentParser(description="Flow-matching model inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pth file (expects args.json in same directory)"
    )
    parser.add_argument(
        "--labels", type=int, nargs="+", default=None,
        help="Class labels to generate. Each label gets --num_per_label images. "
             "If not specified, generates --num_images images with random labels."
    )
    parser.add_argument(
        "--num_per_label", type=int, default=4,
        help="Number of images to generate per label (default: 4)"
    )
    parser.add_argument(
        "--num_images", type=int, default=16,
        help="Total images to generate when --labels is not specified (default: 16)"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=None,
        help="Override the CFG scale from training args (default: use training value)"
    )
    parser.add_argument(
        "--step_size", type=float, default=None,
        help="Override ODE step size (smaller = better quality, slower)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path (default: <checkpoint_dir>/generated.png)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint and its training args."""
    checkpoint_path = Path(checkpoint_path)
    args_path = checkpoint_path.parent / "args.json"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not args_path.exists():
        print(f"Error: args.json not found: {args_path}")
        sys.exit(1)

    with open(args_path, "r") as f:
        train_args = json.load(f)

    print(f"Loading model for dataset: {train_args['dataset']}")
    print(f"  EMA: {train_args.get('use_ema', False)}")
    print(f"  Discrete: {train_args.get('discrete_flow_matching', False)}")

    model = instantiate_model(
        architechture=train_args["dataset"],
        is_discrete=train_args.get("discrete_flow_matching", False),
        use_ema=train_args.get("use_ema", False),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    cfg_model = CFGScaledModel(model=model)
    cfg_model.to(device)
    cfg_model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    return cfg_model, train_args


def get_resolution_and_channels(train_args: dict) -> tuple:
    """Determine image resolution and channels from training args."""
    dataset = train_args["dataset"]
    if dataset in DATASET_INFO:
        info = DATASET_INFO[dataset]
        return info["resolution"], info["channels"]

    # Fallback: read from model config
    config = MODEL_CONFIGS.get(dataset, {})
    channels = config.get("in_channels", 3)
    # Default resolution
    return 32, channels


def build_labels(args, train_args: dict, device: str) -> torch.Tensor:
    """Build the label tensor based on user args."""
    dataset = train_args["dataset"]
    num_classes = DATASET_INFO.get(dataset, {}).get("num_classes")
    config = MODEL_CONFIGS.get(dataset, {})
    model_num_classes = config.get("num_classes")

    if model_num_classes is None:
        # Unconditional model
        if args.labels is not None:
            print("Warning: Model is unconditional (num_classes=None). Labels will be ignored.")
        total = args.num_images
        return torch.zeros(total, dtype=torch.int32, device=device), total

    if args.labels is not None:
        # Repeat each label num_per_label times
        label_list = []
        for label in args.labels:
            label_list.extend([label] * args.num_per_label)
        total = len(label_list)
        return torch.tensor(label_list, dtype=torch.int32, device=device), total
    else:
        # Random labels
        total = args.num_images
        nc = num_classes or model_num_classes
        labels = torch.randint(0, nc, (total,), dtype=torch.int32, device=device)
        return labels, total


@torch.no_grad()
def generate(model, train_args: dict, labels: torch.Tensor,
             num_images: int, device: str, args) -> torch.Tensor:
    """Generate images using the flow-matching ODE solver."""
    resolution, channels = get_resolution_and_channels(train_args)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else train_args["cfg_scale"]
    ode_opts = train_args["ode_options"]
    step_size = args.step_size if args.step_size is not None else ode_opts.get("step_size", 0.01)

    print(f"\nGenerating {num_images} images...")
    print(f"  Resolution: {resolution}x{resolution}, Channels: {channels}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  ODE method: {train_args['ode_method']}, Step size: {step_size}")

    x_0 = torch.randn(
        [num_images, channels, resolution, resolution],
        dtype=torch.float32, device=device
    )

    solver = ODESolver(velocity_model=model)

    if train_args.get("edm_schedule"):
        nfes = ode_opts.get("nfe", 100)
        time_grid = get_time_discretization(nfes=nfes)
    else:
        time_grid = torch.tensor([0.0, 1.0], device=device)

    model.reset_nfe_counter()
    synthetic_samples = solver.sample(
        time_grid=time_grid,
        x_init=x_0,
        method=train_args["ode_method"],
        return_intermediates=False,
        atol=ode_opts.get("atol", 1e-5),
        rtol=ode_opts.get("rtol", 1e-5),
        step_size=step_size,
        label=labels,
        cfg_scale=cfg_scale,
    )

    # Scale from [-1, 1] to [0, 1]
    synthetic_samples = torch.clamp(synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0)

    print(f"  Done! NFE: {model.get_nfe()}")
    return synthetic_samples


def save_grid_with_labels(images: torch.Tensor, labels: torch.Tensor,
                          save_path: str, dataset: str):
    """Save a grid of images with labels displayed above each one."""
    num_images = images.shape[0]
    ncols = min(8, num_images)
    nrows = math.ceil(num_images / ncols)

    # Get class names if available
    class_names = DATASET_INFO.get(dataset, {}).get("class_names")

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = axes[i, j]
            ax.axis("off")
            if idx < num_images:
                img = images[idx].detach().cpu()
                if img.shape[0] == 1:
                    ax.imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img.permute(1, 2, 0).numpy())

                label_val = labels[idx].item()
                if class_names and label_val < len(class_names):
                    title = f"{class_names[label_val]} ({label_val})"
                else:
                    title = f"Label: {label_val}"
                ax.set_title(title, fontsize=10, pad=3)

    plt.suptitle(f"Generated samples — {dataset}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to: {save_path}")


def main():
    args = get_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model, train_args = load_model(args.checkpoint, args.device)

    # Build labels
    labels, num_images = build_labels(args, train_args, args.device)

    # Generate
    images = generate(model, train_args, labels, num_images, args.device, args)

    # Save output
    output_path = args.output or str(Path(args.checkpoint).parent / "generated.png")
    save_grid_with_labels(images, labels, output_path, train_args["dataset"])


if __name__ == "__main__":
    main()
