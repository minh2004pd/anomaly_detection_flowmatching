#!/usr/bin/env python3
"""
Download checkpoint from HuggingFace Hub for local inference.

Usage:
    uv run python hf_download.py --repo vipghn2003/brats-flow-matching
    uv run python hf_download.py --repo vipghn2003/brats-flow-matching --output_dir ./output_brats
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/repo-name")
    parser.add_argument("--output_dir", default="./output_brats", help="Where to save downloaded files")
    parser.add_argument("--filename", default="checkpoint.pth", help="Filename to download")
    parser.add_argument("--token", default=None, help="HF token for private repos (or set HF_TOKEN env var)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in [args.filename, "args.json"]:
        print(f"Downloading {filename} ...")
        try:
            path = hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                local_dir=str(output_dir),
                token=args.token,
            )
            print(f"  -> saved to {path}")
        except Exception as e:
            print(f"  -> skipped ({e})")

    print(f"\nDone. Checkpoint at: {output_dir}/{args.filename}")
    print(f"Run inference with:")
    print(f"  uv run python infer_anomaly.py --checkpoint {output_dir}/{args.filename} --data_path ./data/brats2021")


if __name__ == "__main__":
    main()
