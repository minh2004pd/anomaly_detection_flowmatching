#!/usr/bin/env python3
"""
Automation script for Anomaly Detection Experiments.
Iterates through hyperparameters (t, cfg_scale, step_size) and runs infer_anomaly.py.
Organized into Phases for systematic tuning.
"""

import subprocess
import os
import sys
from pathlib import Path

# Use the conda environment python to avoid ModuleNotFoundError
PYTHON_ENV = "/home/k66/miniconda3/envs/flow_matching/bin/python"

def run_experiment(t, cfg, step, num_samples=20, output_root="experiments"):
    dir_name = f"t{t}_cfg{cfg}_step{step}"
    output_dir = Path(output_root) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        PYTHON_ENV, "infer_anomaly.py",
        "--checkpoint", "./output_brats_256_dropprob0.1/checkpoint.pth",
        "--data_path", "./data/brats2021",
        "--split_file", "./data/brats_split.json",
        "--t", str(t),
        "--cfg_scale", str(cfg),
        "--step_size", str(step),
        "--negative_guidance",
        "--num_samples", str(num_samples),
        "--output_dir", str(output_dir)
    ]
    
    print(f"\n>>> Running Experiment: t={t}, cfg={cfg}, step={step}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Done. Results saved to: {output_dir}")
        
        # Aggregate results into a master CSV
        summary_file = output_dir / "summary.txt"
        master_csv = Path(output_root) / "results_summary.csv"
        
        if summary_file.exists():
            with open(summary_file, "r") as f:
                lines = f.readlines()
                # Expected format:
                # Avg DICE: 0.1234
                # Avg IoU: 0.5678
                # Avg Time: 4.567
                dice = lines[0].split(":")[-1].strip()
                iou = lines[1].split(":")[-1].strip()
                inf_time = lines[2].split(":")[-1].strip()
            
            file_exists = master_csv.exists()
            with open(master_csv, "a") as f:
                if not file_exists:
                    f.write("t,cfg_scale,step_size,avg_dice,avg_iou,avg_time\n")
                f.write(f"{t},{cfg},{step},{dice},{iou},{inf_time}\n")
            print(f"Aggregated metrics to {master_csv}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")

def phase_1_timesteps():
    """Phase 1: Determining Optimal t (Interpolation Balance)"""
    print("\n" + "="*50)
    print("PHASE 1: VARYING t (Interpolation point)")
    print("Fixed: cfg_scale=10.0, step_size=0.02")
    print("="*50)
    for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_experiment(t=t, cfg=3.0, step=0.01, num_samples=100, output_root="exp_phase1_t")

def phase_2_cfg(best_t):
    """Phase 2: Optimizing cfg_scale (Healthy Bias)"""
    print("\n" + "="*50)
    print(f"PHASE 2: VARYING cfg_scale (Guidance strength) | best_t={best_t}")
    print("Fixed: step_size=0.02")
    print("="*50)
    for cfg in [1.0, 5.0, 8.0, 15.0, 19.0, 25.0, 35.0]:
        run_experiment(t=best_t, cfg=cfg, step=0.02, num_samples=50, output_root="exp_phase2_cfg")

def phase_3_step(best_t, best_cfg):
    """Phase 3: Refining step_size (Precision vs. Speed)"""
    print("\n" + "="*50)
    print(f"PHASE 3: VARYING step_size | best_t={best_t}, best_cfg={best_cfg}")
    print("="*50)
    for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
        run_experiment(t=best_t, cfg=best_cfg, step=step, num_samples=50, output_root="exp_phase3_step")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py <phase_num> [best_t] [best_cfg]")
        print("Example Phase 1: python run_experiments.py 1")
        print("Example Phase 2: python run_experiments.py 2 0.6")
        print("Example Phase 3: python run_experiments.py 3 0.6 20.0")
        sys.exit(1)
        
    phase = int(sys.argv[1])
    
    if phase == 1:
        phase_1_timesteps()
    elif phase == 2:
        if len(sys.argv) < 3:
            print("Error: Phase 2 requires best_t")
            sys.exit(1)
        phase_2_cfg(float(sys.argv[2]))
    elif phase == 3:
        if len(sys.argv) < 4:
            print("Error: Phase 3 requires best_t and best_cfg")
            sys.exit(1)
        phase_3_step(float(sys.argv[2]), float(sys.argv[3]))
    else:
        print(f"Unknown phase: {phase}")
