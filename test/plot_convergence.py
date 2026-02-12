#!/usr/bin/env python
"""
Plot convergence curves (val accuracy vs epochs) for rank ablation study.
Creates one plot per dataset with 4 curves (ranks 2, 4, 8, 16).
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Settings
LOG_DIR = "/home/chandan/test_current/outputs/rank_ablation"
OUTPUT_DIR = "/home/chandan/test_current/outputs/rank_ablation/plots"
DATASETS = ["cifar100", "dtd", "sun397", "fer2013", "fgvc_aircraft"]
RANKS = [2, 4, 8, 16]
COLORS = {2: '#e74c3c', 4: '#f39c12', 8: '#27ae60', 16: '#3498db'}
LABELS = {2: 'Rank 2', 4: 'Rank 4', 8: 'Rank 8', 16: 'Rank 16'}


def parse_log_file(log_path):
    """Extract epoch-wise val accuracy from training log."""
    val_accs = []
    pattern = r"Epoch \d+/\d+: .* Val Acc: ([\d.]+)"
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    val_accs.append(float(match.group(1)) * 100)
    except FileNotFoundError:
        print(f"  Warning: Log not found: {log_path}")
        return None
    
    return val_accs if val_accs else None


def plot_dataset_convergence(dataset, log_dir, output_dir):
    """Create convergence plot for a single dataset with all ranks."""
    
    plt.figure(figsize=(8, 6))
    
    # LARGE font size (as big as title would be)
    FONT_SIZE = 22
    
    all_vals = []
    found_data = False
    for rank in RANKS:
        log_path = os.path.join(log_dir, f"train_{dataset}_r{rank}.log")
        val_accs = parse_log_file(log_path)
        
        if val_accs:
            epochs = list(range(1, len(val_accs) + 1))
            plt.plot(epochs, val_accs, 
                    color=COLORS[rank], 
                    label=LABELS[rank],
                    linewidth=2,
                    marker='o',
                    markersize=4)
            found_data = True
            all_vals.extend(val_accs)
            print(f"  {LABELS[rank]}: {len(val_accs)} epochs, final acc = {val_accs[-1]:.2f}%")
    
    if not found_data:
        print(f"  No data found for {dataset}")
        plt.close()
        return None
    
    # NO TITLE, NO GRID
    plt.xlabel('Epoch', fontsize=FONT_SIZE)
    plt.ylabel('Validation Accuracy (%)', fontsize=FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    
    # Add legend ONLY for DTD dataset (bold and bigger)
    if dataset.lower() == 'dtd':
        plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 20}, borderpad=1, labelspacing=1)
    
    plt.xlim(0.5, None)
    # Dynamic limits
    if all_vals:
        y_min = max(0, min(all_vals)) * 0.995
        y_max = min(100, max(all_vals)) * 1.005
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(0, 100)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"convergence_{dataset}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Rank Ablation Convergence Plots")
    print("=" * 60)
    
    # Create individual plots
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        plot_dataset_convergence(dataset, LOG_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
