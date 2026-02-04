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
                    val_accs.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"  Warning: Log not found: {log_path}")
        return None
    
    return val_accs if val_accs else None


def plot_dataset_convergence(dataset, log_dir, output_dir):
    """Create convergence plot for a single dataset with all ranks."""
    
    plt.figure(figsize=(8, 6))
    
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
            print(f"  {LABELS[rank]}: {len(val_accs)} epochs, final acc = {val_accs[-1]:.4f}")
    
    if not found_data:
        print(f"  No data found for {dataset}")
        plt.close()
        return None
    
    # Formatting
    dataset_display = dataset.upper().replace('_', ' ')
    plt.title(f'{dataset_display} - Validation Accuracy Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, None)
    plt.ylim(0, 1)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"convergence_{dataset}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_combined_plot(log_dir, output_dir):
    """Create a 2x3 combined plot with all datasets."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        
        for rank in RANKS:
            log_path = os.path.join(log_dir, f"train_{dataset}_r{rank}.log")
            val_accs = parse_log_file(log_path)
            
            if val_accs:
                epochs = list(range(1, len(val_accs) + 1))
                ax.plot(epochs, val_accs,
                       color=COLORS[rank],
                       label=LABELS[rank],
                       linewidth=1.5,
                       marker='o',
                       markersize=3)
        
        dataset_display = dataset.upper().replace('_', ' ')
        ax.set_title(dataset_display, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Val Accuracy', fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.suptitle('Rank Ablation: Validation Accuracy Convergence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "convergence_all_datasets.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined plot: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Rank Ablation Convergence Plots")
    print("=" * 60)
    
    # Create individual plots
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        plot_dataset_convergence(dataset, LOG_DIR, OUTPUT_DIR)
    
    # Create combined plot
    print("\nCreating combined plot...")
    create_combined_plot(LOG_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
