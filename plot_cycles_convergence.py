#!/usr/bin/env python
"""
Plot convergence curves for num_cycles ablation study.
Creates 4 subplots showing train loss, train accuracy, val loss, and val accuracy.
Each subplot shows curves for cycles = 1, 3, 5, 7.
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Settings
LOG_DIR = "/home/chandan/test_current/outputs/cycles_ablation"
OUTPUT_DIR = "/home/chandan/test_current/outputs/cycles_ablation/plots"
DATASET = "fgvc_aircraft"
METHOD = "way1"
CYCLES = [1, 3, 5, 7]
COLORS = {1: '#e74c3c', 3: '#f39c12', 5: '#27ae60', 7: '#3498db'}
LABELS = {1: '1 Cycle', 3: '3 Cycles', 5: '5 Cycles', 7: '7 Cycles'}


def parse_log_file(log_path):
    """Extract epoch-wise metrics from training log.
    
    Returns:
        dict with keys: train_loss, train_acc, val_loss, val_acc
        Each value is a list of floats (one per epoch)
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Pattern: Epoch X/Y: Train Loss: A, Train Acc: B, Val Loss: C, Val Acc: D, Best Acc: E
    pattern = r"Epoch \d+/\d+: Train Loss: ([\d.]+), Train Acc: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)"
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    train_losses.append(float(match.group(1)))
                    train_accs.append(float(match.group(2)) * 100)  # Convert to percentage
                    val_losses.append(float(match.group(3)))
                    val_accs.append(float(match.group(4)) * 100)  # Convert to percentage
    except FileNotFoundError:
        print(f"  Warning: Log not found: {log_path}")
        return None
    
    if not train_losses:
        return None
    
    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }


def plot_cycles_convergence(log_dir, output_dir):
    """Create 2x2 subplot showing all 4 metrics across different num_cycles."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect data for all cycles
    all_data = {}
    for cycles in CYCLES:
        log_path = os.path.join(log_dir, f"train_{DATASET}_{METHOD}_cycles{cycles}.log")
        metrics = parse_log_file(log_path)
        
        if metrics:
            all_data[cycles] = metrics
            print(f"  {LABELS[cycles]}: {len(metrics['train_loss'])} epochs found")
        else:
            print(f"  {LABELS[cycles]}: No data found")
    
    if not all_data:
        print("  No data found for any cycle configuration!")
        plt.close()
        return None
    
    # Plot 1: Train Loss
    ax = axes[0, 0]
    for cycles in CYCLES:
        if cycles in all_data:
            epochs = list(range(1, len(all_data[cycles]['train_loss']) + 1))
            ax.plot(epochs, all_data[cycles]['train_loss'],
                   color=COLORS[cycles],
                   label=LABELS[cycles],
                   linewidth=2,
                   marker='o',
                   markersize=3)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Train Accuracy
    ax = axes[0, 1]
    for cycles in CYCLES:
        if cycles in all_data:
            epochs = list(range(1, len(all_data[cycles]['train_acc']) + 1))
            ax.plot(epochs, all_data[cycles]['train_acc'],
                   color=COLORS[cycles],
                   label=LABELS[cycles],
                   linewidth=2,
                   marker='o',
                   markersize=3)
    ax.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Val Loss
    ax = axes[1, 0]
    for cycles in CYCLES:
        if cycles in all_data:
            epochs = list(range(1, len(all_data[cycles]['val_loss']) + 1))
            ax.plot(epochs, all_data[cycles]['val_loss'],
                   color=COLORS[cycles],
                   label=LABELS[cycles],
                   linewidth=2,
                   marker='o',
                   markersize=3)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Val Accuracy
    ax = axes[1, 1]
    for cycles in CYCLES:
        if cycles in all_data:
            epochs = list(range(1, len(all_data[cycles]['val_acc']) + 1))
            ax.plot(epochs, all_data[cycles]['val_acc'],
                   color=COLORS[cycles],
                   label=LABELS[cycles],
                   linewidth=2,
                   marker='o',
                   markersize=3)
    ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{DATASET.upper()} - Num Cycles Ablation ({METHOD.upper()}, Rank 16, 20 Epochs)',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cycles_convergence_{DATASET}_{METHOD}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved: {output_path}")
    return output_path


def print_summary(log_dir):
    """Print summary statistics for all cycles."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for cycles in CYCLES:
        log_path = os.path.join(log_dir, f"train_{DATASET}_{METHOD}_cycles{cycles}.log")
        metrics = parse_log_file(log_path)
        
        if metrics:
            final_train_acc = metrics['train_acc'][-1]
            final_val_acc = metrics['val_acc'][-1]
            best_val_acc = max(metrics['val_acc'])
            final_train_loss = metrics['train_loss'][-1]
            final_val_loss = metrics['val_loss'][-1]
            
            print(f"\n{LABELS[cycles]}:")
            print(f"  Final Train Acc: {final_train_acc:.2f}%  |  Final Train Loss: {final_train_loss:.4f}")
            print(f"  Final Val Acc:   {final_val_acc:.2f}%  |  Final Val Loss:   {final_val_loss:.4f}")
            print(f"  Best Val Acc:    {best_val_acc:.2f}%")
        else:
            print(f"\n{LABELS[cycles]}: No data available")


def main():
    print("=" * 60)
    print("Num Cycles Ablation - Convergence Plots")
    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print(f"Method: {METHOD}")
    print(f"Rank: 16")
    print(f"Cycles: {CYCLES}")
    print("=" * 60)
    
    # Create plot
    plot_cycles_convergence(LOG_DIR, OUTPUT_DIR)
    
    # Print summary
    print_summary(LOG_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
