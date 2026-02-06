#!/usr/bin/env python3
"""
t-SNE Visualization for SOARA Ablation Study
=============================================

This script extracts embeddings from trained ViT models and creates t-SNE visualizations
to compare how different SOARA configurations (varying ranks, butterfly vs Givens) 
organize the learned representation space.

Usage:
    # Analyze a single checkpoint
    python tsne_analysis.py --checkpoint path/to/checkpoint.pth --output-dir ./tsne_plots

    # Compare multiple checkpoints
    python tsne_analysis.py \
        --checkpoints way0_r2=/path/to/r2.pth way0_r4=/path/to/r4.pth ... \
        --output-dir ./tsne_plots
"""

import os
import sys
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from transformers import ViTForImageClassification

# Import TrainingConfig and SOARA components to allow proper checkpoint loading
# (TrainingConfig is stored in the checkpoint's 'config' field as __main__.TrainingConfig)
# SOARA components are needed to reconstruct the model architecture
try:
    from train_vit_rotational import TrainingConfig
    from rotational_pissa_unified import (
        RotationalPiSSAConfig,
        replace_linear_with_rotational_pissa,
    )
    # Inject into __main__ so pickle can find it when unpickling
    import __main__
    __main__.TrainingConfig = TrainingConfig
    SOARA_AVAILABLE = True
except ImportError:
    # If running from a different directory, try adding the script dir to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from train_vit_rotational import TrainingConfig
        from rotational_pissa_unified import (
            RotationalPiSSAConfig,
            replace_linear_with_rotational_pissa,
        )
        import __main__
        __main__.TrainingConfig = TrainingConfig
        SOARA_AVAILABLE = True
    except ImportError:
        # Create a dummy class if import fails
        print("Warning: Could not import TrainingConfig/SOARA, checkpoint loading may fail")
        TrainingConfig = None
        SOARA_AVAILABLE = False


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

class EmbeddingExtractor:
    """Extract embeddings from ViT backbone at a specified layer."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", layer_idx: int = -1):
        """
        Initialize the embedding extractor.
        
        Args:
            model: ViT model
            device: Device to use
            layer_idx: Which layer to extract embeddings from.
                      -1 = final layer (default)
                       0 = after patch embeddings (before any transformer blocks)
                       1-12 = after transformer block 1-12 (ViT-Base has 12 blocks)
                       6 = middle layer (recommended for comparison)
        """
        self.model = model.to(device)
        self.device = device
        self.layer_idx = layer_idx
        self.model.eval()
        
    def extract(self, dataloader, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from the ViT backbone at the specified layer.
        
        Args:
            dataloader: DataLoader with (images, labels)
            max_samples: Maximum number of samples to extract (None = all)
            
        Returns:
            embeddings: (N, 768) numpy array of embeddings
            labels: (N,) numpy array of class labels
        """
        embeddings = []
        labels = []
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels) in enumerate(dataloader):
                images = images.to(self.device)
                
                # Forward pass through ViT backbone with hidden states output
                # This returns hidden states from all transformer layers
                outputs = self.model.vit(images, output_hidden_states=True)
                
                # outputs.hidden_states is a tuple of (num_layers + 1) tensors
                # hidden_states[0] = initial embeddings after patch embedding
                # hidden_states[1] = output of transformer block 1
                # ...
                # hidden_states[12] = output of transformer block 12 (final)
                
                if self.layer_idx == -1:
                    # Use last hidden state (same as before)
                    hidden = outputs.last_hidden_state
                else:
                    # Use specified layer
                    if outputs.hidden_states is None:
                        raise ValueError("Model did not return hidden_states. Make sure output_hidden_states=True")
                    hidden = outputs.hidden_states[self.layer_idx]
                
                # Extract CLS token embedding (position 0)
                cls_embeddings = hidden[:, 0, :]  # (batch, 768)
                
                embeddings.append(cls_embeddings.cpu().numpy())
                labels.append(batch_labels.numpy())
                
                total_samples += len(batch_labels)
                
                if max_samples and total_samples >= max_samples:
                    break
        
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        if max_samples:
            embeddings = embeddings[:max_samples]
            labels = labels[:max_samples]
        
        return embeddings, labels


# =============================================================================
# DATA LOADING
# =============================================================================

def get_fgvc_test_loader(
    data_path: str = "./data",
    selected_classes: List[int] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    """
    Load FGVC Aircraft test dataset, optionally filtered to selected classes.
    
    Args:
        data_path: Path to data directory
        selected_classes: List of class indices to include (None = all)
        batch_size: Batch size for dataloader
        num_workers: Number of data loading workers
        
    Returns:
        dataloader: DataLoader for test data
        class_names: List of class names (for selected classes)
    """
    # Standard ImageNet normalization (ViT default)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load FGVC Aircraft test split
    full_dataset = torchvision.datasets.FGVCAircraft(
        root=data_path, split='test', transform=transform, download=True
    )
    
    # Get class names
    all_class_names = full_dataset.classes
    
    if selected_classes is not None:
        # Filter to selected classes
        indices = []
        for idx, (_, label) in enumerate(full_dataset):
            if label in selected_classes:
                indices.append(idx)
        
        # Create subset
        filtered_dataset = torch.utils.data.Subset(full_dataset, indices)
        
        # Remap labels to 0..N-1 for the selected classes
        class_names = [all_class_names[c] for c in selected_classes]
        
        # Create remapping wrapper
        class RemappedDataset(torch.utils.data.Dataset):
            def __init__(self, subset, class_map):
                self.subset = subset
                self.class_map = class_map  # old_label -> new_label
                
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                img, label = self.subset[idx]
                return img, self.class_map[label]
        
        class_map = {c: i for i, c in enumerate(selected_classes)}
        dataset = RemappedDataset(filtered_dataset, class_map)
    else:
        dataset = full_dataset
        class_names = all_class_names
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return dataloader, class_names


def get_best_ratio_classes(data_path: str = "./data") -> Tuple[List[int], List[str]]:
    selected_classes = [57, 71, 92, 90, 83]
    
    # Load dataset to get class names
    temp_dataset = torchvision.datasets.FGVCAircraft(
        root=data_path, split='test', download=True
    )
    class_names = [temp_dataset.classes[c] for c in selected_classes]
    
    for idx, name in zip(selected_classes, class_names):
        print(f"  Class {idx}: {name}")
    
    return selected_classes, class_names


def select_top_accuracy_classes(
    model: nn.Module,
    data_path: str = "./data",
    n_classes: int = 5,
    device: str = "cuda"
) -> Tuple[List[int], List[str]]:
    """
    Select N classes with highest per-class accuracy from FGVC Aircraft.
    
    Args:
        model: Trained model to evaluate
        data_path: Path to data directory
        n_classes: Number of top classes to select
        device: Device to use
        
    Returns:
        selected_classes: List of class indices with highest accuracy
        class_names: Corresponding class names
    """
    # Standard ImageNet normalization (ViT default)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load FGVC Aircraft test split
    dataset = torchvision.datasets.FGVCAircraft(
        root=data_path, split='test', transform=transform, download=True
    )
    
    all_class_names = dataset.classes
    num_classes = len(all_class_names)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Compute per-class accuracy
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            
            for label, pred in zip(labels, preds):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
    
    # Compute per-class accuracy
    per_class_acc = class_correct / (class_total + 1e-8)
    
    # Get top N classes by accuracy
    top_classes = torch.argsort(per_class_acc, descending=True)[:n_classes].tolist()
    
    print(f"\nSelected top {n_classes} classes by accuracy for t-SNE visualization:")
    for idx in top_classes:
        acc = per_class_acc[idx].item() * 100
        print(f"  Class {idx}: {all_class_names[idx]} ({acc:.1f}% accuracy)")
    
    return top_classes, [all_class_names[c] for c in top_classes]


def select_diverse_classes(data_path: str = "./data", n_classes: int = 5, seed: int = 42) -> List[int]:
    """
    Select N diverse classes from FGVC Aircraft for visualization.
    DEPRECATED: Use select_top_accuracy_classes instead.
    
    We select classes that are spread across the alphabet to get visual diversity.
    
    Args:
        data_path: Path to data directory
        n_classes: Number of classes to select
        seed: Random seed for reproducibility
        
    Returns:
        List of class indices
    """
    # Load dataset to get class names
    temp_dataset = torchvision.datasets.FGVCAircraft(
        root=data_path, split='test', download=True
    )
    class_names = temp_dataset.classes
    n_total = len(class_names)
    
    # Select classes spread evenly across the list
    random.seed(seed)
    step = n_total // n_classes
    # Take every 'step'-th class starting from a random offset
    offset = random.randint(0, step - 1)
    selected = [offset + i * step for i in range(n_classes)]
    
    print(f"Selected {n_classes} classes for t-SNE visualization:")
    for idx in selected:
        print(f"  Class {idx}: {class_names[idx]}")
    
    return selected


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[nn.Module, dict]:
    """
    Load trained model from checkpoint, reconstructing SOARA adapters if needed.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model with weights (including SOARA adapters)
        config: Training configuration from checkpoint
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # TrainingConfig is injected into __main__ at module load time,
    # so pickle can find it when unpickling the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config', None)
    state_dict = checkpoint['model_state_dict']
    
    # Determine number of classes (FGVC Aircraft = 100)
    num_classes = 100
    
    # Determine model name
    model_name = "google/vit-base-patch16-224"
    if config and hasattr(config, 'model_name'):
        model_name = config.model_name
    
    # Create base model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    print(f"  Created base model================================================")
    
    # Check if this checkpoint has SOARA adapters by checking for characteristic key patterns
    # SOARA layers register: U, V, S (buffers), R_U, R_V (way0 params), 
    # butterfly_u/v or current_butterfly_u/v (way1 butterfly), givens_layers_u/v (way1 Givens)
    has_soara = any(
        k.endswith('.R_U') or k.endswith('.R_V') or  # Way0 rotation matrices
        k.endswith('.U') or k.endswith('.V') or k.endswith('.S') or  # SVD components
        'butterfly_u' in k or 'butterfly_v' in k or  # Way1 butterfly
        'givens_layers' in k or  # Way1 Givens
        'current_butterfly' in k  # Way1 butterfly sequential
        for k in state_dict.keys()
    )
    
    if has_soara and SOARA_AVAILABLE and config:
        print("  Detected SOARA adapters in checkpoint, reconstructing model architecture...")
        try:
            # Extract SOARA config from training config
            method = getattr(config, 'method', 'way0')
            # TrainingConfig uses 'pissa_rank', fallback to 'rank' then default
            rank = getattr(config, 'pissa_rank', None)
            use_butterfly = getattr(config, 'use_butterfly', False)
            butterfly_sequential = getattr(config, 'butterfly_sequential', False)
            
            # Determine target modules
            target_modules = getattr(config, 'target_modules', ['query', 'key', 'value', 'dense', 'fc1', 'fc2'])
            
            print(f"    Config: method={method}, rank={rank}, butterfly={use_butterfly}, bf_seq={butterfly_sequential}")
            
            # Create SOARA config - RotationalPiSSAConfig uses 'r' not 'rank'
            pissa_config = RotationalPiSSAConfig(
                r=rank,  # Config uses 'r' not 'rank'
                use_butterfly=use_butterfly,
                butterfly_sequential=butterfly_sequential,
                orthogonality_reg_weight=0.0,  # Not needed for inference
                method=method,
            )
            
            # Exclude classifier head
            exclude_modules = ['classifier']
            
            # Apply SOARA transformation to create adapter layers
            adapters = replace_linear_with_rotational_pissa(
                model=model,
                pissa_config=pissa_config,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
                adapter_name="default",
                freeze_base_model=True
            )
            
            print(f"    Reconstructed SOARA with method={method}, rank={rank}, butterfly={use_butterfly}")
            
        except Exception as e:
            print(f"  Warning: Failed to reconstruct SOARA: {e}")
            print("  Falling back to base model...")
    
    # Now load state dict - should match the model architecture
    try:
        # First try strict loading
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded all weights (strict mode)")
    except Exception as e1:
        # Try non-strict loading
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights (non-strict mode)")
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
        except Exception as e2:
            print(f"  Warning: Could not load weights: {e2}")
            # Try loading only matching keys
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() 
                            if k in model_state and model_state[k].shape == v.shape}
            model.load_state_dict(filtered_state, strict=False)
            print(f"  Loaded {len(filtered_state)}/{len(state_dict)} matching keys")
    
    model.to(device)
    model.eval()
    
    return model, config


# =============================================================================
# t-SNE VISUALIZATION
# =============================================================================

def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute t-SNE embedding.
    
    Args:
        embeddings: (N, D) array of high-dimensional embeddings
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
        
    Returns:
        tsne_coords: (N, 2) array of 2D coordinates
    """
    print(f"Computing t-SNE on {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, embeddings.shape[0] - 1),  # perplexity must be < n_samples
        max_iter=n_iter,  # sklearn >= 1.5 uses max_iter instead of n_iter
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    tsne_coords = tsne.fit_transform(embeddings)
    print(f"  t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return tsne_coords


def plot_tsne(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8),
    colors: List[str] = None
):
    """
    Create t-SNE scatter plot with class colors.
    
    Args:
        tsne_coords: (N, 2) array of 2D coordinates
        labels: (N,) array of class labels
        class_names: List of class names
        title: Plot title
        output_path: Path to save the plot
        figsize: Figure size
        colors: Optional list of colors for each class
    """
    n_classes = len(class_names)
    
    # Use a colormap if colors not provided
    if colors is None:
        cmap = plt.colormaps['tab10' if n_classes <= 10 else 'tab20']
        colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for class_idx in range(n_classes):
        mask = labels == class_idx
        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            c=[colors[class_idx]],
            label=class_names[class_idx],
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot to: {output_path}")


def plot_comparison(
    all_tsne_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str],
    output_path: str,
    figsize: Tuple[int, int] = None
):
    """
    Create comparison plot with all configurations side by side.
    
    Args:
        all_tsne_data: Dict mapping config_name -> (tsne_coords, labels)
        class_names: List of class names
        output_path: Path to save comparison plot
        figsize: Figure size (auto-calculated if None)
    """
    n_configs = len(all_tsne_data)
    n_classes = len(class_names)
    
    # Calculate grid layout
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    # Consistent colors across all subplots
    cmap = plt.colormaps['tab10' if n_classes <= 10 else 'tab20']
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_configs == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Flatten for easy iteration
    ax_flat = [ax for row in axes for ax in row]
    
    # Sort configs by name for consistent ordering
    config_names = sorted(all_tsne_data.keys(), key=lambda x: (
        0 if 'r2' in x else 1 if 'r4' in x else 2 if 'r8' in x else 3 if 'r16' in x else 4,
        x
    ))
    
    for idx, config_name in enumerate(config_names):
        if idx >= len(ax_flat):
            break
            
        ax = ax_flat[idx]
        tsne_coords, labels = all_tsne_data[config_name]
        
        for class_idx in range(n_classes):
            mask = labels == class_idx
            ax.scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=[colors[class_idx]],
                label=class_names[class_idx] if idx == 0 else None,
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidths=0.3
            )
        
        ax.set_title(config_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=9)
        ax.set_ylabel('t-SNE 2', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(config_names), len(ax_flat)):
        ax_flat[idx].set_visible(False)
    
    # Add shared legend
    handles, labels_legend = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', fontsize=9, framealpha=0.9)
    
    plt.suptitle('SOARA Ablation: t-SNE Embedding Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute cluster quality metrics for embeddings.
    
    Metrics:
    - intra_cluster_distance: Average distance within clusters (lower = more compact)
    - inter_cluster_distance: Average distance between cluster centers (higher = more separated)
    - silhouette_score: Overall cluster quality (-1 to 1, higher = better)
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of labels
        
    Returns:
        Dict of metric names to values
    """
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    
    unique_labels = np.unique(labels)
    
    # Compute cluster centers
    centers = []
    for label in unique_labels:
        mask = labels == label
        center = embeddings[mask].mean(axis=0)
        centers.append(center)
    centers = np.array(centers)
    
    # Intra-cluster distance: average distance to cluster center
    intra_dists = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        cluster_center = centers[label]
        dists = np.linalg.norm(cluster_points - cluster_center, axis=1)
        intra_dists.extend(dists)
    intra_cluster = np.mean(intra_dists)
    
    # Inter-cluster distance: average pairwise distance between centers
    center_dists = cdist(centers, centers, 'euclidean')
    # Take upper triangle (excluding diagonal)
    n = len(centers)
    inter_cluster = center_dists[np.triu_indices(n, k=1)].mean()
    
    # Silhouette score
    if len(unique_labels) >= 2 and len(embeddings) > len(unique_labels):
        silhouette = silhouette_score(embeddings, labels)
    else:
        silhouette = 0.0
    
    return {
        'intra_cluster_distance': intra_cluster,
        'inter_cluster_distance': inter_cluster,
        'silhouette_score': silhouette,
        'cluster_ratio': inter_cluster / (intra_cluster + 1e-8)  # Higher = better
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for SOARA Ablation")
    
    # Input options (mutually exclusive)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to a single checkpoint file')
    parser.add_argument('--checkpoints', nargs='+', type=str, default=None,
                       help='Multiple checkpoints in format name=path (e.g., way0_r2=/path/to/ckpt.pth)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory containing checkpoint subdirectories')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./tsne_plots',
                       help='Output directory for plots')
    
    # Data options
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--n-classes', type=int, default=5,
                       help='Number of classes to visualize')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples to use for t-SNE')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity')
    parser.add_argument('--layer-idx', type=int, default=6,
                       help='ViT layer to extract embeddings from (-1=final, 0=patch embed, 1-12=transformer blocks, 6=middle recommended)')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect checkpoints first (need to load first one for class selection)
    checkpoints = {}
    
    if args.checkpoint:
        # Single checkpoint
        name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        checkpoints[name] = args.checkpoint
        
    elif args.checkpoints:
        # Multiple checkpoints from command line
        for item in args.checkpoints:
            if '=' in item:
                name, path = item.split('=', 1)
            else:
                name = os.path.splitext(os.path.basename(item))[0]
                path = item
            checkpoints[name] = path
            
    elif args.checkpoint_dir:
        # Search for checkpoints in directory
        for subdir in os.listdir(args.checkpoint_dir):
            subdir_path = os.path.join(args.checkpoint_dir, subdir)
            if os.path.isdir(subdir_path):
                # Look for best model checkpoint
                for fname in os.listdir(subdir_path):
                    if fname.endswith('_best_model.pth'):
                        checkpoints[subdir] = os.path.join(subdir_path, fname)
                        break
            elif subdir.endswith('.pth'):
                name = os.path.splitext(subdir)[0]
                checkpoints[name] = os.path.join(args.checkpoint_dir, subdir)
    else:
        print("Error: Must provide --checkpoint, --checkpoints, or --checkpoint-dir")
        sys.exit(1)
    
    if not checkpoints:
        print("Error: No checkpoints found")
        sys.exit(1)
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for name, path in checkpoints.items():
        print(f"  {name}: {path}")
    
    # Use pre-computed top 5 classes by inter/intra cluster ratio
    selected_classes, class_names = get_best_ratio_classes(args.data_path)
    
    # Load test data with selected classes
    print("\nLoading FGVC Aircraft test data...")
    test_loader, class_names = get_fgvc_test_loader(
        data_path=args.data_path,
        selected_classes=selected_classes,
        batch_size=32,
        num_workers=4
    )
    print(f"  Loaded {len(test_loader.dataset)} test samples from {len(class_names)} classes")
    
    # Process each checkpoint
    all_tsne_data = {}
    all_metrics = {}
    
    for config_name, checkpoint_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Processing: {config_name}")
        print(f"{'='*60}")
        
        # Load model
        try:
            model, config = load_model_from_checkpoint(checkpoint_path, args.device)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            continue
        
        # Extract embeddings from specified layer
        layer_desc = "final" if args.layer_idx == -1 else f"layer {args.layer_idx}"
        print(f"  Extracting embeddings from {layer_desc}...")
        extractor = EmbeddingExtractor(model, args.device, layer_idx=args.layer_idx)
        embeddings, labels = extractor.extract(test_loader, max_samples=args.max_samples)
        print(f"  Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        
        # Compute cluster metrics on raw embeddings
        metrics = compute_cluster_metrics(embeddings, labels)
        all_metrics[config_name] = metrics
        print(f"  Cluster metrics:")
        print(f"    Intra-cluster distance: {metrics['intra_cluster_distance']:.4f}")
        print(f"    Inter-cluster distance: {metrics['inter_cluster_distance']:.4f}")
        print(f"    Silhouette score: {metrics['silhouette_score']:.4f}")
        print(f"    Cluster ratio (inter/intra): {metrics['cluster_ratio']:.4f}")
        
        # Compute t-SNE
        tsne_coords = compute_tsne(embeddings, perplexity=args.perplexity)
        all_tsne_data[config_name] = (tsne_coords, labels)
        
        # Save individual plot
        plot_path = os.path.join(args.output_dir, f'tsne_{config_name}.png')
        plot_tsne(
            tsne_coords, labels, class_names,
            title=f't-SNE: {config_name}',
            output_path=plot_path
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Create comparison plot if we have multiple configs
    if len(all_tsne_data) > 1:
        comparison_path = os.path.join(args.output_dir, 'tsne_comparison.png')
        plot_comparison(all_tsne_data, class_names, comparison_path)
    
    # Print summary metrics table
    print(f"\n{'='*60}")
    print("CLUSTER QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Intra':>10} {'Inter':>10} {'Silhouette':>12} {'Ratio':>10}")
    print("-" * 70)
    
    # Sort by cluster ratio (higher = better)
    sorted_configs = sorted(all_metrics.items(), key=lambda x: x[1]['cluster_ratio'], reverse=True)
    for config_name, metrics in sorted_configs:
        print(f"{config_name:<25} {metrics['intra_cluster_distance']:>10.4f} "
              f"{metrics['inter_cluster_distance']:>10.4f} {metrics['silhouette_score']:>12.4f} "
              f"{metrics['cluster_ratio']:>10.4f}")
    
    print(f"\nPlots saved to: {args.output_dir}")
    print("\nHypothesis check:")
    print("  - Higher rank should show lower intra-cluster distance (more compact)")
    print("  - Higher cluster_ratio indicates better class separation")


if __name__ == "__main__":
    main()
