"""
Training script for ViT with Rotational PiSSA
==============================================

This script demonstrates training ViT models on multiple datasets using all 4 rotational methods:
- Way 0: Direct parameterization with regularization
- Way 1: Greedy sequential Givens rotations
- Way 2: Low-rank symmetric perturbation
- Way 3: Exponential map of skew-symmetric matrix

Supported Models:


Supported Datasets:
- cifar10 (10 classes, 50k train / 10k test)
- cifar100 (100 classes, 50k train / 10k test)
- flowers102 (102 classes, flower recognition)
- food101 (101 classes, food recognition)
- resisc45 (45 classes, remote sensing scenes)

Usage:
    # Train on CIFAR-100 with ViT-B/16
    python train_vit_rotational.py --method way0 --dataset cifar100 --epochs 10
    
    # Train on Flowers102 with ViT-L/16
    python train_vit_rotational.py --method way1 --dataset flowers102 --model vit_large_patch16_224 --epochs 20
    
    # Train on Food101 with ViT-B/16
    python train_vit_rotational.py --method way2 --dataset food101 --epochs 15
    
    # Train all methods
    python train_vit_rotational.py --method all --dataset cifar10 --epochs 5
"""

import os
import sys



# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from collections import Counter

from transformers import ViTForImageClassification, ViTConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import wandb
import pandas as pd
from PIL import Image

# Import our rotational PiSSA modules
from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    RotationalLinearLayer,
    replace_linear_with_rotational_pissa,
    RotationalPiSSATrainer
)
from vram_profiler import (
    profile_model_memory, 
    print_memory_report, 
    profile_vram_during_training,
    FLOPSProfiler,
    estimate_training_flops
)


@dataclass
class TrainingConfig:
    """Configuration for ViT training with rotational PiSSA."""
    
    # Model configuration
    model_name: str = None
    image_size: int = 224
    
    # Training configuration
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0
    warmup_epochs: float = 1.0
    
    # Rotational PiSSA configuration
    method: str = "way0"  # way0, way1, way2, way3, or 'all'
    pissa_rank: int = 16
    pissa_alpha: float = 32.0
    lora_dropout: float = 0.0  # Dropout rate (should be 0 for PiSSA)
    orthogonality_weight: float = 1e-3
    regularization_type: str = "frobenius"  # frobenius, determinant, log_determinant (Way 0 only)
    steps_per_phase: int = 0   # For way1: 0 = auto-compute, >0 = manual override
    total_cycles: int = 2      # For way1: how many full cycles through all Givens layers
    low_rank_r: int = 4        # For way2/way3
    quantize_residual: bool = False  # Whether to NF4 quantize W_residual
    quantize_base_components: bool = False  # Whether to NF4 quantize U and V^T
    
    # Way 1 Butterfly options
    use_butterfly: bool = False
    butterfly_sequential: bool = False
    butterfly_block_size: int = 2
    
    # Target modules for adaptation (HuggingFace ViT naming)
    target_modules: List[str] = field(default_factory=lambda: [
        "query",    # Query projection only
        "value",    # Value projection only
        "key",    # Uncomment to include key projection
        "dense"     # Attention output projection (equiv to timm's 'proj')
    ])
    
    # Data configuration
    dataset: str = None
    data_path: str = "./data"
    num_workers: int = 4
    k_shot: Optional[int] = None  # If set, limit to k samples per class (with upsampling if needed)
    use_dataset_stats: bool = False  # If True, compute mean/std from training data instead of using ImageNet stats
    
    # Logging configuration
    use_wandb: bool = True
    project_name: str = "rotational-pissa-vit"
    experiment_name: Optional[str] = None
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    
    # Device configuration
    device: str = "auto"  # auto, cuda, cpu
    
    # Freezing strategy to control trainable parameter count
    freeze_backbone: bool = True            # Freeze all non-adapter params by default
    train_head: bool = True                 # Keep classifier head trainable
    track_grad_norm: bool = False           # Whether to compute and log gradient norm
    
    @property
    def num_classes(self) -> int:
        """Get number of classes based on dataset."""
        dataset_classes = {
            "cifar10": 10,
            "cifar100": 100,
            "flowers102": 102,
            "food101": 101,
            "resisc45": 45
        }
        if self.dataset not in dataset_classes:
            raise ValueError(f"Unsupported dataset: {self.dataset}. Supported: {list(dataset_classes.keys())}")
        return dataset_classes[self.dataset]


class ParquetImageDataset(Dataset):
    """Custom dataset for loading images from Parquet files.
    
    Expected parquet structure:
        - image: dict with 'bytes' key containing JPEG/PNG bytes
        - label: integer class label
        - image_id: optional identifier
    """
    
    def __init__(self, parquet_path: str, transform=None):
        """
        Args:
            parquet_path: Path to the parquet file
            transform: Optional transform to apply to images
        """
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extract image bytes from nested dict structure
        image_data = row['image']
        if isinstance(image_data, dict):
            image_bytes = image_data['bytes']
        else:
            image_bytes = image_data
        
        # Decode image from bytes
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = int(row['label'])
        
        return image, label


class ViTDataset:
    """Dataset wrapper for ViT training on multiple datasets."""
   
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset_mean = None
        self.dataset_std = None
        
        # Will be set after computing stats if use_dataset_stats=True
        # Otherwise use ImageNet defaults
        if config.use_dataset_stats:
            print("📊 Will compute dataset-specific normalization statistics...")
            self.norm_mean = None  # Computed later
            self.norm_std = None
        else:
            self.norm_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
            self.norm_std = [0.229, 0.224, 0.225]
        
        # Define standard augmentations for training (normalization added later)
        self.train_transform_base = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])
        
        # Validation transform (no augmentation, normalization added later)
        self.val_transform_base = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor()
        ])
        
        # Placeholder for final transforms (with normalization)
        self.train_transform = None
        self.val_transform = None
    
    def _compute_dataset_statistics(self, dataset, max_samples=5000):
        """Compute mean and std of dataset for normalization.
        
        Args:
            dataset: Dataset to compute statistics from
            max_samples: Maximum number of samples to use for computation
        
        Returns:
            (mean, std): Tuples of per-channel mean and std
        """
        print(f"Computing dataset statistics from up to {max_samples} samples...")
        
        # Use transform without normalization
        temp_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor()
        ])
        
        # Temporarily replace transform
        original_transform = None
        if hasattr(dataset, 'transform'):
            original_transform = dataset.transform
            dataset.transform = temp_transform
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
            original_transform = dataset.dataset.transform
            dataset.dataset.transform = temp_transform
        
        # Compute statistics
        n_samples = min(len(dataset), max_samples)
        indices = torch.randperm(len(dataset))[:n_samples].tolist()
        
        # Accumulate mean
        mean = torch.zeros(3)
        std = torch.zeros(3)
        
        print("  Computing mean...")
        for i, idx in enumerate(indices):
            if i % 1000 == 0:
                print(f"    Processed {i}/{n_samples} samples")
            img, _ = dataset[idx]
            mean += img.reshape(3, -1).mean(dim=1)
        mean /= n_samples
        
        print("  Computing std...")
        for i, idx in enumerate(indices):
            if i % 1000 == 0:
                print(f"    Processed {i}/{n_samples} samples")
            img, _ = dataset[idx]
            std += ((img - mean.view(3, 1, 1)) ** 2).reshape(3, -1).mean(dim=1)
        std = torch.sqrt(std / n_samples)
        
        # Restore original transform
        if original_transform is not None:
            if hasattr(dataset, 'transform'):
                dataset.transform = original_transform
            elif hasattr(dataset, 'dataset'):
                dataset.dataset.transform = original_transform
        
        mean_list = mean.tolist()
        std_list = std.tolist()
        
        print(f"✓ Dataset statistics computed:")
        print(f"  Mean: [{mean_list[0]:.4f}, {mean_list[1]:.4f}, {mean_list[2]:.4f}]")
        print(f"  Std:  [{std_list[0]:.4f}, {std_list[1]:.4f}, {std_list[2]:.4f}]")
        print(f"  (ImageNet default: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225])")
        
        return mean_list, std_list
    
    def _extract_stratified_val_split(self, dataset, val_fraction=0.1, seed=42):
        """Extract a stratified validation split from a dataset.
        
        Args:
            dataset: Source dataset
            val_fraction: Fraction of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            (train_subset, val_subset): Two Subset datasets with stratified class distribution
        """
        from torch.utils.data import Subset
        from collections import defaultdict
        import random
        
        # Extract labels from dataset - try multiple common attribute names
        labels = None
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, '_labels'):
            labels = dataset._labels
        else:
            raise ValueError(f"Dataset (type: {type(dataset)}) must have 'targets', 'labels', or '_labels' attribute for stratified splitting")
        
        # Group indices by class
        random.seed(seed)
        cls_to_idx = defaultdict(list)
        for idx, label in enumerate(labels):
            cls_to_idx[int(label)].append(idx)
        
        # Split each class proportionally
        train_idx = []
        val_idx = []
        for cls, inds in cls_to_idx.items():
            random.shuffle(inds)
            n_val = max(1, int(len(inds) * val_fraction))
            val_idx.extend(inds[:n_val])
            train_idx.extend(inds[n_val:])
        
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get training, validation and unseen test dataloaders.
        
        Flow:
        1. Load raw datasets (train, val, test)
        2. For datasets without val split: extract stratified val from train
        3. Apply k-shot filtering to train (if specified)
        4. Apply balancing sampler to train (if needed)
        5. Create dataloaders
        
        Only the train set is balanced. Val and test sets remain as-is.
        """
        from torch.utils.data import Subset
        import os
        
        # ========== 1. Load raw datasets ==========
        # Create temporary transform for loading (without normalization)
        temp_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor()
        ])
        
        if self.config.dataset == "cifar10":
            # CIFAR-10: has train/test split, need to carve out val from train
            dataset_class = torchvision.datasets.CIFAR10
            full_train = dataset_class(
                root=self.config.data_path, train=True,
                transform=temp_transform, download=True
            )
            train_dataset, val_dataset = self._extract_stratified_val_split(full_train, val_fraction=0.1)
            test_dataset = dataset_class(
                root=self.config.data_path, train=False,
                transform=temp_transform, download=True
            )

        elif self.config.dataset == "cifar100":
            # CIFAR-100: has train/test split, need to carve out val from train
            dataset_class = torchvision.datasets.CIFAR100
            full_train = dataset_class(
                root=self.config.data_path, train=True,
                transform=temp_transform, download=True
            )
            train_dataset, val_dataset = self._extract_stratified_val_split(full_train, val_fraction=0.1)
            test_dataset = dataset_class(
                root=self.config.data_path, train=False,
                transform=temp_transform, download=True
            )

        elif self.config.dataset == "food101":
            # Food101: has train/test split, need to carve out val from train
            dataset_class = torchvision.datasets.Food101
            full_train = dataset_class(
                root=self.config.data_path, split='train',
                transform=temp_transform, download=True
            )
            train_dataset, val_dataset = self._extract_stratified_val_split(full_train, val_fraction=0.05)
            test_dataset = dataset_class(
                root=self.config.data_path, split='test',
                transform=temp_transform, download=True
            )

        elif self.config.dataset == "flowers102":
            # Flowers102: has train/val/test splits
            dataset_class = torchvision.datasets.Flowers102
            train_dataset = dataset_class(
                root=self.config.data_path, split='train',
                transform=temp_transform, download=True
            )
            val_dataset = dataset_class(
                root=self.config.data_path, split='val',
                transform=temp_transform, download=True
            )
            test_dataset = dataset_class(
                root=self.config.data_path, split='test',
                transform=temp_transform, download=True
            )

        elif self.config.dataset == "resisc45":
            # RESISC45: has train/val/test splits (parquet format)
            train_parquet = os.path.join(self.config.data_path, 'resisc45', 'train-00000-of-00001.parquet')
            val_parquet = os.path.join(self.config.data_path, 'resisc45', 'validation-00000-of-00001.parquet')
            test_parquet = os.path.join(self.config.data_path, 'resisc45', 'test-00000-of-00001.parquet')

            if not os.path.exists(train_parquet):
                raise FileNotFoundError(f"RESISC45 train parquet not found: {train_parquet}")
            if not os.path.exists(val_parquet):
                raise FileNotFoundError(f"RESISC45 validation parquet not found: {val_parquet}")
            if not os.path.exists(test_parquet):
                print(f"Warning: Test parquet not found, using validation as test")
                test_parquet = val_parquet

            train_dataset = ParquetImageDataset(train_parquet, transform=temp_transform)
            val_dataset = ParquetImageDataset(val_parquet, transform=temp_transform)
            test_dataset = ParquetImageDataset(test_parquet, transform=temp_transform)

        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        # ========== 2. Compute dataset statistics (if requested) ==========
        if self.config.use_dataset_stats and self.norm_mean is None:
            # Compute stats from training data before k-shot sampling
            self.norm_mean, self.norm_std = self._compute_dataset_statistics(train_dataset)
        
        # If not using dataset stats, use ImageNet defaults (already set in __init__)
        if self.norm_mean is None:
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
        
        # Now finalize transforms with normalization
        self.train_transform = transforms.Compose([
            self.train_transform_base,
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.val_transform = transforms.Compose([
            self.val_transform_base,
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        
        # Apply transforms to datasets
        # For Subset datasets, we need to update the underlying dataset's transform
        if hasattr(train_dataset, 'dataset'):
            train_dataset.dataset.transform = self.train_transform
        else:
            train_dataset.transform = self.train_transform
            
        if hasattr(val_dataset, 'dataset'):
            val_dataset.dataset.transform = self.val_transform
        else:
            val_dataset.transform = self.val_transform
            
        if hasattr(test_dataset, 'dataset'):
            test_dataset.dataset.transform = self.val_transform
        else:
            test_dataset.transform = self.val_transform
        
        # ========== 3. Apply k-shot filtering to train (if specified) ==========
        if self.config.k_shot is not None:
            train_dataset = self._apply_kshot_sampling(train_dataset, self.config.k_shot)
            print(f"Applied {self.config.k_shot}-shot sampling to training set")

        # ========== 4. Apply balancing sampler to train (if needed) ==========
        train_sampler = self._make_balanced_sampler(train_dataset)
        if train_sampler is not None:
            print(f"Applied balanced sampling to training set")
        
        # ========== 5. Create dataloaders ==========
        if train_sampler is not None:
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                sampler=train_sampler, num_workers=self.config.num_workers, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                shuffle=True, num_workers=self.config.num_workers, pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader
    
    def _make_balanced_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
        """Create a balanced sampler if dataset is imbalanced.
        
        Uses deterministic oversampling to ensure perfect class balance.
        Returns None if dataset is already perfectly balanced.
        """
        # Extract labels
        labels = None
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, '_labels'):
            labels = dataset._labels
        elif hasattr(dataset, 'dataset'):
            # Handle Subset datasets
            if hasattr(dataset.dataset, 'targets') and hasattr(dataset, 'indices'):
                labels = [dataset.dataset.targets[i] for i in dataset.indices]
            elif hasattr(dataset.dataset, '_labels') and hasattr(dataset, 'indices'):
                labels = [dataset.dataset._labels[i] for i in dataset.indices]
        elif isinstance(dataset, ParquetImageDataset):
            labels = dataset.df['label'].tolist()
        
        if labels is None:
            return None

        labels = list(labels)
        counts = Counter(labels)
        max_count = max(counts.values())
        min_count = min(counts.values())
        
        # Check if already perfectly balanced
        if max_count == min_count:
            return None

        # Create deterministic oversampling for perfect balance
        from collections import defaultdict
        import random
        
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[int(label)].append(idx)
        
        # For each class, oversample to max_count
        balanced_indices = []
        random.seed(42)  # For reproducibility
        for cls, indices in sorted(class_to_indices.items()):
            if len(indices) < max_count:
                # Oversample with replacement
                balanced_indices.extend(random.choices(indices, k=max_count))
            else:
                # Already has max_count or more, just take max_count
                balanced_indices.extend(indices[:max_count])
        
        # Shuffle for good measure
        random.shuffle(balanced_indices)
        
        # Create a sampler that uses these indices
        sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
        return sampler
    
    def _apply_kshot_sampling(self, dataset, k: int):
        """Apply k-shot sampling to a dataset.
        
        For each class:
        - If class has >= k samples: randomly select k samples
        - If class has < k samples: upsample with replacement to reach k samples
        
        Args:
            dataset: The dataset to sample from
            k: Number of samples per class
            
        Returns:
            Subset dataset with k samples per class
        """
        from torch.utils.data import Subset
        
        # Extract labels
        labels = None
        if hasattr(dataset, 'targets'):
            labels = list(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = list(dataset.labels)
        elif hasattr(dataset, '_labels'):
            labels = list(dataset._labels)
        elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # For Subset datasets - try multiple label attributes
            if hasattr(dataset.dataset, 'targets'):
                full_labels = dataset.dataset.targets
                labels = [full_labels[i] for i in dataset.indices]
            elif hasattr(dataset.dataset, '_labels'):
                full_labels = dataset.dataset._labels
                labels = [full_labels[i] for i in dataset.indices]
            else:
                raise ValueError(f"Subset's parent dataset has no recognized label attribute")
        elif isinstance(dataset, ParquetImageDataset):
            labels = dataset.df['label'].tolist()
        else:
            print(f"Warning: Could not extract labels from dataset type {type(dataset)}. Skipping k-shot sampling.")
            return dataset
        
        # Group indices by class
        class_to_indices = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Sample k indices per class
        selected_indices = []
        import random
        random.seed(42)  # For reproducibility
        
        for class_label, indices in sorted(class_to_indices.items()):
            if len(indices) >= k:
                # Randomly select k samples
                sampled = random.sample(indices, k)
            else:
                # Upsample: sample with replacement to reach k
                sampled = random.choices(indices, k=k)
                print(f"  Class {class_label}: upsampled from {len(indices)} to {k} samples")
            
            selected_indices.extend(sampled)
        
        # Create subset with selected indices
        # If dataset is already a Subset, we need to map back to original indices
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # Map selected_indices through the Subset's indices
            original_indices = [dataset.indices[i] for i in selected_indices]
            return Subset(dataset.dataset, original_indices)
        else:
            return Subset(dataset, selected_indices)


class ViTRotationalTrainer:
    """Trainer for ViT with rotational PiSSA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()

        # Setup data
        self.dataset = ViTDataset(config)
        self.train_loader, self.val_loader, self.test_loader = self.dataset.get_dataloaders()

        # Initialize results storage
        self.results = {}

        print(f"🚀 ViT Rotational PiSSA Trainer Initialized")
        print(f"Device: {self.device}")
        try:
            train_n = len(self.train_loader.dataset)
        except Exception:
            pass
            train_n = getattr(self.train_loader.dataset, '__len__', lambda: 'N/A')()
        try:
            val_n = len(self.val_loader.dataset)
        except Exception:
            pass
            val_n = getattr(self.val_loader.dataset, '__len__', lambda: 'N/A')()
        try:
            test_n = len(self.test_loader.dataset)
        except Exception:
            pass
            test_n = getattr(self.test_loader.dataset, '__len__', lambda: 'N/A')()
        print(f"Dataset: {self.config.dataset} ({train_n} train, {val_n} val, {test_n} test)")

    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def create_model(self, method: str) -> Tuple[nn.Module, Optional[RotationalPiSSATrainer]]:
        """Create ViT model with rotational PiSSA adaptation."""
        
        # Load pre-trained ViT from HuggingFace with separate Q/K/V projections
        model = ViTForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            ignore_mismatched_sizes=True  # Allow classifier head mismatch
        )
        
        print(f"Created {self.config.model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Note: Using HuggingFace model with separate query/key/value projections")
        
        # Configure rotational PiSSA
        if method == "way0":
            pissa_config = RotationalPiSSAConfig(
                r=self.config.pissa_rank,
                lora_alpha=self.config.pissa_alpha,
                lora_dropout=self.config.lora_dropout,
                method="way0",
                orthogonality_reg_weight=self.config.orthogonality_weight,
                regularization_type=self.config.regularization_type,
                quantize_residual=self.config.quantize_residual,
                quantize_base_components=self.config.quantize_base_components
            )
        elif method == "way1":
            # Compute steps_per_phase if not explicitly set, or use config value
            # Formula: (total_steps) / (total_cycles × num_phases_per_cycle)
            # where num_phases_per_cycle = r - 1 (number of Givens layers per cycle)
            total_steps = len(self.train_loader) * self.config.epochs
            if (self.config.pissa_rank - 1) % 2 == 0:
                num_phases_per_cycle = self.config.pissa_rank - 1
            else:
                num_phases_per_cycle = self.config.pissa_rank
            

            # Auto-compute to fit exactly into total epochs
            steps_per_phase = total_steps // (self.config.total_cycles * num_phases_per_cycle)
            # Ensure at least 1 step per phase
            steps_per_phase = max(1, steps_per_phase)
        
            print(f"Way1 config: total_steps={total_steps}, cycles={self.config.total_cycles}, "
                  f"phases_per_cycle={num_phases_per_cycle}, steps_per_phase={steps_per_phase}")
            
            pissa_config = RotationalPiSSAConfig(
                r=self.config.pissa_rank,
                lora_alpha=self.config.pissa_alpha,
                lora_dropout=self.config.lora_dropout,
                method="way1",
                steps_per_phase=steps_per_phase,
                total_cycles=self.config.total_cycles,
                quantize_residual=self.config.quantize_residual,
                quantize_base_components=self.config.quantize_base_components,
                use_butterfly=self.config.use_butterfly,
                butterfly_sequential=self.config.butterfly_sequential,
                butterfly_block_size=self.config.butterfly_block_size
            )
        elif method == "way2":
            pissa_config = RotationalPiSSAConfig(
                r=self.config.pissa_rank,
                lora_alpha=self.config.pissa_alpha,
                lora_dropout=self.config.lora_dropout,
                method="way2",
                low_rank_r=self.config.low_rank_r,
                quantize_residual=self.config.quantize_residual,
                quantize_base_components=self.config.quantize_base_components
            )
        elif method == "way3":
            pissa_config = RotationalPiSSAConfig(
                r=self.config.pissa_rank,
                lora_alpha=self.config.pissa_alpha,
                lora_dropout=self.config.lora_dropout,
                method="way3",
                low_rank_r=self.config.low_rank_r,
                quantize_residual=self.config.quantize_residual,
                quantize_base_components=self.config.quantize_base_components
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Replace linear layers with rotational adapters
        # freeze_base_model flag handles all freezing internally (method-agnostic)
        adapters = replace_linear_with_rotational_pissa(
            model=model,
            pissa_config=pissa_config,
            target_modules=self.config.target_modules,
            adapter_name="default",
            freeze_base_model=self.config.freeze_backbone  # Pass config flag
        )
        
        print(f"Created {len(adapters)} rotational adapters using {method}")
        
        # Create rotational trainer helper
        rotational_trainer = RotationalPiSSATrainer(model, pissa_config)
        
        # Optionally keep classifier head trainable (if freeze_backbone is enabled)
        # HuggingFace ViT uses 'classifier' instead of 'head'
        if self.config.freeze_backbone and self.config.train_head:
            if hasattr(model, 'classifier'):
                for p in model.classifier.parameters():
                    p.requires_grad = True
            elif hasattr(model, 'head'):
                for p in model.head.parameters():
                    p.requires_grad = True
        
        # Print parameter breakdown
        if self.config.freeze_backbone:
            # HuggingFace ViT uses 'classifier' instead of 'head'
            head_params = 0
            if hasattr(model, 'classifier'):
                head_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
            elif hasattr(model, 'head'):
                head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            adapter_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and any(adp_name in name for adp_name in adapters.keys()))
            other_trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad) - adapter_params - head_params
            print(f"Trainable breakdown -> adapters: {adapter_params:,}, head: {head_params:,}, other: {other_trainable:,}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params:.4f})")
        
        # Profile VRAM usage
        print(f"\n{'='*60}")
        print("VRAM PROFILING AFTER MODEL CREATION")
        print(f"{'='*60}")
        memory_breakdown = profile_model_memory(model)
        print_memory_report(memory_breakdown)
        
        return model.to(self.device), rotational_trainer

    # core training
    def train_single_method(self, method: str) -> Dict:
        """Train model with a single rotational method."""
        
        print(f"\n{'='*60}")
        print(f"Training with {method.upper()}")
        print(f"{'='*60}")
        
        # Create model
        model, rotational_trainer = self.create_model(method)
        



        #============================
        print(f'\nCLASSIFIER MODULE DETAILS:')
        if hasattr(model, 'classifier'):
            head = model.classifier
            print(f'Classifier type: {type(head)}')
            print(f'Classifier parameters:')
            for name, param in head.named_parameters():
                print(f'  {name}: requires_grad={param.requires_grad}, shape={param.shape}')
        elif hasattr(model, 'head'):
            head = model.head
            print(f'Head type: {type(head)}')
            print(f'Head parameters:')
            for name, param in head.named_parameters():
                print(f'  {name}: requires_grad={param.requires_grad}, shape={param.shape}')
        else:
            print('No classifier/head attribute found')


        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )


        #============================LEARNIGN RATE======================
        
        # Setup scheduler with warmup
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        # Create cosine scheduler with warmup
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(progress * math.pi))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        #============================LEARNIGN RATE======================

        # # Initialize FLOPS profiler
        # flops_profiler = FLOPSProfiler(device=str(self.device))
        
        # # Count FLOPs per batch using actual batch from dataset (works for any data type)
        # # Get a sample batch from train_loader
        # sample_batch, _ = next(iter(self.train_loader))
        # sample_batch = sample_batch.to(self.device)
        
        # flops_per_batch = estimate_training_flops(
        #     model=model,
        #     sample_batch=sample_batch,
        #     backward_multiplier=2.0,
        #     device=str(self.device)
        # )
        
        # # Clean up sample batch
        # del sample_batch
        # torch.cuda.empty_cache()
        
        # total_batches = len(self.train_loader) * self.config.epochs
        # total_estimated_flops = flops_per_batch * total_batches
        
        # print(f"\n📊 FLOPS Analysis:")
        # print(f"  FLOPs per batch: {flops_per_batch/1e9:.2f} GFLOP")
        # print(f"  Total batches: {total_batches}")
        # print(f"  Total training FLOPs: {total_estimated_flops/1e12:.2f} TFLOP")
        # print(f"  Device: {flops_profiler.gpu_name}")
        # print()

        # Initialize wandb if enabled
        if self.config.use_wandb:
            # Check if this is a sweep run
            if wandb.run is None:
                run_name = f"{method}_{self.config.experiment_name or 'vit_rotational'}"
                wandb.init(
                    project=str(self.config.project_name)+str(self.config.dataset),
                    name=run_name,
                    config={
                        "method": method,
                        "rank": self.config.pissa_rank,
                        "alpha": self.config.pissa_alpha,
                        "orthogonality_reg_weight": self.config.orthogonality_weight,
                        "regularization_type": self.config.regularization_type,
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                        "epochs": self.config.epochs,
                        "low_rank_r": self.config.low_rank_r,
                        "steps_per_phase": self.config.steps_per_phase,
                        "total_cycles": self.config.total_cycles
                    }
                )
            else:
                # This is a sweep run, config already updated before trainer creation
                print(f"Running sweep with config: alpha={self.config.pissa_alpha}, "
                      f"ortho_weight={self.config.orthogonality_weight}, "
                      f"reg_type={self.config.regularization_type}, "
                      f"lr={self.config.learning_rate}, rank={self.config.pissa_rank}")
        
        # Training loop
        best_acc = 0.0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.epochs):
            # Log total trainable parameters at the start of each epoch
            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} - Trainable parameters: {total_trainable:,}")
            
            # # Start FLOPS profiling for this epoch
            # flops_profiler.start_epoch()

            # Training phase (run training, then re-evaluate train metrics on full train set)
            _ = self._train_epoch(model, optimizer, scheduler, rotational_trainer, epoch, warmup_steps)
            
            # # End FLOPS profiling for this epoch
            # epoch_flops = flops_per_batch * len(self.train_loader)
            # epoch_tflops = flops_profiler.end_epoch(epoch_flops)
            
            # print(f"  Epoch FLOPS: {epoch_flops/1e12:.3f} TFLOP, Throughput: {epoch_tflops:.2f} TFLOPS/s")
            
            # Recompute train metrics after epoch end using current model weights
            train_acc, train_loss = self._evaluate_on_loader(model, self.train_loader)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            val_acc, val_loss = self._evaluate_on_loader(model, self.val_loader)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                if self.config.save_checkpoints:
                    self._save_checkpoint(model, method, epoch, val_acc)
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "best_accuracy": best_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                # "epoch_tflops": epoch_tflops,
                # "cumulative_tflop": flops_profiler.total_flops / 1e12,
                # "average_tflops": (flops_profiler.total_flops / flops_profiler.total_time) / 1e12 if flops_profiler.total_time > 0 else 0
            }
            
            if self.config.use_wandb:
                # Use epoch-end global step for epoch-level logging
                epoch_end_step = (epoch + 1) * len(self.train_loader) - 1
                wandb.log(metrics, step=epoch_end_step)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Best Acc: {best_acc:.4f}")
        
        # Evaluate on unseen test set and log
        test_acc, test_loss = self._evaluate_on_loader(model, self.test_loader)
        
        # # Print FLOPS profiling summary
        # flops_profiler.print_summary()
        # flops_summary = flops_profiler.get_summary()

        if self.config.use_wandb:
            # Log final metrics for sweep optimization including unseen test
            total_steps = self.config.epochs * len(self.train_loader) - 1
            wandb.log({
                "final_val_accuracy": best_acc,
                "final_train_accuracy": train_accuracies[-1] if train_accuracies else 0,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "total_epochs": self.config.epochs,
                # "total_tflop": flops_summary['total_flops'] / 1e12,
                # "average_tflops": flops_summary['average_tflops'],
                # "compute_efficiency_percent": flops_summary['efficiency_percent'],
                # "theoretical_peak_tflops": flops_summary['theoretical_tflops']
            }, step=total_steps)
            wandb.finish()
        
        # Return results
        result = {
            "method": method,
            "best_accuracy": best_acc,
            "final_accuracy": val_accuracies[-1],
            "final_train_accuracy": train_accuracies[-1],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "total_epochs": self.config.epochs,
            # "flops_summary": flops_summary
        }

        return result
    
    def _train_epoch(self, model, optimizer, scheduler, rotational_trainer, epoch, warmup_steps):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        # Moving average for recent batches (last 100 batches)
        from collections import deque
        recent_losses = deque(maxlen=100)
        recent_accs = deque(maxlen=100)
        
        # Flag for VRAM profiling (only on first batch of first epoch)
        profile_vram = (epoch == 2)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Calculate global step (across all epochs) early so every wandb.log can include it
            global_step = epoch * len(self.train_loader) + batch_idx

            # Profile VRAM during first training batch
            if profile_vram and batch_idx == 0:
                print(f"\n{'='*60}")
                print("VRAM PROFILING DURING FIRST TRAINING BATCH")
                print(f"{'='*60}")
                profile_vram_during_training(model, optimizer, (data, target), self.device)
                profile_vram = False  # Only profile once
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            # HuggingFace models return ModelOutput objects, extract logits
            # print("hasattr(output, 'logits'):", hasattr(output, 'logits'))
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(logits, target)
            
            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # Add orthogonality regularization for Way 0
            if rotational_trainer.config.method == "way0":
                ortho_loss = rotational_trainer.get_orthogonality_loss()
                loss = loss + ortho_loss
                
                if batch_idx % 10 == 0 and self.config.use_wandb:
                    wandb.log({"orthogonality_loss": ortho_loss.item()}, step=global_step)
            
            # Backward pass
            # Gradient clipping (always performed for stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Rotational PiSSA update
            if rotational_trainer.config.method == "way1" and rotational_trainer.should_step_phase(global_step):
                rotational_trainer.step_phase()
            
            # Logging
            if self.config.use_wandb and global_step % 10 == 0:
                log_dict = {
                    "train_step_loss": loss.item(),
                    "epoch": epoch + batch_idx / len(self.train_loader),
                    "lr": optimizer.param_groups[0]['lr']
                }
                
                # Only sync/log grad norm if enabled to save speed
                if self.config.track_grad_norm:
                    log_dict["grad_norm"] = grad_norm.item()
                
                wandb.log(log_dict)
            
            # Step scheduler every step for proper warmup
            scheduler.step()
            
            # Calculate global step (across all epochs)
            global_step = epoch * len(self.train_loader) + batch_idx
            
            # ========================= Debug learning rate changes during warmup and first few steps
            current_lr = optimizer.param_groups[0]['lr']
            
            if global_step < warmup_steps + 10 or batch_idx % 100 == 0:
                # Print LR changes during warmup
                if global_step < warmup_steps:
                    if batch_idx % 100 == 0:
                        print(f"  Warmup Step {global_step}/{warmup_steps}: LR = {current_lr:.6f}")
                elif global_step == warmup_steps:
                    print(f"  Warmup Complete! Step {global_step}: LR = {current_lr:.6f}")
            # ========================= Debug learning rate changes during warmup and first few steps
            
            total_loss += loss.item()
            
            # Track recent batch metrics for moving average
            batch_acc = pred.eq(target).sum().item() / target.size(0)
            recent_losses.append(loss.item())
            recent_accs.append(batch_acc)
            
            # Log metrics to wandb every 10 steps
            if self.config.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_acc": batch_acc,
                    "train/learning_rate": current_lr,
                    "epoch": epoch
                }, step=global_step)
            
            # Handle phase transitions for Way 1
            if rotational_trainer.should_step_phase(global_step):
                print(f"  Step {global_step}: Advancing rotation phase")
                params_before, params_after = rotational_trainer.step_phase()
                
                if self.config.use_wandb:
                    adapter = rotational_trainer.adapters[0]
                    log_dict = {}
                    
                    # Add rotation layer tracking
                    if hasattr(adapter, 'current_layer_index'):
                        log_dict["rotation_layer_index"] = adapter.current_layer_index
                        log_dict["rotation_cycle"] = adapter.current_cycle
                    
                    # Add parameter count tracking for Way1
                    if self.config.method == "way1" and params_before > 0:
                        log_dict["way1/trainable_params_before"] = params_before
                        log_dict["way1/trainable_params_after"] = params_after
                        log_dict["way1/params_delta"] = params_after - params_before
                        print(f"    Parameters: {params_before} → {params_after} (Δ = {params_after - params_before})")
                    
                    wandb.log(log_dict, step=global_step)
            
            if batch_idx % 100 == 0:
                current_acc = correct / total_samples  # Cumulative accuracy from epoch start
                recent_avg_acc = sum(recent_accs) / len(recent_accs) if recent_accs else 0.0  # Moving avg
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.4f}, Cumulative Acc {current_acc:.4f} (recent_avg: {recent_avg_acc:.4f})")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def _evaluate_on_loader(self, model, loader):
        """Evaluate model on an arbitrary loader and return (accuracy, avg_loss).
        
        Uses model.eval() and torch.no_grad() to ensure:
        - No gradient computation (saves VRAM)
        - Deterministic behavior (no dropout)
        - Does not affect training TFLOPS measurement
        """
        was_training = model.training
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                # HuggingFace models return ModelOutput objects, extract logits
                logits = output.logits if hasattr(output, 'logits') else output
                loss = F.cross_entropy(logits, target)
                total_loss += loss.item() * target.size(0)

                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += target.size(0)

        # Restore training mode
        if was_training:
            model.train()

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return accuracy, avg_loss
    
    def _save_checkpoint(self, model, method, epoch, accuracy):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            "method": method,
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "config": self.config
        }
        
        filename = f"{method}_best_model.pth"
        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"  Saved checkpoint: {filepath}")
    
    # no need to read below... just read all 4 methods..
    def train_all_methods(self) -> Dict:
        """Train with all 4 rotational methods."""
        methods = ["way0", "way1", "way2", "way3"]
        all_results = {}
        
        print(f"\n🚀 Training all methods: {methods}")
        print(f"Epochs per method: {self.config.epochs}")
        
        start_time = time.time()
        
        for method in methods:
            method_start = time.time()
            
            try:
                result = self.train_single_method(method)
                all_results[method] = result
                
                method_time = time.time() - method_start
                print(f"✅ {method} completed in {method_time:.1f}s - Best Acc: {result['best_accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ {method} failed: {e}")
                all_results[method] = {"error": str(e)}
        
        total_time = time.time() - start_time
        print(f"\n🎉 All methods completed in {total_time:.1f}s")
        
        # Create comparison
        self._create_comparison_report(all_results)
        
        return all_results
    
    def _create_comparison_report(self, results: Dict):
        """Create comparison report and visualizations."""
        
        print(f"\n{'='*60}")
        print("COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Summary table
        print(f"{'Method':<8} {'Best Val Acc':<12} {'Final Val Acc':<14} {'Final Train Acc':<16} {'Status':<10}")
        print("-" * 70)
        
        valid_results = {}
        for method, result in results.items():
            if "error" in result:
                print(f"{method:<8} {'ERROR':<12} {'ERROR':<14} {'ERROR':<16} {'FAILED':<10}")
            else:
                final_train_acc = result.get('final_train_accuracy', 'N/A')
                final_train_str = f"{final_train_acc:.4f}" if isinstance(final_train_acc, float) else str(final_train_acc)
                print(f"{method:<8} {result['best_accuracy']:.4f}       {result['final_accuracy']:.4f}        {final_train_str:<16} {'SUCCESS':<10}")
                valid_results[method] = result
        
        if not valid_results:
            print("No successful runs to compare.")
            return
        
        # Create plots
        try:
            self._plot_comparison(valid_results)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
        
        # Save results
        results_file = os.path.join(self.config.output_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    def _plot_comparison(self, results: Dict):
        """Create comparison plots."""
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rotational PiSSA Methods Comparison on ViT-B/16', fontsize=16)
        
        methods = list(results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        # 1. Training Loss
        ax1 = axes[0, 0]
        for i, (method, result) in enumerate(results.items()):
            epochs = range(1, len(result['train_losses']) + 1)
            ax1.plot(epochs, result['train_losses'], label=method, color=colors[i], linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training and Validation Accuracy
        ax2 = axes[0, 1]
        for i, (method, result) in enumerate(results.items()):
            epochs = range(1, len(result['val_accuracies']) + 1)
            # Plot validation accuracy (solid line)
            ax2.plot(epochs, result['val_accuracies'], label=f'{method} (val)', 
                    color=colors[i], linewidth=2, linestyle='-')
            # Plot training accuracy (dashed line)
            if 'train_accuracies' in result:
                ax2.plot(epochs, result['train_accuracies'], label=f'{method} (train)', 
                        color=colors[i], linewidth=2, linestyle='--', alpha=0.7)
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Best Accuracy Comparison
        ax3 = axes[1, 0]
        best_accs = [result['best_accuracy'] for result in results.values()]
        bars = ax3.bar(methods, best_accs, color=colors[:len(methods)])
        ax3.set_title('Best Validation Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, best_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. Method characteristics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for method in methods:
            if method == "way0":
                chars = ["Direct", "Regularized", "2r²"]
            elif method == "way1":
                chars = ["Sequential", "Exact Ortho", "~r²/2"]
            elif method == "way2":
                chars = ["Low-rank", "Approx Ortho", "4kr"]
            elif method == "way3":
                chars = ["Exponential", "Exact Ortho", "4kr"]
            else:
                chars = ["Unknown", "Unknown", "Unknown"]
            
            table_data.append([method] + chars)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'Type', 'Orthogonality', 'Parameters'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Method Characteristics')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.config.output_dir, exist_ok=True)
        plot_file = os.path.join(self.config.output_dir, "methods_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train ViT with Rotational PiSSA")
    
    # Model and dataset selection
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224",
                       choices=["google/vit-base-patch16-224", "google/vit-large-patch16-224", 
                               "google/vit-base-patch32-224", "google/vit-large-patch32-224"],
                       help="ViT model architecture to use (HuggingFace model name)")
    parser.add_argument("--dataset", type=str, default="cifar100",
                       choices=["cifar10", "cifar100", "flowers102", "food101", "resisc45"],
                       help="Dataset to train on")
    
    # Method selection
    parser.add_argument("--method", type=str, default="way0",
                       choices=["way0", "way1", "way2", "way3", "all"],
                       help="Rotational method to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.00028558,
                       help="Learning rate")
    # 0.00028558  0.001
    # Rotational PiSSA parameters
    parser.add_argument("--rank", type=int, default=16,
                       help="PiSSA rank")
    parser.add_argument("--alpha", type=float, default=16,
                       help="PiSSA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                       help="Dropout rate for LoRA/PiSSA layers (should be 0 for PiSSA)")
    parser.add_argument("--orthogonality-weight", type=float, default=0.040204,
                       help="Orthogonality regularization weight (way0)")
    parser.add_argument("--regularization-type", type=str, default="frobenius",
                       choices=["frobenius", "determinant", "log_determinant"],
                       help="Regularization type for way0: frobenius (||R^T@R-I||), determinant ((det(R)-1)^2), or log_determinant (log(det(R))^2)")
    parser.add_argument("--steps-per-phase", type=int, default=50,
                       help="Steps per phase (way1)")
    parser.add_argument("--total-cycles", type=int, default=2,
                       help="Total cycles (way1)")
    parser.add_argument("--low-rank-r", type=int, default=4,
                       help="Low rank r (way2/way3)")
    parser.add_argument("--quantize-residual", action="store_true",
                       help="NF4 quantize W_residual (requires bitsandbytes)")
    parser.add_argument("--quantize-base-components", action="store_true",
                       help="NF4 quantize U and V^T (requires bitsandbytes)")
    
    # Butterfly parameters (Way 1)
    parser.add_argument("--use-butterfly", action="store_true",
                       help="Use butterfly factorization for Way 1")
    parser.add_argument("--butterfly-sequential", action="store_true",
                       help="Train butterfly components sequentially (requires --use-butterfly)")
    parser.add_argument("--butterfly-block-size", type=int, default=2,
                       help="Block size for butterfly factorization (default: 2)")
       
    # Other options
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--sweep", action="store_true",
                       help="Enable wandb sweep mode")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--k-shot", type=int, default=None,
                       help="K-shot learning: limit to k samples per class (with upsampling if needed)")
    parser.add_argument("--use-dataset-stats", action="store_true",
                       help="Compute normalization mean/std from training data instead of using ImageNet defaults")
    
    args = parser.parse_args()

    # Create configuration
    train_config = TrainingConfig(
        # Model
        model_name=args.model,
        
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        
        # Dataset
        dataset=args.dataset,
        
        # Rotational PiSSA
        method=args.method,
        pissa_rank=args.rank,
        pissa_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        orthogonality_weight=args.orthogonality_weight,
        regularization_type=args.regularization_type,
        steps_per_phase=args.steps_per_phase,
        total_cycles=args.total_cycles,
        low_rank_r=args.low_rank_r,
        quantize_residual=args.quantize_residual,
        quantize_base_components=args.quantize_base_components,
        use_butterfly=args.use_butterfly,
        butterfly_sequential=args.butterfly_sequential,
        butterfly_block_size=args.butterfly_block_size,
        
        # Other
        output_dir=args.output_dir,
        use_wandb=(not args.no_wandb) or args.sweep,  # Enable wandb unless explicitly disabled, OR if running sweep
        experiment_name=args.experiment_name,
        device=args.device,
        k_shot=args.k_shot,
        use_dataset_stats=args.use_dataset_stats
    )
    print("config.use_wandb", train_config.use_wandb)
    print("args.sweep", args.sweep)
    
    # Handle sweep mode
    if args.sweep:
        print("🔄 Starting wandb sweep agent...")
        
        def sweep_train():
            print("sweep_train called")
            """Training function for wandb sweep."""
            # Initialize wandb for sweep
            wandb.init()

            # Update config with sweep parameters BEFORE creating trainer
            print("🔄 Updating config with sweep parameters...")
            print("before update train_config:", train_config)

            for key, value in wandb.config.items():
                if hasattr(train_config, key):
                    # Convert to appropriate type based on attribute
                  
                    attr = getattr(train_config, key)
                    if isinstance(attr, float):
                        value = float(value)
                    elif isinstance(attr, int):
                        value = int(value)
                    elif isinstance(attr, bool):
                        value = bool(value)
                    elif isinstance(attr, str):
                        value = str(value)
                    
                    setattr(train_config, key, value)
                    print(f"Updated {key} from sweep: {value}")
                else:
                    print(f"Warning: Unknown sweep parameter '{key}' - skipping")
            print("after update train_config:", train_config)
            # train_config.epochs = 6
            
            if (train_config.method=="way3") or (train_config.method=="way2"):
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                print("🎯 Configured to use GPU:",os.environ["CUDA_VISIBLE_DEVICES"]  )
            elif  (train_config.method=="way0") or (train_config.method=="way1"):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print("🎯 Configured to use GPU:",os.environ["CUDA_VISIBLE_DEVICES"]  )
                
            # Create trainer with updated config
            trainer = ViTRotationalTrainer(train_config)
            
            # Train with the method specified (or from sweep)
            method = train_config.method if train_config.method != "all" else "way0"
            result = trainer.train_single_method(method)
            
            # wandb.finish() is already called inside train_single_method
            return result
        
        # This will be called by wandb sweep agent
        return sweep_train()
    
    else:
        # Normal training mode
        # Create trainer
        trainer = ViTRotationalTrainer(train_config)
        
        # Train
        if args.method == "all":
            results = trainer.train_all_methods()
            print(f"\n🎉 Training completed! Results saved to {train_config.output_dir}; results:{results}")
        else:
            result = trainer.train_single_method(args.method)
            print(f"\n🎉 Training completed! Best accuracy: {result['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()