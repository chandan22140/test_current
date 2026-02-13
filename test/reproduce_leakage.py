import sys
import os
import torch
import torchvision.transforms as transforms
from collections import Counter

# Add current directory to path
sys.path.append(os.getcwd())

from train_vit_rotational import TrainingConfig, ViTDataset

def check_transform_bug():
    print("Checking for transform bug...")
    config = TrainingConfig()
    config.dataset = "food101"
    config.data_path = "./data"
    config.batch_size = 4
    config.num_workers = 0 # Avoid multiprocessing for check
    
    # Initialize dataset wrapper
    vit_dataset = ViTDataset(config)
    
    # Get dataloaders
    # This triggers get_dataloaders which has the bug logic
    try:
        train_loader, val_loader, test_loader = vit_dataset.get_dataloaders()
    except Exception as e:
        print(f"Error getting dataloaders: {e}")
        return

    # Check transforms
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    
    if hasattr(train_ds, 'dataset'): # Subset
        train_inner = train_ds.dataset
        val_inner = val_ds.dataset
        
        print(f"Train subset underlying dataset: {type(train_inner)}")
        print(f"Val subset underlying dataset: {type(val_inner)}")
        
        if train_inner is val_inner:
            print("(!) FAIL: Train and Val subsets share the SAME underlying dataset object.")
        else:
            print("OK: Train and Val subsets have different underlying dataset objects.")
            
        print("\nChecking transforms:")
        print(f"Train transform: {train_inner.transform}")
        print(f"Val transform: {val_inner.transform}")
        
        # Check if train has RandAugment
        has_randaug = any(isinstance(t, transforms.RandAugment) for t in train_inner.transform.transforms)
        print(f"Train transform has RandAugment: {has_randaug}")
        
        if not has_randaug:
            print("(!) FAIL: Training transform does NOT have RandAugment (likely overwritten by val transform).")
        else:
            print("OK: Training transform has RandAugment.")
            
    else:
        print("Train dataset is not a Subset (unexpected for Food101).")

def check_indices_overlap():
    print("\nChecking for index overlap...")
    config = TrainingConfig()
    config.dataset = "food101"
    config.data_path = "./data"
    
    vit_dataset = ViTDataset(config)
    train_loader, val_loader, test_loader = vit_dataset.get_dataloaders()
    
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    
    overlap = train_indices.intersection(val_indices)
    print(f"Train indices count: {len(train_indices)}")
    print(f"Val indices count: {len(val_indices)}")
    print(f"Overlap count: {len(overlap)}")
    
    if len(overlap) > 0:
        print("(!) FAIL: Train and Val indices OVERLAP!")
    else:
        print("OK: No overlap between Train and Val indices.")
        
    # Check total coverage
    total_indices = len(train_indices) + len(val_indices)
    print(f"Total split indices: {total_indices}")
    # Food101 train split size is 75750
    print(f"Expected size (75750): {total_indices == 75750}")

if __name__ == "__main__":
    check_transform_bug()
    check_indices_overlap()
