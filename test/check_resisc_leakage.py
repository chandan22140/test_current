import torch
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import sys
import os

# Define a simple hash function for images to detect duplicates
def hash_image(image):
    # Resize to small to be robust to minor transforms if needed, 
    # but here we want exact match detection or close match
    # Convert to numpy and flatten
    return hash(np.array(image).tobytes())

def check_leakage():
    print("Loading timm/resisc45...")
    try:
        # Load exactly as the training script does
        hf_resisc = load_dataset("timm/resisc45", cache_dir="./data")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Available splits:", hf_resisc.keys())

    splits = {}
    hashes = {}
    
    # Analyze provided splits
    for split_name in hf_resisc.keys():
        print(f"Processing split: {split_name}")
        dataset = hf_resisc[split_name]
        split_hashes = set()
        
        # Collect hashes
        # We use a limited number if dataset is huge, but RESISC45 is 31.5k images total
        for i in tqdm(range(len(dataset))):
            img = dataset[i]['image']
            h = hash_image(img)
            split_hashes.add(h)
        
        splits[split_name] = split_hashes
        print(f"  Count: {len(dataset)}")
        print(f"  Unique hashes: {len(split_hashes)}")

    # Check for intersections
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name1 = split_names[i]
            name2 = split_names[j]
            
            intersection = splits[name1].intersection(splits[name2])
            if intersection:
                print(f"⚠️ LEAKAGE DETECTED between {name1} and {name2}!")
                print(f"  Overlap count: {len(intersection)}")
                print(f"  {name1} total: {len(splits[name1])}")
                print(f"  {name2} total: {len(splits[name2])}")
            else:
                print(f"✅ No leakage between {name1} and {name2}")

    # Check class distribution
    print("\nChecking class distribution...")
    from collections import Counter
    
    for split_name in hf_resisc.keys():
        print(f"Split: {split_name}")
        dataset = hf_resisc[split_name]
        labels = dataset['label']
        counts = Counter(labels)
        
        # Check standard deviation of counts to see balance
        counts_vals = list(counts.values())
        avg = np.mean(counts_vals)
        std = np.std(counts_vals)
        min_c = np.min(counts_vals)
        max_c = np.max(counts_vals)
        
        print(f"  Classes: {len(counts)}")
        print(f"  Avg samples/class: {avg:.1f}")
        print(f"  Min samples/class: {min_c}")
        print(f"  Max samples/class: {max_c}")
        print(f"  Std Dev: {std:.2f}")
        
        if std > 1.0:
            print("  ⚠️ Imbalanced split!")
        else:
            print("  ✅ Balanced split")

if __name__ == "__main__":
    check_leakage()
