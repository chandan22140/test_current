#!/usr/bin/env python3
"""
Cluster Quality Analysis for SOARA Ablation
===========================================

Identifies classes with the best clustering properties (Silhouette Score)
across different model configurations and layers.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import silhouette_samples
from transformers import ViTForImageClassification
from tqdm import tqdm
from typing import List, Dict, Tuple

# Try to import TrainingConfig and SOARA components
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
        print("Warning: Could not import TrainingConfig/SOARA, checkpoint loading may fail")
        TrainingConfig = None
        SOARA_AVAILABLE = False


def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Load model from checkpoint with SOARA support."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', None)
    state_dict = checkpoint['model_state_dict']
    
    # Create base model (ViT-Base, 100 classes for FGVC)
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    
    # Check for SOARA
    has_soara = any(
        k.endswith('.R_U') or k.endswith('.R_V') or 
        'butterfly' in k or 'givens' in k 
        for k in state_dict.keys()
    )
    
    if has_soara and SOARA_AVAILABLE and config:
        try:
            method = getattr(config, 'method', 'way0')
            rank = getattr(config, 'pissa_rank', getattr(config, 'rank', None))
            use_butterfly = getattr(config, 'use_butterfly', False)
            butterfly_sequential = getattr(config, 'butterfly_sequential', False)
            
            pissa_config = RotationalPiSSAConfig(
                r=rank,
                use_butterfly=use_butterfly,
                butterfly_sequential=butterfly_sequential,
                method=method,
                orthogonality_reg_weight=0.0
            )
            
            replace_linear_with_rotational_pissa(
                model=model,
                pissa_config=pissa_config,
                target_modules=getattr(config, 'target_modules', ['query', 'key', 'value', 'dense', 'fc1', 'fc2']),
                exclude_modules=['classifier'],
                adapter_name="default",
                freeze_base_model=True
            )
            print(f"  Reconstructed SOARA: method={method}, rank={rank}, bf={use_butterfly}, bf_seq={butterfly_sequential}")
        except Exception as e:
            print(f"  Failed to reconstruct SOARA: {e}")

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Error loading state dict: {e}")
        
    model.to(device)
    model.eval()
    return model


def get_embeddings(model, dataloader, device, layer_idx):
    """Extract embeddings from specific layer."""
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc=f"Extracting layer {layer_idx}"):
            imgs = imgs.to(device)
            outputs = model.vit(imgs, output_hidden_states=True)
            
            if layer_idx == -1:
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[layer_idx]
                
            # CLS token
            emb = hidden[:, 0, :]
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.concatenate(embeddings), np.concatenate(labels)


def analyze_clustering(embeddings, labels, class_names):
    """Compute per-class silhouette scores."""
    print("Computing silhouette scores...")
    # Compute silhouette score for each sample
    sample_scores = silhouette_samples(embeddings, labels)
    
    class_scores = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        avg_score = sample_scores[mask].mean()
        class_scores[label] = avg_score
        
    # Sort by score descending
    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data-path', type=str, default='./data', help='Data directory')
    parser.add_argument('--layers', type=int, nargs='+', default=[12], help='Layers to analyze (0-12)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top classes to show')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-classes', type=int, default=100, help='Total classes in dataset')
    args = parser.parse_args()

    # Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.FGVCAircraft(root=args.data_path, split='test', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    class_names = dataset.classes

    # Load Model
    model = load_model(args.checkpoint, args.device)

    # Analyze Layer(s)
    results = {}
    
    print("\n" + "="*60)
    print(f"Analysis for checkpoint: {os.path.basename(args.checkpoint)}")
    print("="*60)

    for layer_idx in args.layers:
        print(f"\nAnalyzing Layer {layer_idx}...")
        emb, lbl = get_embeddings(model, dataloader, args.device, layer_idx)
        
        sorted_classes = analyze_clustering(emb, lbl, class_names)
        results[layer_idx] = sorted_classes
        
        print(f"\nTop {args.top_k} Classes by Silhouette Score (Layer {layer_idx}):")
        print(f"{'Rank':<5} {'Score':<8} {'Class ID':<10} {'Class Name'}")
        print("-" * 50)
        for rank, (cls_idx, score) in enumerate(sorted_classes[:args.top_k], 1):
            print(f"{rank:<5} {score:.4f}   {cls_idx:<10} {class_names[cls_idx]}")

        # Also print bottom 5 just to see bad ones
        # print("-" * 50)
        # print("Bottom 5 classes:")
        # for cls_idx, score in sorted_classes[-5:]:
        #    print(f"{score:.4f} (ID {cls_idx}): {class_names[cls_idx]}")

if __name__ == "__main__":
    main()
