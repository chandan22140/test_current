"""
Quick Reconstruction Error Analysis (Lightweight Version)
==========================================================

This script analyzes a SINGLE layer from each model to quickly estimate
reconstruction errors without downloading full models.

Much faster than full analysis - good for initial testing.
"""

import torch
import gc
import json
from typing import Dict, List
from transformers import AutoModelForCausalLM
import numpy as np


def compute_reconstruction_error_single_layer(
    weight: torch.Tensor,
    rank: int,
) -> Dict[str, float]:
    """Compute reconstruction error for U U^T W V V^T approximation."""
    with torch.no_grad():
        w = weight.detach().cpu().float()
        
        if min(w.shape) < rank:
            return None
        
        # SVD
        U, S, Vt = torch.linalg.svd(w, full_matrices=False)
        
        # Top-r components
        U_r = U[:, :rank]
        V_r = Vt[:rank, :]
        
        # Way 4 reconstruction: U U^T W V^T V
        temp1 = w @ V_r.T @ V_r
        temp2 = U_r @ (U_r.T @ temp1)
        
        # Errors
        error_abs = torch.norm(w - temp2, p='fro').item()
        w_norm = torch.norm(w, p='fro').item()
        error_rel = (error_abs / w_norm * 100) if w_norm > 0 else 0.0
        
        # Best rank-r approximation
        W_best = U_r @ torch.diag(S[:rank]) @ V_r
        best_error_abs = torch.norm(w - W_best, p='fro').item()
        best_error_rel = (best_error_abs / w_norm * 100) if w_norm > 0 else 0.0
        
        return {
            'way4_error_percent': error_rel,
            'best_rank_r_error_percent': best_error_rel,
            'shape': list(w.shape),
        }


def quick_test():
    """Quick test on first linear layer of each model."""
    
    models = [
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "google/gemma-7b",
        "meta-llama/Meta-Llama-3-8B",
    ]
    
    ranks = [8, 16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    results = {}
    
    print("="*80)
    print("QUICK RECONSTRUCTION ERROR ANALYSIS")
    print("(Testing only first linear layer per model)")
    print("="*80)
    
    for model_name in models:
        model_short = model_name.split('/')[-1]
        print(f"\n{'='*80}")
        print(f"Model: {model_short}")
        print(f"{'='*80}")
        
        results[model_short] = {}
        
        for dtype in dtypes:
            dtype_str = str(dtype).split('.')[-1]
            print(f"\nDtype: {dtype_str}")
            
            try:
                # Load model
                print(f"  Loading model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map='cpu',
                    trust_remote_code=True,
                )
                
                # Find first linear layer
                first_layer = None
                layer_name = None
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        first_layer = module
                        layer_name = name
                        break
                
                if first_layer is None:
                    print("  ❌ No linear layer found!")
                    continue
                
                print(f"  Layer: {layer_name}")
                print(f"  Shape: {list(first_layer.weight.shape)}")
                print(f"\n  {'Rank':<6} {'Way4 Error':<15} {'Best Rank-r Error':<15}")
                print(f"  {'-'*40}")
                
                results[model_short][dtype_str] = {}
                
                for rank in ranks:
                    metrics = compute_reconstruction_error_single_layer(
                        first_layer.weight,
                        rank=rank
                    )
                    
                    if metrics:
                        results[model_short][dtype_str][f'rank{rank}'] = metrics
                        print(f"  {rank:<6} {metrics['way4_error_percent']:>12.2f}%  {metrics['best_rank_r_error_percent']:>12.2f}%")
                
                del model
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Save results
    with open("quick_reconstruction_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(f"{'Model':<25} {'Dtype':<10} {'Rank':<6} {'Way4 Error':<12} {'Best Rank-r':<12}")
    print("-" * 80)
    
    for model_short, dtype_results in results.items():
        for dtype_str, rank_results in dtype_results.items():
            for rank_key, metrics in rank_results.items():
                rank = rank_key.replace('rank', '')
                print(f"{model_short:<25} {dtype_str:<10} {rank:<6} "
                      f"{metrics['way4_error_percent']:>10.2f}% "
                      f"{metrics['best_rank_r_error_percent']:>10.2f}%")
    
    print(f"\n✓ Results saved to: quick_reconstruction_analysis.json")


if __name__ == "__main__":
    quick_test()
