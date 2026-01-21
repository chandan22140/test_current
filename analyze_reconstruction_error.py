"""
Analyze SVD Reconstruction Error for Way 4 Approach
=====================================================

This script computes the reconstruction error when approximating w_base 
with U U^T w_base V V^T (the proposed "way 4" approach).

For each model and dtype, we:
1. Load the model
2. For each linear layer, compute SVD: W = U S V^T
3. Compute reconstruction: W_reconstructed = U[:,:r] U[:,:r]^T @ W @ V[:,:r]^T V[:,:r]
4. Measure error: ||W - W_reconstructed||_F / ||W||_F

This quantifies information loss from the double projection.

Models tested:
- LLaMA-2-7B (meta-llama/Llama-2-7b-hf)
- Mistral-7B (mistralai/Mistral-7B-v0.1)
- Gemma-7B (google/gemma-7b)
- LLaMA-3-8B (meta-llama/Meta-Llama-3-8B)

Dtypes tested: BF16, FP32
Ranks tested: 8, 16, 32, 64, 128
"""

import torch
import gc
import json
import os
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
from collections import defaultdict
import numpy as np


def compute_reconstruction_error(
    weight: torch.Tensor,
    rank: int,
    dtype: torch.dtype
) -> Dict[str, float]:
    """
    Compute reconstruction error for U U^T W V V^T approximation.
    
    Args:
        weight: Weight matrix [out_features, in_features]
        rank: Rank r for SVD truncation
        dtype: Dtype to use for computation
        
    Returns:
        Dictionary with error metrics
    """
    with torch.no_grad():
        # Convert to target dtype and ensure on CPU for SVD
        original_dtype = weight.dtype
        original_device = weight.device
        w = weight.detach().cpu().to(dtype)
        
        # Skip very small matrices
        if min(w.shape) < rank:
            return None
        
        # Compute full SVD (expensive, but necessary for accurate error measurement)
        try:
            U, S, Vt = torch.linalg.svd(w.float(), full_matrices=False)
        except Exception as e:
            print(f"  Warning: SVD failed for shape {w.shape}: {e}")
            return None
        
        # Extract top-r components
        U_r = U[:, :rank].contiguous()  # [out_features, r]
        V_r = Vt[:rank, :].contiguous()  # [r, in_features]
        
        # Reconstruct using double projection: U U^T W V^T V
        # Method 1: Direct computation (can be memory-intensive)
        # W_reconstructed = U_r @ U_r.T @ w @ V_r.T @ V_r
        
        # Method 2: More memory-efficient via intermediate results
        # W @ V^T @ V = W @ V_r^T @ V_r
        temp1 = w @ V_r.T  # [out_features, r]
        temp2 = temp1 @ V_r  # [out_features, in_features]
        
        # U @ U^T @ (...)
        temp3 = U_r.T @ temp2  # [r, in_features]
        W_reconstructed = U_r @ temp3  # [out_features, in_features]
        
        # Compute errors
        error_abs = torch.norm(w - W_reconstructed, p='fro').item()
        w_norm = torch.norm(w, p='fro').item()
        error_rel = error_abs / w_norm if w_norm > 0 else 0.0
        
        # Also compute the "ideal" rank-r approximation error (best possible)
        # Best rank-r approximation: U[:,:r] @ S[:r,:r] @ V[:,:r]
        S_r = torch.diag(S[:rank])
        W_best_rank_r = U_r @ S_r @ V_r
        error_best_abs = torch.norm(w - W_best_rank_r, p='fro').item()
        error_best_rel = error_best_abs / w_norm if w_norm > 0 else 0.0
        
        # Cleanup
        del U, S, Vt, U_r, V_r, temp1, temp2, temp3, W_reconstructed, W_best_rank_r, S_r
        gc.collect()
        
        return {
            'error_abs': error_abs,
            'error_rel': error_rel,
            'error_rel_percent': error_rel * 100,
            'best_rank_r_error_abs': error_best_abs,
            'best_rank_r_error_rel': error_best_rel,
            'best_rank_r_error_rel_percent': error_best_rel * 100,
            'weight_norm': w_norm,
            'shape': list(w.shape),
        }


def analyze_model(
    model_name: str,
    ranks: List[int] = [8, 16, 32, 64, 128],
    dtypes: List[torch.dtype] = [torch.bfloat16, torch.float32],
    layer_limit: int = None,
    use_batched_svd: bool = True,
) -> Dict:
    """
    Analyze reconstruction error for all linear layers in a model.
    
    Args:
        model_name: HuggingFace model identifier
        ranks: List of ranks to test
        dtypes: List of dtypes to test
        layer_limit: Max number of layers to analyze (None = all)
        use_batched_svd: Use GPU-accelerated batched SVD (much faster)
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}\n")
    
    results = {
        'model_name': model_name,
        'ranks': ranks,
        'dtypes': [str(dt) for dt in dtypes],
        'layers': {}
    }
    
    # Load model config first to check size
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        vocab_size = getattr(config, 'vocab_size', 'N/A')
        print(f"Model config loaded. Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Warning loading config: {e}")
        # Continue anyway - config check is not critical
    
    for dtype in dtypes:
        dtype_str = str(dtype).split('.')[-1]
        print(f"\n--- Testing dtype: {dtype_str} ---\n")
        
        try:
            # Load model in the target dtype
            print(f"Loading model in {dtype_str}...")
            
            # Use GPU if available for batched SVD
            if use_batched_svd and torch.cuda.is_available():
                device_map = 'auto'
                print(f"✓ Using GPU for batched SVD acceleration")
            else:
                device_map = 'cpu'
            
            # Try CausalLM first, fall back to generic AutoModel for ViT/BERT etc.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except ValueError:
                # Not a causal LM - use generic AutoModel (works for ViT, BERT, etc.)
                print(f"Not a CausalLM, using AutoModel...")
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            print(f"✓ Model loaded successfully\n")
            
            # Find all linear layers
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append((name, module))
            
            print(f"Found {len(linear_layers)} linear layers")
            if layer_limit:
                linear_layers = linear_layers[:layer_limit]
                print(f"Limiting analysis to first {layer_limit} layers")
            
            # Use batched SVD if enabled and on GPU
            if use_batched_svd and torch.cuda.is_available():
                results['layers'] = analyze_layers_batched(
                    linear_layers, ranks, dtype, dtype_str
                )
            else:
                # Sequential fallback
                results['layers'] = analyze_layers_sequential(
                    linear_layers, ranks, dtype, dtype_str
                )
            
            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error loading/analyzing model in {dtype_str}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute aggregate statistics
    print(f"\n--- Computing Statistics ---\n")
    results['statistics'] = compute_statistics(results)
    
    return results


def analyze_layers_batched(
    linear_layers: List[Tuple[str, torch.nn.Linear]],
    ranks: List[int],
    dtype: torch.dtype,
    dtype_str: str,
) -> Dict:
    """
    Analyze layers using batched GPU-accelerated SVD.
    Groups layers by shape and processes in parallel batches.
    """
    import time
    
    results = {}
    
    # Group layers by shape for batched processing
    shape_groups = {}
    for name, module in linear_layers:
        shape = (module.out_features, module.in_features)
        shape_groups.setdefault(shape, []).append((name, module))
    
    print(f"Grouped into {len(shape_groups)} shape groups for batched SVD\n")
    
    # Process each shape group with batched SVD (mini-batched to avoid OOM)
    MAX_BATCH = 16  # Limit batch size to avoid GPU OOM
    
    for shape, group_layers in shape_groups.items():
        total_layers = len(group_layers)
        print(f"Processing shape {shape}: {total_layers} layers...")
        
        # Process in mini-batches to avoid OOM
        for batch_start in range(0, total_layers, MAX_BATCH):
            batch_end = min(batch_start + MAX_BATCH, total_layers)
            batch_layers = group_layers[batch_start:batch_end]
            batch_size = len(batch_layers)
            
            if total_layers > MAX_BATCH:
                print(f"  Mini-batch [{batch_start+1}-{batch_end}]/{total_layers}...")
            
            # Stack weights into batch
            weight_batch = torch.stack([m.weight.data for _, m in batch_layers])
        
            # Move to GPU and convert to FP32 for SVD
            if weight_batch.device.type != 'cuda':
                weight_batch = weight_batch.cuda()
            
            if weight_batch.dtype in (torch.bfloat16, torch.float16):
                weight_batch = weight_batch.float()
            
            print(f"  Running batched SVD on {weight_batch.device}...")
            start_time = time.time()
            
            # Batched SVD - processes all layers of same shape in parallel
            U_batch, S_batch, V_batch = torch.linalg.svd(weight_batch, full_matrices=False)
            
            torch.cuda.synchronize()
            svd_time = time.time() - start_time
            print(f"  ✓ Batched SVD complete in {svd_time:.2f}s ({svd_time/batch_size:.3f}s per layer)\n")
            
            # Compute errors for each layer in this mini-batch
            for i, (layer_name, _) in enumerate(batch_layers):
                print(f"  [{batch_start+i+1}/{total_layers}] {layer_name}")
                
                # Extract this layer's SVD components
                U_i = U_batch[i]
                S_i = S_batch[i]
                V_i = V_batch[i]
                W_i = weight_batch[i]
                
                results[layer_name] = {}
                
                for rank in ranks:
                    if rank > min(shape):
                        continue
                    
                    # Compute reconstruction errors
                    error_metrics = compute_reconstruction_error_from_svd(
                        W_i, U_i, S_i, V_i, rank
                    )
                    
                    if error_metrics is not None:
                        key = f"{dtype_str}_rank{rank}"
                        results[layer_name][key] = error_metrics
                        
                        print(f"    Rank {rank:3d}: "
                              f"Way4 error: {error_metrics['error_rel_percent']:6.2f}% | "
                              f"Best rank-{rank} error: {error_metrics['best_rank_r_error_rel_percent']:6.2f}%")
            
            # Cleanup batch tensors after each mini-batch
            del weight_batch, U_batch, S_batch, V_batch
            gc.collect()
            torch.cuda.empty_cache()
        
        print()
    
    return results


def analyze_layers_sequential(
    linear_layers: List[Tuple[str, torch.nn.Linear]],
    ranks: List[int],
    dtype: torch.dtype,
    dtype_str: str,
) -> Dict:
    """Sequential layer analysis (fallback when GPU not available)."""
    results = {}
    
    for layer_idx, (layer_name, layer) in enumerate(linear_layers):
        print(f"  [{layer_idx+1}/{len(linear_layers)}] {layer_name} {list(layer.weight.shape)}")
        
        results[layer_name] = {}
        
        for rank in ranks:
            error_metrics = compute_reconstruction_error(
                layer.weight,
                rank=rank,
                dtype=dtype
            )
            
            if error_metrics is not None:
                key = f"{dtype_str}_rank{rank}"
                results[layer_name][key] = error_metrics
                
                print(f"    Rank {rank:3d}: "
                      f"Way4 error: {error_metrics['error_rel_percent']:6.2f}% | "
                      f"Best rank-{rank} error: {error_metrics['best_rank_r_error_rel_percent']:6.2f}%")
    
    return results


def compute_reconstruction_error_from_svd(
    W: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    rank: int,
) -> Dict[str, float]:
    """
    Compute reconstruction error using pre-computed SVD components.
    Much faster than recomputing SVD for each rank.
    
    Args:
        W: Original weight matrix
        U, S, V: Full SVD components (from batched SVD)
        rank: Rank r for truncation
    """
    with torch.no_grad():
        # Extract top-r components
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:rank, :]
        
        # Way 4 reconstruction: U U^T W V^T V
        temp1 = W @ V_r.T @ V_r
        temp2 = U_r @ (U_r.T @ temp1)
        
        # Errors
        error_abs = torch.norm(W - temp2, p='fro').item()
        w_norm = torch.norm(W, p='fro').item()
        error_rel = error_abs / w_norm if w_norm > 0 else 0.0
        
        # Best rank-r approximation: U_r S_r V_r
        W_best = U_r @ torch.diag(S_r) @ V_r
        error_best_abs = torch.norm(W - W_best, p='fro').item()
        error_best_rel = error_best_abs / w_norm if w_norm > 0 else 0.0
        
        return {
            'error_abs': error_abs,
            'error_rel': error_rel,
            'error_rel_percent': error_rel * 100,
            'best_rank_r_error_abs': error_best_abs,
            'best_rank_r_error_rel': error_best_rel,
            'best_rank_r_error_rel_percent': error_best_rel * 100,
            'weight_norm': w_norm,
            'shape': list(W.shape),
        }


def compute_statistics(results: Dict) -> Dict:
    """Compute aggregate statistics across all layers."""
    stats = defaultdict(lambda: defaultdict(list))
    
    for layer_name, layer_results in results['layers'].items():
        for key, metrics in layer_results.items():
            stats[key]['error_rel_percent'].append(metrics['error_rel_percent'])
            stats[key]['best_rank_r_error_rel_percent'].append(metrics['best_rank_r_error_rel_percent'])
    
    # Compute mean, median, min, max for each configuration
    aggregated = {}
    for key, values in stats.items():
        aggregated[key] = {
            'way4_error': {
                'mean': np.mean(values['error_rel_percent']),
                'median': np.median(values['error_rel_percent']),
                'min': np.min(values['error_rel_percent']),
                'max': np.max(values['error_rel_percent']),
                'std': np.std(values['error_rel_percent']),
            },
            'best_rank_r_error': {
                'mean': np.mean(values['best_rank_r_error_rel_percent']),
                'median': np.median(values['best_rank_r_error_rel_percent']),
                'min': np.min(values['best_rank_r_error_rel_percent']),
                'max': np.max(values['best_rank_r_error_rel_percent']),
                'std': np.std(values['best_rank_r_error_rel_percent']),
            }
        }
    
    # Print summary
    for key in sorted(aggregated.keys()):
        print(f"\n{key}:")
        print(f"  Way 4 Error:      {aggregated[key]['way4_error']['mean']:.2f}% ± {aggregated[key]['way4_error']['std']:.2f}% "
              f"(median: {aggregated[key]['way4_error']['median']:.2f}%)")
        print(f"  Best Rank-r Error: {aggregated[key]['best_rank_r_error']['mean']:.2f}% ± {aggregated[key]['best_rank_r_error']['std']:.2f}% "
              f"(median: {aggregated[key]['best_rank_r_error']['median']:.2f}%)")
    
    return aggregated


def main():
    """Main analysis script."""
    
    # Configuration
    models = [
        "google/vit-base-patch16-224",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1", 
        "meta-llama/Meta-Llama-3-8B",
    ]
    
    ranks = [8, 16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    # Limit layers for faster testing (set to None for full analysis)
    layer_limit = None  # Analyze all layers
    
    # Output file
    output_file = "reconstruction_error_analysis.json"
    
    all_results = {}
    
    for model_name in models:
        try:
            results = analyze_model(
                model_name=model_name,
                ranks=ranks,
                dtypes=dtypes,
                layer_limit=layer_limit,
            )
            all_results[model_name] = results
            
        except Exception as e:
            print(f"\n❌ Failed to analyze {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_file}")
    print(f"{'='*80}\n")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✓ Analysis complete!")
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Way 4 Reconstruction Error (Mean % across layers)")
    print(f"{'='*80}\n")
    print(f"{'Model':<30} {'Dtype':<10} {'Rank':<6} {'Way4 Error':<12} {'Best Rank-r':<12}")
    print("-" * 80)
    
    for model_name, model_results in all_results.items():
        if 'statistics' in model_results:
            model_short = model_name.split('/')[-1]
            for key in sorted(model_results['statistics'].keys()):
                dtype_str, rank_str = key.split('_')
                rank = rank_str.replace('rank', '')
                way4_err = model_results['statistics'][key]['way4_error']['mean']
                best_err = model_results['statistics'][key]['best_rank_r_error']['mean']
                print(f"{model_short:<30} {dtype_str:<10} {rank:<6} {way4_err:>10.2f}% {best_err:>10.2f}%")


if __name__ == "__main__":
    main()
