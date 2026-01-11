#!/usr/bin/env python3
"""Performance analysis for RotationalLinearLayer forward pass."""
import torch
import time

def main():
    # Simulate the forward pass - measure each operation
    batch = 32
    seq_len = 512
    in_features = 1152
    out_features = 1024
    r = 128

    device = 'cuda'
    dtype = torch.bfloat16

    # Create test tensors
    x = torch.randn(batch, seq_len, in_features, device=device, dtype=dtype)
    U = torch.randn(out_features, r, device=device, dtype=dtype)
    V = torch.randn(r, in_features, device=device, dtype=dtype)
    S = torch.randn(r, device=device, dtype=torch.float32)  # FP32 S
    R_U = torch.randn(r, r, device=device, dtype=dtype)
    R_V = torch.randn(r, r, device=device, dtype=dtype)
    base_weight = torch.randn(out_features, in_features, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        result = x @ base_weight.T
    torch.cuda.synchronize()

    print(f'Testing with batch={batch}, seq_len={seq_len}, in={in_features}, out={out_features}, r={r}')
    print()

    # Test 1: Base layer (single matmul)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        result = x @ base_weight.T
    torch.cuda.synchronize()
    base_time = (time.perf_counter() - start) / 10 * 1000
    print(f'1. Base layer (x @ W^T): {base_time:.3f} ms')

    # Test 2: Current forward pass (sequential matmuls + dtype conversion)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        target_dtype = x.dtype
        U_cur = U.to(target_dtype)  # ISSUE: redundant .to() calls
        V_cur = V.to(target_dtype)
        R_U_cur = R_U.to(target_dtype)
        R_V_cur = R_V.to(target_dtype)
        S_cur = S.to(target_dtype)  # FP32 -> bf16 conversion EVERY forward!
        
        x_adapted = x @ V_cur.T  
        x_adapted = x_adapted @ R_V_cur.T
        x_adapted = x_adapted * S_cur
        x_adapted = x_adapted @ R_U_cur.T
        x_adapted = x_adapted @ U_cur.T
    torch.cuda.synchronize()
    adapter_time = (time.perf_counter() - start) / 10 * 1000
    print(f'2. Adapter path (5 ops + dtype conv): {adapter_time:.3f} ms')

    # Test 3: Without dtype conversions (tensors already in correct dtype)
    V_fixed = V.to(dtype)
    U_fixed = U.to(dtype)
    R_U_fixed = R_U.to(dtype)
    R_V_fixed = R_V.to(dtype)
    S_fixed = S.to(dtype)  # Pre-converted

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        x_adapted = x @ V_fixed.T  
        x_adapted = x_adapted @ R_V_fixed.T
        x_adapted = x_adapted * S_fixed
        x_adapted = x_adapted @ R_U_fixed.T
        x_adapted = x_adapted @ U_fixed.T
    torch.cuda.synchronize()
    no_conv_time = (time.perf_counter() - start) / 10 * 1000
    print(f'3. Adapter path (no dtype conv): {no_conv_time:.3f} ms')

    # Test 4: Fused approach - precompute R_U @ diag(S) @ R_V
    middle = R_U_fixed @ torch.diag(S_fixed) @ R_V_fixed.T
    middle_UV = U_fixed @ middle @ V_fixed  # Full adapter weight

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        x_adapted = x @ V_fixed.T  
        x_adapted = x_adapted @ middle.T  # Fused R_U @ S @ R_V
        x_adapted = x_adapted @ U_fixed.T
    torch.cuda.synchronize()
    fused_middle = (time.perf_counter() - start) / 10 * 1000
    print(f'4. Fused middle (3 ops): {fused_middle:.3f} ms')

    # Test 5: Fully precomputed W_principal (like standard LoRA merge)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        x_adapted = x @ middle_UV.T  # Single matmul for adapter
    torch.cuda.synchronize()
    fully_fused = (time.perf_counter() - start) / 10 * 1000
    print(f'5. Fully fused (1 op): {fully_fused:.3f} ms')

    print()
    print('=== PERFORMANCE ANALYSIS ===')
    print(f'Current overhead vs base: {adapter_time - base_time:.3f} ms ({(adapter_time/base_time - 1)*100:.1f}% slower)')
    print(f'Removing dtype conv saves: {adapter_time - no_conv_time:.3f} ms')
    print(f'Fusing middle saves: {no_conv_time - fused_middle:.3f} ms')
    print(f'Full fusion saves: {fused_middle - fully_fused:.3f} ms')
    print()
    print(f'BOTTLENECK: Current adapter path is {adapter_time/base_time:.1f}x slower than base layer')
    print()
    print('=== RECOMMENDATIONS ===')
    print('1. Cache S.to(bfloat16) instead of converting every forward')
    print('2. Remove redundant .to() calls for tensors already in correct dtype')
    print('3. Consider fusing middle = R_U @ diag(S) @ R_V.T and caching it')
    print('4. For inference: fully merge adapter into base weights')

if __name__ == '__main__':
    main()
