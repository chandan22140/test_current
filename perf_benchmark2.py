#!/usr/bin/env python3
"""Performance analysis for dtype conversions and orthogonality loss."""
import torch
import time

def main():
    device = 'cuda'
    r = 128
    n_layers = 182  # Number of adapter layers
    
    # Create test tensors
    R_U = torch.randn(r, r, device=device, dtype=torch.bfloat16, requires_grad=True)
    R_V = torch.randn(r, r, device=device, dtype=torch.bfloat16, requires_grad=True)
    S = torch.randn(r, device=device, dtype=torch.float32, requires_grad=True)
    
    print(f'Testing with r={r}, n_layers={n_layers}')
    print()
    
    # Test 1: S.to(bf16) conversion overhead
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        S_bf16 = S.to(torch.bfloat16)
    torch.cuda.synchronize()
    s_conv_time = (time.perf_counter() - start) / 100 * 1000
    print(f'1. S.to(bf16) per call: {s_conv_time:.4f} ms')
    print(f'   Per forward (x{n_layers} layers): {s_conv_time * n_layers:.2f} ms')
    
    # Test 2: Orthogonality loss with FP32 conversion
    I = torch.eye(r, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        R_U_fp32 = R_U.float()
        R_V_fp32 = R_V.float()
        loss_U = torch.norm(R_U_fp32 @ R_U_fp32.T - I, p='fro') ** 2
        loss_V = torch.norm(R_V_fp32 @ R_V_fp32.T - I, p='fro') ** 2
        loss = loss_U + loss_V
    torch.cuda.synchronize()
    ortho_fp32_time = (time.perf_counter() - start) / 100 * 1000
    print(f'2. Ortho loss (bf16->fp32 conv): {ortho_fp32_time:.4f} ms')
    print(f'   Per forward (x{n_layers} layers): {ortho_fp32_time * n_layers:.2f} ms')
    
    # Test 3: Orthogonality loss in native bf16
    I_bf16 = torch.eye(r, device=device, dtype=torch.bfloat16)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        loss_U = torch.norm(R_U @ R_U.T - I_bf16, p='fro') ** 2
        loss_V = torch.norm(R_V @ R_V.T - I_bf16, p='fro') ** 2
        loss = loss_U + loss_V
    torch.cuda.synchronize()
    ortho_bf16_time = (time.perf_counter() - start) / 100 * 1000
    print(f'3. Ortho loss (native bf16): {ortho_bf16_time:.4f} ms')
    print(f'   Per forward (x{n_layers} layers): {ortho_bf16_time * n_layers:.2f} ms')
    
    # Test 4: torch.det in FP32
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        R_U_fp32 = R_U.float()
        det = torch.det(R_U_fp32)
        loss = (det - 1.0) ** 2
    torch.cuda.synchronize()
    det_time = (time.perf_counter() - start) / 100 * 1000
    print(f'4. torch.det (fp32): {det_time:.4f} ms')
    print(f'   Per forward (x{n_layers} layers): {det_time * n_layers:.2f} ms')
    
    # Test 5: torch.compile benefit (if available)
    print()
    print('=== SUMMARY ===')
    print(f'S dtype conversion overhead: {s_conv_time * n_layers:.1f} ms/forward')
    print(f'Ortho loss (fp32): {ortho_fp32_time * n_layers:.1f} ms/forward')
    print(f'Ortho loss (bf16): {ortho_bf16_time * n_layers:.1f} ms/forward')
    print(f'Savings if using bf16 for ortho: {(ortho_fp32_time - ortho_bf16_time) * n_layers:.1f} ms/forward')
    print()
    
    # Test 6: torch.compile demo
    print('=== torch.compile EXPLANATION ===')
    print("""
torch.compile() is PyTorch 2.0+ feature that JIT-compiles your model:
- Fuses multiple operations into single GPU kernels
- Eliminates Python overhead
- Can optimize dtype conversions

Example usage:
    model = torch.compile(model)  # Wrap your model
    # First forward is slow (compiling), subsequent are faster

For RotationalLinearLayer:
    @torch.compile
    def forward(self, x):
        ...
    
This can automatically fuse:
    x @ V.T @ R_V.T * S @ R_U.T @ U.T
into fewer GPU kernel launches.
""")

if __name__ == '__main__':
    main()
