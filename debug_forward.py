#!/usr/bin/env python
"""
Debug script to test rotational PiSSA forward/backward pass in isolation.
Run this to verify gradients flow correctly before running full training.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/chandan/DL_Quantization/backup_131/DL_Quantization/PiSSA/test_current')

from rotational_pissa_unified import RotationalLinearLayer, RotationalPiSSAConfig

def test_rotational_forward_backward():
    print("=" * 60)
    print("Testing Rotational PiSSA Forward/Backward")
    print("=" * 60)
    
    # Use bf16 to match training config
    torch.set_default_dtype(torch.bfloat16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create a simple linear layer (similar to model layers)
    in_features, out_features = 640, 1024
    linear = nn.Linear(in_features, out_features, bias=False)
    linear = linear.to(device)
    
    # Create config matching your training
    config = RotationalPiSSAConfig(
        r=32,
        lora_alpha=8,
        method="way0",
        orthogonality_reg_weight=0.0,
        init_identity=True,
        freeze_singular_values=False,
    )
    
    print(f"\nConfig: {config}")
    
    # Create rotational layer
    rot_layer = RotationalLinearLayer(linear, config, adapter_name="test")
    rot_layer = rot_layer.to(device)
    
    print("\n--- Parameter Summary ---")
    for name, p in rot_layer.named_parameters():
        print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, dtype={p.dtype}")
    
    print("\n--- Buffer Summary ---")
    for name, b in rot_layer.named_buffers():
        print(f"  {name}: shape={tuple(b.shape)}, dtype={b.dtype}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    batch_size = 2
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)
    
    y = rot_layer(x)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    print(f"Output has NaN: {torch.isnan(y).any()}")
    print(f"Output has Inf: {torch.isinf(y).any()}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    # Test backward pass
    print("\n--- Backward Pass ---")
    loss = y.sum()
    print(f"Loss: {loss.item():.6f}, dtype: {loss.dtype}")
    
    loss.backward()
    
    print("\n--- Gradient Check ---")
    for name, p in rot_layer.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                has_nan = torch.isnan(p.grad).any().item()
                has_inf = torch.isinf(p.grad).any().item()
                print(f"  {name}: grad_norm={grad_norm:.6f}, has_nan={has_nan}, has_inf={has_inf}")
            else:
                print(f"  {name}: grad is None!")
    
    print("\n--- Orthogonality Loss ---")
    ortho_loss = rot_layer.get_orthogonality_loss()
    print(f"Orthogonality loss: {ortho_loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_rotational_forward_backward()
