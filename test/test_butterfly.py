#!/usr/bin/env python3
"""Test script for butterfly parameterization in rotational_pissa_unified.py"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/chandan/test_current')

from rotational_pissa_unified import (
    ButterflyComponent, ButterflyRotationLayer, 
    RotationalPiSSAConfig, RotationalLinearLayer
)


def test_butterfly_component_orthogonality():
    """Test that each ButterflyComponent produces an orthogonal matrix."""
    print("=" * 60)
    print("Test 1: ButterflyComponent Orthogonality")
    print("=" * 60)
    
    for d in [4, 8, 16]:
        for k in [2, d // 2, d]:
            if k > d:
                continue
            component = ButterflyComponent(d=d, k=k, block_size=2)
            
            # Random initialization
            component.thetas.data = torch.randn_like(component.thetas) * 0.5
            
            R = component()
            
            # Check orthogonality: R @ R.T == I
            identity = torch.eye(d)
            RTR = R @ R.T
            is_orthogonal = torch.allclose(RTR, identity, atol=1e-5)
            
            print(f"  d={d}, k={k}: R@R.T ≈ I? {is_orthogonal}")
            assert is_orthogonal, f"ButterflyComponent(d={d}, k={k}) not orthogonal!"
    
    print("✓ All ButterflyComponent tests passed!\n")


def test_butterfly_rotation_layer_orthogonality():
    """Test that ButterflyRotationLayer produces an orthogonal matrix."""
    print("=" * 60)
    print("Test 2: ButterflyRotationLayer Orthogonality")
    print("=" * 60)
    
    for d in [4, 8, 16, 32, 64, 128]:
        layer = ButterflyRotationLayer(d=d, block_size=2)
        
        # Random initialization
        for component in layer.components:
            component.thetas.data = torch.randn_like(component.thetas) * 0.3
        
        R = layer()
        
        # Check orthogonality
        identity = torch.eye(d)
        RTR = R @ R.T
        is_orthogonal = torch.allclose(RTR, identity, atol=1e-5)
        
        print(f"  d={d}: R@R.T ≈ I? {is_orthogonal} (shape: {R.shape})")
        assert is_orthogonal, f"ButterflyRotationLayer(d={d}) not orthogonal!"
    
    print("✓ All ButterflyRotationLayer orthogonality tests passed!\n")


def test_parameter_count():
    """Verify parameter count matches O(d * log(d) / 2) for block_size=2."""
    print("=" * 60)
    print("Test 3: Parameter Count O(d log d)")
    print("=" * 60)
    
    import math
    
    for d in [4, 8, 16, 32, 64, 128]:
        layer = ButterflyRotationLayer(d=d, block_size=2)
        
        n_params = layer.get_num_parameters()
        n_levels = int(math.log2(d))
        
        # Expected: d/2 * log2(d) angles (each level has d/2 rotations)
        expected = (d // 2) * n_levels
        
        print(f"  d={d}: params={n_params}, expected={expected}, levels={n_levels}")
        assert n_params == expected, f"Expected {expected} params, got {n_params}"
    
    print("✓ All parameter count tests passed!\n")


def test_identity_initialization():
    """Test that butterfly starts at identity when thetas=0."""
    print("=" * 60)
    print("Test 4: Identity Initialization")
    print("=" * 60)
    
    for d in [8, 16, 32]:
        layer = ButterflyRotationLayer(d=d)
        
        R = layer()
        identity = torch.eye(d)
        
        is_identity = torch.allclose(R, identity, atol=1e-6)
        print(f"  d={d}: R ≈ I at init? {is_identity}")
        assert is_identity, f"ButterflyRotationLayer(d={d}) not identity at init!"
    
    print("✓ All identity initialization tests passed!\n")


def test_way1_butterfly_integration():
    """Test that Way 1 with use_butterfly=True works in RotationalLinearLayer."""
    print("=" * 60)
    print("Test 5: Way 1 Butterfly Integration")
    print("=" * 60)
    
    # Create a simple linear layer
    base_layer = nn.Linear(64, 128, bias=False)
    
    # Create config with butterfly mode
    config = RotationalPiSSAConfig(
        r=16,  # Power of 2 for butterfly
        method="way1",
        use_butterfly=True,
        butterfly_block_size=2,
    )
    
    # Create rotational layer
    rotational_layer = RotationalLinearLayer(base_layer, config)
    
    # Test forward pass
    x = torch.randn(2, 64)
    y = rotational_layer(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Has butterfly_u: {hasattr(rotational_layer, 'butterfly_u')}")
    print(f"  Has butterfly_v: {hasattr(rotational_layer, 'butterfly_v')}")
    
    # Get rotation matrices
    R_U, R_V = rotational_layer.get_rotation_matrices()
    print(f"  R_U shape: {R_U.shape}")
    print(f"  R_V shape: {R_V.shape}")
    
    # Check orthogonality
    I = torch.eye(config.r, device=R_U.device, dtype=R_U.dtype)
    is_orthogonal_U = torch.allclose(R_U @ R_U.T, I, atol=1e-5)
    is_orthogonal_V = torch.allclose(R_V @ R_V.T, I, atol=1e-5)
    
    print(f"  R_U orthogonal: {is_orthogonal_U}")
    print(f"  R_V orthogonal: {is_orthogonal_V}")
    
    assert is_orthogonal_U and is_orthogonal_V, "Rotation matrices not orthogonal!"
    
    print("✓ Way 1 butterfly integration test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through butterfly layers."""
    print("=" * 60)
    print("Test 6: Gradient Flow")
    print("=" * 60)
    
    # Create rotational layer with butterfly
    base_layer = nn.Linear(32, 64, bias=False)
    config = RotationalPiSSAConfig(
        r=8,
        method="way1",
        use_butterfly=True,
    )
    rotational_layer = RotationalLinearLayer(base_layer, config)
    
    # Forward pass
    x = torch.randn(2, 32, requires_grad=True)
    y = rotational_layer(x)
    loss = y.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients on butterfly thetas
    has_grad = False
    for name, param in rotational_layer.named_parameters():
        if 'butterfly' in name and 'thetas' in name:
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.6f}")
    
    assert has_grad, "No gradients found on butterfly parameters!"
    print("✓ Gradient flow test passed!\n")


def test_apply_equivalence():
    """Test that apply() and apply_transpose() produce same results as matrix multiplication."""
    print("=" * 60)
    print("Test 7: Apply/Apply_Transpose Equivalence")
    print("=" * 60)
    
    for d in [8, 16, 32, 64, 128]:
        # Create butterfly layer
        layer = ButterflyRotationLayer(d=d, block_size=2)
        
        # Random initialization
        for component in layer.components:
            component.thetas.data = torch.randn_like(component.thetas) * 0.3
        
        # Random input
        batch_size = 4
        x = torch.randn(batch_size, d)
        
        # Get full matrix
        R = layer()  # Get full d×d matrix
        
        # Test apply(): x @ R
        result_matrix = x @ R
        result_apply = layer.apply(x)
        is_equiv_apply = torch.allclose(result_matrix, result_apply, atol=1e-5)
        
        # Test apply_transpose(): x @ R.T
        result_matrix_T = x @ R.T
        result_apply_T = layer.apply_transpose(x)
        is_equiv_apply_T = torch.allclose(result_matrix_T, result_apply_T, atol=1e-5)
        
        max_diff_apply = (result_matrix - result_apply).abs().max().item()
        max_diff_apply_T = (result_matrix_T - result_apply_T).abs().max().item()
        
        print(f"  d={d}: apply() ≈ x@R? {is_equiv_apply} (diff: {max_diff_apply:.2e})")
        print(f"  d={d}: apply_transpose() ≈ x@R.T? {is_equiv_apply_T} (diff: {max_diff_apply_T:.2e})")
        
        assert is_equiv_apply, f"apply() not equivalent for d={d}!"
        assert is_equiv_apply_T, f"apply_transpose() not equivalent for d={d}!"
    
    print("✓ Apply equivalence tests passed!\n")


if __name__ == "__main__":
    print("\n🦋 Running Butterfly Parameterization Tests\n")
    
    test_butterfly_component_orthogonality()
    test_butterfly_rotation_layer_orthogonality()
    test_parameter_count()
    test_identity_initialization()
    test_way1_butterfly_integration()
    test_gradient_flow()
    test_apply_equivalence()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
