
import torch
import copy
import matplotlib.pyplot as plt

def test_bf16_update_issue():
    """
    Demonstrates the issue: Small updates to large values in BF16 are lost due to precision limits.
    """
    print("=== Test 1: The Problem (BF16 Precision) ===")
    
    # Simulate a large singular value often seen in LLMs
    # BF16 has 7 mantissa bits => ~3 significant decimal digits
    S_val = 14.5
    
    # BF16 representation
    S_bf16 = torch.tensor([S_val], dtype=torch.bfloat16)
    
    # Typical AdamW update for LoRA/pissa might be very small
    # lr=2e-5, grad~=0.1 => update ~= 2e-6
    update_val = 2e-5 # Learning rate magnitude
    
    print(f"Initial S (BF16): {S_bf16.item()}")
    print(f"Desired Update:   {update_val}")
    
    # Simulate update
    S_new = S_bf16 + torch.tensor([update_val], dtype=torch.bfloat16)
    
    print(f"Updated S (BF16): {S_new.item()}")
    
    if S_new.item() == S_bf16.item():
        print("❌ FAILURE: Update was completely lost due to BF16 precision!")
    else:
        print("✅ SUCCESS: Update was applied.")
        
    # Calculate relative error
    true_val = S_val + update_val
    actual_val = S_new.item()
    print(f"True Value: {true_val:.8f}")
    print(f"BF16 Value: {actual_val:.8f}")
    print(f"Error: {abs(true_val - actual_val):.8f}")
    print()

def test_normalization_solution():
    """
    Tests the proposed solution: Normalize S by a factor (e.g., its mean or max) 
    so values are ~0.1 or ~1.0, apply scaling factor separately.
    """
    print("=== Test 2: The Solution (Normalization) ===")
    
    S_val = 14.5
    normalization_factor = 100.0 # Scale down by 100
    
    # Normalized S
    S_norm_val = S_val / normalization_factor # 0.145
    S_norm_bf16 = torch.tensor([S_norm_val], dtype=torch.bfloat16)
    
    # We must also scale the update!
    # If Y = (S_norm * scale) * X, then dL/dS_norm = dL/d(S) * scale
    # Effectively, the gradient flows to S_norm directly. 
    # But does the optimizer step work better?
    
    update_val = 2e-5 
    # If we apply the update to the normalized value, is it preserved?
    # Note: In the real model, gradients would be scaled, but we just want to see 
    # if BF16 can represent (0.145 + small_update) better than (14.5 + small_update).
    
    # Let's say we want to apply the SAME relative update magnitude
    # True update to S is 2e-5.
    # Update to S_norm should be 2e-5 / 100 = 2e-7? 
    # ACTUALLY NO:
    # We want to represent the value `(14.5 + 0.00002)`.
    # Normalized: `(14.5 + 0.00002) / 100` = `0.145 + 0.0000002`.
    
    # Wait, if we scale down S, the update we need to represent becomes even smaller!
    # Does this help?
    # Floating point precision is relative.
    # 14.5 has exponent 3 (2^3=8). 
    # 0.145 has exponent -3 (2^-3=0.125).
    # The gap between representable numbers scales with the exponent.
    
    # Let's verify precision resolution.
    
    # Case A: Value ~ 16.0
    val_large = torch.tensor([16.0], dtype=torch.bfloat16)
    next_up_large = torch.nextafter(val_large, torch.tensor([100.0], dtype=torch.bfloat16))
    gap_large = next_up_large - val_large
    print(f"BF16 Resolution at 16.0: {gap_large.item():.8f}")
    
    # Case B: Value ~ 0.16
    val_small = torch.tensor([0.16], dtype=torch.bfloat16)
    next_up_small = torch.nextafter(val_small, torch.tensor([100.0], dtype=torch.bfloat16))
    gap_small = next_up_small - val_small
    print(f"BF16 Resolution at 0.16: {gap_small.item():.8f}")
    
    # Check if our update fits
    update_needed = 2e-5
    
    print(f"\nUpdate needed: {update_needed:.8f}")
    
    if gap_large > update_needed:
        print(f"❌ At 16.0, gap ({gap_large.item():.8f}) > update ({update_needed:.8f}) -> Update LOST")
    else:
        print(f"✅ At 16.0, gap ({gap_large.item():.8f}) < update ({update_needed:.8f}) -> Update SAVED")
        
    # For normalized version:
    # If we factor out 100, S becomes 0.145. 
    # BUT gradients/updates also scale.
    # Effective weight = S_param * 100
    # Gradient w.r.t S_param = Gradient w.r.t Weight * 100
    # Update step = lr * grad * 100
    # So the update step applied to S_param is 100x LARGER than the update applied to Weight?
    #
    # Let's trace the math:
    # W = S * C (where C=100)
    # y = Wx = S*C*x
    # dL/dS = dL/dy * C * x
    # delta_S = lr * dL/dS = lr * (dL/dW * C) = (lr * dL/dW) * C
    #
    # So yes! If we scale S down by factor C, the update to S scales UP by factor C.
    #
    # Example:
    # S_real = 14.5
    # Desired shift in S_real = 2e-5.
    #
    # Normalized Param: P = S_real / 100 = 0.145
    # Constant Multiplier: C = 100
    # Effective S = P * C = 14.5
    #
    # Desired change in effective S = 2e-5
    # New Effective S = 14.50002
    # New P = 14.50002 / 100 = 0.1450002
    # Change in P = 0.0000002 (2e-7)
    #
    # Wait... if P is smaller, and the change is smaller, do we gain anything?
    #
    # Let's check the ratio of (Change / Value).
    # Large: 2e-5 / 14.5 = 1.37e-6
    # Small: 2e-7 / 0.145 = 1.37e-6
    # The relative precision required is identical. Floating point stores relative precision (mantissa).
    #
    # HOWEVER, let's look at the resolution.
    # Resolution at 0.145 is ~9.7e-4 * 2^-7? 
    # Gap at 0.16 is 0.00097656
    #
    # Let's run the simulation.
    
    print("\n--- Simulation with Scaling ---")
    factor = 100.0
    S_param = torch.tensor([14.5 / factor], dtype=torch.bfloat16)  # 0.145
    
    # We want effective S to move by 2e-5
    # So S_param must move by 2e-5 / 100 = 2e-7
    update_to_param = 2e-5 / factor
    
    print(f"S_param (BF16): {S_param.item():.10f}")
    print(f"Update to param: {update_to_param:.10f}")
    print(f"Resolution at S_param: {gap_small.item():.10f}")
    
    S_param_new = S_param + torch.tensor([update_to_param], dtype=torch.bfloat16)
    
    change = S_param_new.item() - S_param.item()
    print(f"Actual change in S_param: {change:.10f}")
    
    effective_change = change * factor
    print(f"Effective change in S: {effective_change:.10f}")
    
    if effective_change == 0:
         print("❌ Normalization FAILED to capture the update.")
    else:
         print("✅ Normalization WORKED?")

if __name__ == "__main__":
    test_bf16_update_issue()
    test_normalization_solution()
