from fire import Fire
# CHANGED: Import rotational PiSSA instead of PEFT
# from peft import PeftModel
from rotational_pissa_unified import replace_linear_with_rotational_pissa, RotationalPiSSAConfig
from utils import initialize_text_to_text_model
import os
import torch
import torch.nn as nn
# CHANGED: No need for bnb LoRA layers since we use rotational PiSSA
# from peft.tuners.lora.bnb import (
#     Linear8bitLt as LoraLinear8bitLt,
#     Linear4bit as LoraLinear4bit,
# )


def get_float_weight(model: torch.nn.Module):
    model: torch.nn.Linear

    device = model.weight.device
    in_features = model.in_features
    with torch.no_grad():
        I = torch.eye(in_features).to(device)
        w = model(I)
        if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = model.weight.requires_grad
    return w


def replace_A_with_Linear(model: torch.nn.Module, target):
    for name, module in model.named_children():
        if isinstance(module, target):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(module).data
                if bias:
                    new_module.bias.data = (
                        module.bias if module.bias is not None else None
                    )
            setattr(model, name, new_module)

        else:
            replace_A_with_Linear(module, target)


def dequantize(model, dtype):
    # CHANGED: This function is kept for compatibility but may not be needed
    # for rotational PiSSA if we don't use quantization
    try:
        from bitsandbytes.nn import Linear8bitLt as LoraLinear8bitLt
        from bitsandbytes.nn import Linear4bit as LoraLinear4bit
        
        if dtype == "int8":
            target = LoraLinear8bitLt
        elif dtype == "nf4":
            target = LoraLinear4bit
        replace_A_with_Linear(model=model, target=target)
    except ImportError:
        print("Warning: bitsandbytes not available, skipping dequantization")


def merge_rotational_pissa_weights(checkpoint_state_dict, base_model, pissa_config=None):
    """
    Merge Rotational PiSSA adapter weights back into standard linear layer weights.
    
    The forward pass computes:
        Y = X @ W_residual + X @ V^T @ R_V^T @ S @ R_U^T @ U^T * scaling
    
    So the merged weight is:
        W_merged = W_residual + scaling * (U @ R_U @ diag(S) @ R_V^T @ V^T)
        
    Where:
        - W_residual is stored in base_layer.weight
        - U is [r, out_features], V is [r, in_features]
        - R_U, R_V are [r, r] rotation matrices
        - S is [r] singular values
        - scaling = lora_alpha / r
    """
    merged_state_dict = {}
    
    # Default config
    lora_alpha = 8.0 if pissa_config is None else getattr(pissa_config, 'lora_alpha', 8.0)
    
    # Find all rotational PiSSA layers by looking for .S parameters
    pissa_layers = set()
    for key in checkpoint_state_dict.keys():
        if key.endswith('.S'):
            # print(f"Found PiSSA layer: {key}")
            # Extract the layer prefix (e.g., "model.layers.0.self_attn.q_proj")
            layer_prefix = key[:-2]  # Remove ".S"
            # print(f"Layer prefix: {layer_prefix}")
            pissa_layers.add(layer_prefix)
    
    if not pissa_layers:
        print("No Rotational PiSSA layers found. Returning checkpoint as-is.")
        return checkpoint_state_dict
    
    print(f"Found {len(pissa_layers)} Rotational PiSSA layers to merge")
    
    # Process each layer
    merged_layer_count = 0
    for layer_prefix in pissa_layers:
        # Get all components for this layer
        S_key = f"{layer_prefix}.S"
        R_U_key = f"{layer_prefix}.R_U"
        R_V_key = f"{layer_prefix}.R_V"
        U_key = f"{layer_prefix}.U"
        V_key = f"{layer_prefix}.V"
        base_weight_key = f"{layer_prefix}.base_layer.weight"
        base_bias_key = f"{layer_prefix}.base_layer.bias"
        
        # Check if all required components exist
        if not all(k in checkpoint_state_dict for k in [S_key, U_key, V_key, base_weight_key]):
            print(f"Warning: Missing components for {layer_prefix}, skipping")
            continue
        
        # Load components
        S = checkpoint_state_dict[S_key]  # [r]
        U = checkpoint_state_dict[U_key]  # [r, out_features]
        V = checkpoint_state_dict[V_key]  # [r, in_features]
        W_residual = checkpoint_state_dict[base_weight_key]  # [out_features, in_features]
        
        # Get rotation matrices - handle all 4 rotation methods
        r = S.shape[0]
        
        # Keys for way2/way3 parameters
        B_U_key = f"{layer_prefix}.B_U"
        C_U_key = f"{layer_prefix}.C_U"
        B_V_key = f"{layer_prefix}.B_V"
        C_V_key = f"{layer_prefix}.C_V"
        
        # Key for way1 Givens angles (after final step, these are merged into U/V)
        givens_u_key = f"{layer_prefix}.current_givens_u.thetas"
        givens_v_key = f"{layer_prefix}.current_givens_v.thetas"
        
        # Determine rotation method and compute R_U, R_V
        # Way 0: R_U and R_V stored directly
        if R_U_key in checkpoint_state_dict and R_V_key in checkpoint_state_dict:
            R_U = checkpoint_state_dict[R_U_key]
            R_V = checkpoint_state_dict[R_V_key]
            if merged_layer_count == 0:
                print(f"  Using rotation method: way0 (direct R_U, R_V)")
        
        # Way 2/3: Compute R from B, C matrices
        elif B_U_key in checkpoint_state_dict and C_U_key in checkpoint_state_dict:
            B_U = checkpoint_state_dict[B_U_key].to(S.dtype)
            C_U = checkpoint_state_dict[C_U_key].to(S.dtype)
            B_V = checkpoint_state_dict[B_V_key].to(S.dtype)
            C_V = checkpoint_state_dict[C_V_key].to(S.dtype)
            
            # Determine if way2 or way3 based on pissa_config if available
            use_exp = False  # Default to way2
            if pissa_config is not None:
                method = getattr(pissa_config, 'method', 'way2')
                use_exp = (method == 'way3')
            
            if use_exp:
                # Way 3: R = exp(BC^T - CB^T)
                skew_U = B_U @ C_U.T - C_U @ B_U.T
                skew_V = B_V @ C_V.T - C_V @ B_V.T
                R_U = torch.matrix_exp(skew_U.float()).to(S.dtype)
                R_V = torch.matrix_exp(skew_V.float()).to(S.dtype)
                if merged_layer_count == 0:
                    print(f"  Using rotation method: way3 (exp of skew-symmetric)")
            else:
                # Way 2: R = I + BC^T - CB^T
                I = torch.eye(r, dtype=S.dtype, device=S.device)
                R_U = I + B_U @ C_U.T - C_U @ B_U.T
                R_V = I + B_V @ C_V.T - C_V @ B_V.T
                if merged_layer_count == 0:
                    print(f"  Using rotation method: way2 (I + BC^T - CB^T)")
        
        # Way 1: Givens rotations - after training, rotations are merged into U/V
        # So we use identity here (the rotations are already in U/V)
        elif givens_u_key in checkpoint_state_dict:
            # Givens angles present - need to compute rotation matrix from angles
            # This is rare since step_phase() merges rotations into U/V
            # But we support it for completeness
            from rotational_pissa_unified import GivensRotationLayer, generate_givens_pairings
            
            thetas_u = checkpoint_state_dict[givens_u_key]
            thetas_v = checkpoint_state_dict[givens_v_key]
            
            # Get the pairings (same as what was used in training)
            n_layers = pissa_config.n_givens_layers if pissa_config else (r - 1)
            pairings = generate_givens_pairings(r, n_layers)[0]  # Use first layer pairings
            
            # Compute rotation matrices from Givens angles
            R_U = torch.eye(r, device=thetas_u.device, dtype=S.dtype)
            R_V = torch.eye(r, device=thetas_v.device, dtype=S.dtype)
            
            cos_u, sin_u = torch.cos(thetas_u), torch.sin(thetas_u)
            cos_v, sin_v = torch.cos(thetas_v), torch.sin(thetas_v)
            
            for i, (p, q) in enumerate(pairings):
                if p < r and q < r:
                    R_U[p, p], R_U[q, q] = cos_u[i], cos_u[i]
                    R_U[p, q], R_U[q, p] = -sin_u[i], sin_u[i]
                    R_V[p, p], R_V[q, q] = cos_v[i], cos_v[i]
                    R_V[p, q], R_V[q, p] = -sin_v[i], sin_v[i]
            
            if merged_layer_count == 0:
                print(f"  Using rotation method: way1 (Givens angles)")
        
        else:
            # Default: identity (way1 after all rotations merged into U/V)
            R_U = torch.eye(r, dtype=S.dtype, device=S.device)
            R_V = torch.eye(r, dtype=S.dtype, device=S.device)
            if merged_layer_count == 0:
                print(f"  Using rotation method: identity (way1 with rotations merged into U/V)")
        
        # Compute scaling
        # NOTE: In rotational_pissa_unified.py line 497, scaling = 1 (lora_alpha/r is commented out)
        # So we use 1.0 here, not lora_alpha / r
        scaling = 1.0
        
        # Compute W_principal = U @ R_U @ diag(S) @ R_V^T @ V^T
        # Step by step for clarity:
        # Note: U is [r, out_features], V is [r, in_features]
        # W_principal should be [out_features, in_features]
        
        # U^T is [out_features, r]
        # R_U^T is [r, r]
        # diag(S) is [r, r] or just element-wise multiply
        # R_V is [r, r]
        # V is [r, in_features]
        
        # Method: U^T @ R_U^T = [out_features, r]
        #         then multiply by S element-wise in the middle
        #         then @ R_V @ V^T would give [out_features, in_features]
        # Wait, forward is: x @ V^T @ R_V^T @ S @ R_U^T @ U^T
        # So W_principal^T = V^T @ R_V^T @ S @ R_U^T @ U^T
        # W_principal = (V^T @ R_V^T @ S @ R_U^T @ U^T)^T = U @ R_U @ S @ R_V @ V
        # Actually, let's trace forward more carefully:
        # Forward computes: x @ V^T @ R_V^T  (we have R_V.T in code, which is R_V transposed)
        # Wait, the code does:
        #   x_adapted = x_adapted @ R_V.T   which is x @ R_V^T
        #   x_adapted = x_adapted @ R_U.T   which is x @ R_U^T
        # So the chain is: x @ V^T @ R_V^T @ diag(S) @ R_U^T @ U^T
        # This means W = (V^T @ R_V^T @ diag(S) @ R_U^T @ U^T)^T = U @ R_U @ diag(S) @ R_V @ V
        
        with torch.no_grad():
            # Ensure everything is in the same dtype
            dtype = W_residual.dtype
            S = S.to(dtype)
            U = U.to(dtype)  # [out_features, r]
            V = V.to(dtype)  # [r, in_features]
            R_U = R_U.to(dtype)  # [r, r]
            R_V = R_V.to(dtype)  # [r, r]
            
            # The forward pass computes (see rotational_pissa_unified.py lines 824-833):
            # x @ V^T @ R_V^T @ diag(S) @ R_U^T @ U^T
            # 
            # This is equivalent to: Y = X @ W_principal^T
            # where W_principal^T = V^T @ R_V^T @ diag(S) @ R_U^T @ U^T
            # 
            # Taking transpose of both sides:
            # W_principal = (V^T @ R_V^T @ diag(S) @ R_U^T @ U^T)^T
            #             = U @ R_U @ diag(S) @ R_V @ V   (using (ABC)^T = C^T B^T A^T)
            # 
            # Note: diag(S)^T = diag(S) since it's diagonal
            
            # Shapes:
            # U: [out_features, r]
            # R_U: [r, r]
            # diag(S): [r, r]
            # R_V: [r, r]  <-- NOT R_V.T!
            # V: [r, in_features]
            # Result: [out_features, in_features]
            
            # Compute step by step:
            # Step 1: R_U @ diag(S) @ R_V = [r, r]
            middle = R_U @ torch.diag(S) @ R_V  # [r, r] @ [r, r] @ [r, r] = [r, r]
            
            # Step 2: U @ middle @ V = [out_features, r] @ [r, r] @ [r, in_features] = [out_features, in_features]
            W_principal = U @ middle @ V  # [out_features, in_features]
            
            # Merge: W_merged = W_residual + scaling * W_principal
            # Note: In the training code, scaling = 1 (see line 497 in rotational_pissa_unified.py)
            # The lora_alpha/r scaling is commented out
            W_merged = W_residual + scaling * W_principal
        
        # Store merged weight with standard key (remove .base_layer)
        merged_weight_key = f"{layer_prefix}.weight"
        merged_state_dict[merged_weight_key] = W_merged
        
        # Handle bias if present
        if base_bias_key in checkpoint_state_dict:
            merged_bias_key = f"{layer_prefix}.bias"
            merged_state_dict[merged_bias_key] = checkpoint_state_dict[base_bias_key]
        
        merged_layer_count += 1
        
    # Copy over non-PiSSA layers as-is
    for key, value in checkpoint_state_dict.items():
        # Skip PiSSA-specific keys
        skip = False
        for layer_prefix in pissa_layers:
            if key.startswith(layer_prefix + "."):
                skip = True
                break
        
        if not skip:
            merged_state_dict[key] = value
    
    print(f"Successfully merged {merged_layer_count} Rotational PiSSA layers")
    return merged_state_dict


def merge(
    checkpoint: str,
    dtype: str = "bf16",
    model_name="google/gemma-3-1b-it",
    model_type="CausalLM",
    merge_suffix="merged_checkpoint",
    lora_alpha: float = 8.0,
    snapshot_path: str = None,
):
    """
    Merge Rotational PiSSA adapters back into base model weights.
    
    Args:
        checkpoint: Path to the checkpoint directory containing pytorch_model.bin
        dtype: Model dtype (bf16, fp32, etc.)
        model_name: Base model identifier
        model_type: Model type (CausalLM, ConditionalGeneration)
        merge_suffix: Suffix for merged checkpoint directory
        lora_alpha: LoRA alpha scaling factor used during training
        snapshot_path: Path to the snapshot folder containing final_checkpoint.pt (if different from checkpoint)
    """
    print(f"Loading checkpoint from: {checkpoint}")
    
    # Find the checkpoint file
    pytorch_model_path = os.path.join(checkpoint, "pytorch_model.bin")
    
    # final_checkpoint.pt may be in a different folder (snapshot_path)
    if snapshot_path:
        final_checkpoint_path = os.path.join(snapshot_path, "final_checkpoint.pt")
        init_checkpoint_path = os.path.join(snapshot_path, "init_checkpoint.pt")
        print(f"Using snapshot path: {snapshot_path}")
    else:
        final_checkpoint_path = os.path.join(checkpoint, "final_checkpoint.pt")
        init_checkpoint_path = os.path.join(checkpoint, "init_checkpoint.pt")
    
    checkpoint_data = None
    pissa_config = None
    
    # Prefer final_checkpoint.pt from snapshot (has pissa_config metadata)
    if os.path.exists(final_checkpoint_path):
        print(f"Loading from final_checkpoint.pt: {final_checkpoint_path}")
        loaded = torch.load(final_checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            print("loaded was a dict")
            checkpoint_data = loaded['model_state_dict']
            pissa_config = loaded.get('pissa_config')
        else:
            checkpoint_data = loaded
    elif os.path.exists(pytorch_model_path):
        print(f"Loading from pytorch_model.bin")
        checkpoint_data = torch.load(pytorch_model_path, map_location='cpu')
    elif os.path.exists(init_checkpoint_path):
        print(f"Loading from init_checkpoint.pt: {init_checkpoint_path}")
        loaded = torch.load(init_checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            checkpoint_data = loaded['model_state_dict']
            pissa_config = loaded.get('pissa_config')
        else:
            checkpoint_data = loaded
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint} or {snapshot_path}")
    
    # Check if this is a Rotational PiSSA checkpoint (has .S, .U, .V, .R_U, .R_V keys)
    has_pissa_keys = any(k.endswith('.S') for k in checkpoint_data.keys())
    
    if has_pissa_keys:
        print("Detected Rotational PiSSA checkpoint. Merging adapters...")
        
        # Create a dummy config if not loaded
        class SimpleConfig:
            def __init__(self, alpha):
                self.lora_alpha = alpha
        
        if pissa_config is None:
            pissa_config = SimpleConfig(lora_alpha)
            print(f"Using lora_alpha={lora_alpha}")
        else:
            print(f"Using lora_alpha from config: {getattr(pissa_config, 'lora_alpha', lora_alpha)}")
        
        # Merge the weights
        merged_state_dict = merge_rotational_pissa_weights(
            checkpoint_data, 
            base_model=None,  # Not needed for pure merge
            pissa_config=pissa_config
        )
    else:
        print("No Rotational PiSSA layers detected. Using checkpoint as-is.")
        merged_state_dict = checkpoint_data
    
    # Now load the base model and apply merged weights
    print(f"Loading base model: {model_name}")
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, dtype="bf16"
    )
    
    # Load the merged state dict
    print("Loading merged weights into model...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    # Save merged model
    output_path = os.path.join(checkpoint, merge_suffix)
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    Fire(merge)
