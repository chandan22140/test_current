"""
Unified Rotational PiSSA Implementation
======================================

This module provides a complete implementation of Rotational PiSSA, merging PiSSA 
with the Method of Rotation of SVD Subspaces.

Mathematical Framework (Following PiSSA Paper):
1. Original weight W is decomposed: W = U @ S @ V^T
2. Split into principal and residual components based on rank r
3. W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized, frozen)
4. W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T (trainable via rotations)
5. Final computation: Y = X @ (W_residual + W_principal)

Trainable Components:
- R_U, R_V: Rotation matrices (4 different parameterization ways)
- S[:r,:r]: Principal singular values (optionally trainable)

Frozen Components:
- W_residual: NF4 quantized residual matrix
- U[:,:r], V[:,:r]: Principal singular vectors (optionally NF4 quantized)

Rotation Parameterization Methods:
- Way 0: Direct optimization with orthogonality regularization
- Way 1: Greedy sequential Givens rotations  
- Way 2: Low-rank skew-symmetric perturbation I + BC^T - CB^T
- Way 3: Exponential map of skew-symmetric matrix exp(BC^T - CB^T)
"""

import os
import sys
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict, Any, Literal, Union, List, Tuple
from dataclasses import dataclass, field


# ============================================================================
# MEMORY-EFFICIENT TENSOR UTILITIES
# ============================================================================

def to_free(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert tensor to different dtype/device AND free the original tensor's memory.
    
    Unlike tensor.to() which creates a new tensor but leaves the original in memory
    until garbage collection, this function explicitly frees the original tensor
    immediately after conversion.
    
    Args:
        tensor: The input tensor to convert (will be freed after conversion)
        dtype: Target dtype (e.g., torch.float16). If None, keeps original dtype.
        device: Target device (e.g., "cuda:0"). If None, keeps original device.
        
    Returns:
        New tensor with converted dtype/device. Original tensor memory is freed.
        
    Example:
        # Instead of:
        #   x = x.to(torch.float16)  # Original fp32 tensor still in memory!
        #   del old_x  # Would need to save reference first
        #   gc.collect()
        #   torch.cuda.empty_cache()
        
        # Use:
        x = to_free(x, dtype=torch.float16)  # Original freed automatically
        
    Warning:
        After calling to_free(tensor, ...), the original tensor reference is invalid.
        Only use the returned tensor.
    """
    # Build kwargs for .to() call
    to_kwargs = {}
    if dtype is not None:
        to_kwargs['dtype'] = dtype
    if device is not None:
        to_kwargs['device'] = device
    
    # If no conversion needed, return original tensor (no copy)
    if not to_kwargs:
        return tensor
    
    # Check if conversion is actually needed
    needs_dtype_change = dtype is not None and tensor.dtype != dtype
    needs_device_change = device is not None and tensor.device != device
    
    if not needs_dtype_change and not needs_device_change:
        return tensor  # No conversion needed, return original
    
    # Create new tensor with converted dtype/device
    new_tensor = tensor.to(**to_kwargs)
    
    # Explicitly free the original tensor's memory
    # We use tensor.data to access underlying storage and clear it
    # Then delete the tensor and force garbage collection
    del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return new_tensor


def to_free_inplace(
    obj: object,
    attr_name: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Convert an object's tensor attribute in-place and free the original memory.
    
    This is useful when you have a tensor stored as an attribute (e.g., self.weight)
    and want to convert it while freeing the original.
    
    Args:
        obj: Object containing the tensor attribute
        attr_name: Name of the attribute (e.g., "weight")
        dtype: Target dtype
        device: Target device
        
    Example:
        # Instead of:
        #   old = self.weight
        #   self.weight = self.weight.to(torch.float16)
        #   del old
        #   gc.collect()
        #   torch.cuda.empty_cache()
        
        # Use:
        to_free_inplace(self, "weight", dtype=torch.float16)
    """
    old_tensor = getattr(obj, attr_name)
    new_tensor = to_free(old_tensor, dtype=dtype, device=device)
    setattr(obj, attr_name, new_tensor)


# ============================================================================
# CORE CONFIGURATION
# ============================================================================
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    Linear4bit = None

@dataclass
class RotationalPiSSAConfig:
    """Configuration for Rotational PiSSA adapter."""
    
    # Core parameters
    r: int = 16                                   # Rank
    lora_alpha: float = 16.0                     # Scaling factor
    lora_dropout: float = 0.0                    # Should be 0 for PiSSA
    
    # Rotation parameterization method
    method: Literal["way0", "way1", "way2", "way3"] = "way0"
    
    # Way 0 specific parameters (Direct orthogonality regularization)
    orthogonality_reg_weight: float = 1e-4      # Weight for orthogonality regularization loss (frobenius only)
    regularization_type: str = "frobenius"    # frobenius (recommended - fast), determinant, log_determinant
    
    # Way 1 specific parameters (Sequential Givens rotations)
    n_givens_layers: Optional[int] = None        # Number of Givens layers (default: r-1)
    steps_per_phase: int = 100                   # Steps to train each Givens layer
    total_cycles: int = 3                        # Total cycles through all layers
    
    # Way 2/3 specific parameters (Low-rank methods)
    low_rank_r: int = 4                          # Low rank for B,C matrices in ways 2&3
    
    # General parameters
    init_identity: bool = True                # Initialize rotation matrices as identity
    freeze_singular_values: bool = False         # Whether to freeze S matrix
    s_dtype_fp32: bool = True                    # Store S in FP32 for precision (bfloat16 rounds small updates to 0)
    
    # Quantization parameters
    quantize_residual: bool = False               # Whether to NF4 quantize W_residual
    quantize_base_components: bool = False       # Whether to NF4 quantize U and V^T

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class ResidualLinear4bit(nn.Module):
    """
    4-bit quantized linear layer for W_residual following QLoRA pattern.
    
    This properly handles NF4 quantization by:
    1. Storing weight as Params4bit (NF4 quantized, ~4x memory reduction)
    2. Using dequantize_4bit for forward pass (reconstructs fp16 weight on-the-fly)
    3. Keeping bias in fp16/fp32 (not quantized, negligible memory)
    
    Memory efficiency:
    - Weight is stored in packed NF4 format (~4 bits per element)
    - quant_state stores quantization metadata (scales, zero points)
    - Forward pass dequantizes to fp16, computes matmul, then discards dequantized weight
    - No intermediate fp16 weight is stored, only the quantized representation
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        weight_data: The weight tensor to quantize (required, not optional)
        bias_data: Optional bias tensor (will be stored in fp16, not quantized)
        compute_dtype: Dtype for forward pass computation (default: float16)
        quant_type: Quantization type, "nf4" for normalized float 4-bit (default)
        compress_statistics: Whether to compress quantization statistics (default: True)
        device: Target device for the layer
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_data: torch.Tensor,
        bias_data: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
        compress_statistics: bool = True,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        
        # ====== WEIGHT QUANTIZATION ======
        # Directly create Params4bit from weight_data, no intermediate nn.Parameter allocation
        # This saves memory by avoiding: empty tensor -> copy weight -> quantize -> delete old
        if HAS_BITSANDBYTES:
            try:
                from bitsandbytes.nn import Params4bit
                
                # Params4bit expects fp16 input, will internally quantize to NF4
                # The resulting tensor is ~4x smaller than fp16
                # quant_state stores: absmax (scales), code (NF4 lookup table), blocksize, dtype
                self.weight = Params4bit(
                    weight_data.to(dtype=torch.float16, device=device),
                    requires_grad=False,  # Frozen, not trainable
                    quant_type=quant_type,  # "nf4" = normalized float 4-bit
                    compress_statistics=compress_statistics,  # Compress absmax for more savings
                )
                # Immediately free the original weight_data to reclaim memory
                del weight_data
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Warning: NF4 quantization failed, falling back to fp16: {e}")
                # Fallback: store as regular fp16 parameter (no quantization)
                self.weight = nn.Parameter(
                    weight_data.to(dtype=torch.float16, device=device),
                    requires_grad=False
                )
        else:
            # bitsandbytes not available, use fp16
            self.weight = nn.Parameter(
                weight_data.to(dtype=torch.float16, device=device),
                requires_grad=False
            )
        
        # ====== BIAS (not quantized) ======
        # Bias is small (out_features,) so quantization overhead not worth it
        if bias_data is not None:
            self.bias = nn.Parameter(
                bias_data.to(dtype=torch.float16, device=device),
                requires_grad=False  # Frozen along with weight
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResidualLinear4bit.
        Handles both quantized (NF4/Params4bit) and non-quantized weights.

        Quantization logic:
        - If weight is quantized (Params4bit with quant_state), use bitsandbytes dequantization to recover the full-precision weight for matmul.
        - This avoids shape errors from using F.linear directly on packed/flattened quantized weights.
        - If dequantization fails (rare), fallback to creating a temporary Linear4bit layer and use its forward method.
        - If weight is not quantized, use standard F.linear.
        """
        inp_dtype = x.dtype

        # Ensure input is in compute_dtype (usually float16/bfloat16)
        if x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)

        # --- Quantized weight path ---
        # Params4bit is a bitsandbytes class for 4-bit quantized weights
        # quant_state stores quantization metadata (scales, zero points, etc)
        if HAS_BITSANDBYTES and hasattr(self.weight, 'quant_state') and self.weight.quant_state is not None:
            try:
                # Dequantize the packed 4-bit weight to full precision
                # This reconstructs the original 2D weight matrix for matmul
                from bitsandbytes.functional import dequantize_4bit
                weight_deq = dequantize_4bit(
                    self.weight.data, 
                    self.weight.quant_state
                ).to(self.compute_dtype)

                # Bias must also match compute_dtype
                bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
                # Use standard F.linear with dequantized weight
                out = F.linear(x, weight_deq, bias)

            except Exception as e:
                print(f"⚠️ Warning: dequantize_4bit failed ({e}), trying Linear4bit wrapper...")
                # Fallback: Create a temporary Linear4bit layer
                # Linear4bit is a bitsandbytes nn.Module that handles quantized weights internally
                try:
                    linear_4bit = Linear4bit(
                        self.in_features, 
                        self.out_features,
                        bias=self.bias is not None,
                        compute_dtype=self.compute_dtype,
                        quant_type="nf4",
                    )
                    # Assign quantized weight and bias
                    linear_4bit.weight = self.weight
                    if self.bias is not None:
                        linear_4bit.bias = self.bias
                    linear_4bit = linear_4bit.to(x.device)
                    # Use Linear4bit's forward method
                    out = linear_4bit(x)
                except Exception as e2:
                    raise RuntimeError(f"All 4-bit forward methods failed: dequantize={e}, Linear4bit={e2}")
        else:
            # --- Non-quantized weight path ---
            # Use standard F.linear for fp16/fp32 weights
            out = F.linear(x, self.weight, self.bias)

        # Return output in original input dtype for consistency
        return out.to(inp_dtype)


def dequantize_params4bit(params: torch.Tensor) -> torch.Tensor:
    """
    Dequantize Params4bit tensor to fp16/fp32 for use in custom operations.
    
    Args:
        params: Params4bit tensor with quant_state
        
    Returns:
        Dequantized fp16 tensor
    """
    if not HAS_BITSANDBYTES or not hasattr(params, 'quant_state'):
        return params
    
    try:
        from bitsandbytes.functional import dequantize_4bit
        return dequantize_4bit(params, params.quant_state)
    except Exception as e:
        print(f"Warning: Dequantization failed: {e}")
        return params

def generate_givens_pairings(r: int, n_layers: int) -> List[List[Tuple[int, int]]]:
    """
    Generate disjoint pairings for Givens rotations using round-robin tournament.
    
    Args:
        r: Dimension of rotation space
        n_layers: Number of rotation layers to generate
        
    Returns:
        List of rotation layers, each containing disjoint pairs
    """
    if r <= 1:
        return []
        
    nodes = list(range(r))
    
    # Handle odd dimensions by adding dummy node
    if r % 2 == 1:
        nodes.append(-1)  # Dummy node
    
    fixed_node = nodes[0]
    rotating_nodes = nodes[1:]
    
    all_pairings = []
    for phase in range(min(n_layers, len(nodes) - 1)):
        current_pairs = []
        
        # Pair fixed node with first rotating node
        if rotating_nodes[0] != -1:  # Skip dummy
            current_pairs.append((fixed_node, rotating_nodes[0]))
        
        # Pair remaining nodes horizontally
        # After pairing position 0, we have positions 1 to n-1 left (even count)
        # We need to pair them: (1, n-1), (2, n-2), ..., up to the middle
        for i in range(1, (len(rotating_nodes) + 1) // 2):
            if rotating_nodes[i] != -1 and rotating_nodes[-i] != -1:
                current_pairs.append((rotating_nodes[i], rotating_nodes[-i]))
        
        all_pairings.append(current_pairs)
        
        # Rotate for next phase
        rotating_nodes = [rotating_nodes[-1]] + rotating_nodes[:-1]
    
    return all_pairings

# ============================================================================
# ROTATION LAYER IMPLEMENTATIONS
# ============================================================================

class GivensRotationLayer(nn.Module):
    """A layer representing parallel Givens rotations on disjoint axis pairs."""
    
    def __init__(self, dimension: int, pairs: List[Tuple[int, int]]):
        super().__init__()
        self.dimension = dimension
        self.pairs = pairs
        
        # Trainable rotation angles for each pair
        self.thetas = nn.Parameter(torch.zeros(len(pairs)))
    
    def forward(self) -> torch.Tensor:
        """Construct the rotation matrix from Givens rotations."""
        R = torch.eye(self.dimension, device=self.thetas.device, dtype=self.thetas.dtype)
        
        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)
        
        for i, (p, q) in enumerate(self.pairs):
            if p < self.dimension and q < self.dimension:  # Valid indices
                R[p, p] = cos_thetas[i]
                R[q, q] = cos_thetas[i]
                R[p, q] = -sin_thetas[i]
                R[q, p] = sin_thetas[i]
        
        return R
    
    def reset(self):
        """Reset all rotation angles to zero (identity rotation)."""
        with torch.no_grad():
            self.thetas.zero_()

# ============================================================================
# MAIN ROTATIONAL PISSA LAYER
# ============================================================================

class RotationalLinearLayer(nn.Module):
    """
    A Linear layer with rotational PiSSA adaptation.
    
    Implementation follows PiSSA paper methodology:
    1. SVD decomposition: W = U @ S @ V^T
    2. Split into principal and residual components
    3. W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized and frozen)
    4. W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T (trainable via rotations)
    5. Final output: Y = X * (W_residual + W_principal)
    
    Where:
    - U[:,:r], V[:,:r]^T are frozen (optionally NF4 quantized)
    - R_U @ S[:r,:r] @ R_V^T is the trainable rotation part
    - R_U, R_V are learnable rotation matrices
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        pissa_config: RotationalPiSSAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.pissa_config = pissa_config
        self.adapter_name = adapter_name
        self.r = pissa_config.r
        
        # Store base layer properties
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Perform SVD decomposition on base weight
        self._initialize_svd_components()
        
        # Initialize rotation matrices based on selected method
        self._initialize_rotation_matrices()
        
        # Dropout layer
        if pissa_config.lora_dropout > 0:
            self.dropout = nn.Dropout(pissa_config.lora_dropout)
        else:
            self.dropout = nn.Identity()
            
        # Scaling factor
        self.scaling = 1
        # self.scaling = pissa_config.lora_alpha / pissa_config.r
        
        # Training state for Way 1 (sequential Givens)
        if pissa_config.method == "way1":
            self.current_layer_index = 0
            self.current_cycle = 0
            self.step_count = 0
    
    def _initialize_svd_components(self):
        """Perform SVD on base layer weight and implement PiSSA methodology.
        
        Following PiSSA paper:
        1. Decompose W = U @ S @ V^T
        2. Split into principal {U[:,:r], S[:r,:r], V[:,:r]} and residual {U[:,r:], S[r:,r:], V[:,r:]}
        3. Create W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (NF4 quantized, frozen)
        4. Store principal components for rotational adapter: U[:,:r], S[:r,:r], V[:,:r]
        5. Adapter computes W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T
        """
        import gc
        
        with torch.no_grad():
            original_weight = self.base_layer.weight.data.clone()
            
            # SVD doesn't support BF16/FP16 on CUDA - convert to FP32 for SVD
            original_dtype = original_weight.dtype
            original_device = original_weight.device  # Preserve device (GPU)
            if original_weight.dtype in (torch.bfloat16, torch.float16):
                original_weight = original_weight.float()
            
            # Perform SVD: W = U @ S @ V^T
            U, S, V = torch.linalg.svd(original_weight, full_matrices=False)
            
            # Immediately delete original_weight copy to free VRAM
            del original_weight
            gc.collect()
            torch.cuda.empty_cache()
            
            # Split into principal (top-r) and residual components
            U_principal = U[:, :self.r].contiguous()  # [out_features, r]
            U_residual = U[:, self.r:].contiguous()   # [out_features, remaining]
            
            S_principal = S[:self.r].contiguous()     # [r]
            S_residual = S[self.r:].contiguous()      # [remaining]
            
            V_principal = V[:self.r, :].contiguous()  # [r, in_features]
            V_residual = V[self.r:, :].contiguous()   # [remaining, in_features]
            
            # Delete full U, S, V to free VRAM
            del U, S, V
            gc.collect()
            torch.cuda.empty_cache()
            
            # === RESIDUAL COMPONENTS (FROZEN, OPTIONALLY QUANTIZED) ===
            # Compute residual matrix: W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T
            if U_residual.shape[1] > 0:  # Check if there are residual components
                S_residual_diag = torch.diag(S_residual)
                W_residual = U_residual @ S_residual_diag @ V_residual
                
                # Delete temporary residual components immediately
                del S_residual_diag
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # If r equals the rank of the matrix, residual is zero
                W_residual = torch.zeros_like(self.base_layer.weight.data)
            
            # Delete U_residual, S_residual, V_residual to free VRAM
            del U_residual, S_residual, V_residual
            gc.collect()
            torch.cuda.empty_cache()
            
            # ========== Quantize W_residual (QLoRA pattern) ==========
            if self.pissa_config.quantize_residual and HAS_BITSANDBYTES:
                print(f"      Quantizing W_residual to NF4 (QLoRA pattern)")
                
                # Get bias data if exists (before we replace base_layer)
                bias_data = self.base_layer.bias.data.clone() if self.base_layer.bias is not None else None
                
                # Create ResidualLinear4bit with weight_data directly
                # This quantizes W_residual immediately in __init__, no intermediate allocation
                quantized_layer = ResidualLinear4bit(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    weight_data=W_residual,  # Passed directly, quantized in __init__
                    bias_data=bias_data,      # Passed directly, stored as fp16
                    compute_dtype=torch.float16,
                    device=original_device,
                )
                
                # Replace base layer with quantized version
                self.base_layer = quantized_layer
            else:
                # Standard fp16/fp32 approach - use to_free to convert and free W_residual
                self.base_layer.weight.data = to_free(W_residual, dtype=original_dtype, device=original_device)
                self.base_layer.weight.requires_grad = False
                # Note: W_residual already freed by to_free() above, no need for explicit del
            
            # Note: In quantized path, W_residual is passed to ResidualLinear4bit which frees it internally
            
            # ========== Store principal components U, V, S ==========
            # These are kept in fp16 by default (like LoRA A/B in QLoRA)
            # Optional: quantize U, V if requested (not recommended, but supported)
            
            if self.pissa_config.quantize_base_components and HAS_BITSANDBYTES:
                print(f"      Quantizing U and V to NF4 (not recommended, may impact accuracy)")
                from bitsandbytes.nn import Params4bit
                
                try:
                    # Convert to fp16 and free original fp32 tensors
                    U_fp16 = to_free(U_principal, dtype=torch.float16, device=original_device)
                    V_fp16 = to_free(V_principal, dtype=torch.float16, device=original_device)
                    
                    # Quantize U and V (Params4bit takes ownership of the tensor)
                    U_quantized = Params4bit(
                        U_fp16,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    V_quantized = Params4bit(
                        V_fp16,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    
                    # Free the fp16 intermediates (Params4bit made a copy)
                    del U_fp16, V_fp16
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Register as buffers (non-trainable)
                    self.register_buffer("U", U_quantized)
                    self.register_buffer("V", V_quantized)
                    
                    print(f"      ✓ U, V quantized to NF4")
                    
                except Exception as e:
                    print(f"      Warning: U/V quantization failed, using fp16: {e}")
                    # Fallback to fp16 - use to_free to convert and free originals
                    self.register_buffer("U", to_free(U_principal, dtype=torch.float16, device=original_device))
                    self.register_buffer("V", to_free(V_principal, dtype=torch.float16, device=original_device))
            else:
                # Standard approach: keep U, V in same dtype as original model
                # Use to_free to convert dtype and free the original fp32 tensors
                self.register_buffer("U", to_free(U_principal, dtype=original_dtype, device=original_device))
                self.register_buffer("V", to_free(V_principal, dtype=original_dtype, device=original_device))
            
            # Note: U_principal and V_principal are already freed by to_free() above
            # No need for explicit del + gc.collect + empty_cache here
            
            # S: Use FP32 if s_dtype_fp32 is enabled (fixes bfloat16 precision issues)
            # With bfloat16, small optimizer updates (lr*grad ~ 2e-6) round to 0 for values ~6.0
            s_dtype = torch.float32 if self.pissa_config.s_dtype_fp32 else original_dtype
            S_converted = to_free(S_principal, dtype=s_dtype, device=original_device)
            if self.pissa_config.freeze_singular_values:
                self.register_buffer("S", S_converted)
            else:
                self.S = nn.Parameter(S_converted)
            
            # Note: S_principal already freed by to_free() above
            
            # print(f"Layer {self.adapter_name}: S stats - Min: {self.S.min().item():.4f}, Max: {self.S.max().item():.4f}, Mean: {self.S.mean().item():.4f}, dtype: {self.S.dtype}")
    
    def _initialize_rotation_matrices(self):
        """Initialize rotation matrices R_U and R_V based on the selected method."""
        if self.pissa_config.method == "way0":
            self._init_way0()
        elif self.pissa_config.method == "way1":
            self._init_way1()
        elif self.pissa_config.method == "way2":
            self._init_way2()
        elif self.pissa_config.method == "way3":
            self._init_way3()
        else:
            raise ValueError(f"Unknown rotation method: {self.pissa_config.method}")
    
    def _init_way0(self):
        """Way 0: Direct parameterization with orthogonality regularization."""
        # Get device and dtype from U buffer (which is already on correct device)
        device = self.U.device
        dtype = self.U.dtype
        
        if self.pissa_config.init_identity:
            self.R_U = nn.Parameter(torch.eye(self.r, dtype=dtype, device=device))
            self.R_V = nn.Parameter(torch.eye(self.r, dtype=dtype, device=device))
        else:
            # Initialize as random orthogonal matrices
            R_U_init = torch.randn(self.r, self.r, dtype=dtype, device=device)
            R_V_init = torch.randn(self.r, self.r, dtype=dtype, device=device)
            
            # QR decomposition to get orthogonal matrices
            Q_U, _ = torch.linalg.qr(R_U_init)
            Q_V, _ = torch.linalg.qr(R_V_init)
            
            self.R_U = nn.Parameter(Q_U)
            self.R_V = nn.Parameter(Q_V)
    
    def _init_way1(self):
        """Way 1: Greedy sequential Givens rotations with dynamic layer instantiation.
        
        To save VRAM, we only instantiate the current active Givens layer.
        After training each layer, we:
        1. Merge its rotation into U and V (in-place)
        2. Delete the layer to free VRAM
        3. Instantiate the next layer
        """
        n_layers = self.pissa_config.n_givens_layers or (self.r - 1)
        
        # Generate pairings for all phases (we'll instantiate layers on-demand)
        self.givens_pairings = generate_givens_pairings(self.r, n_layers)
        
        # Instead of creating all layers, we'll create them dynamically
        # Store only the current active layer - must use add_module for proper registration
        current_u = GivensRotationLayer(self.r, self.givens_pairings[0])
        current_v = GivensRotationLayer(self.r, self.givens_pairings[0])
        
        # Register as submodules so PyTorch tracks them properly
        self.add_module('current_givens_u', current_u)
        self.add_module('current_givens_v', current_v)
        
        # Note: U and V buffers will be updated in-place as we merge rotations
        # No need for separate U_base/V_base since we modify U/V directly
    
    def _init_way2(self):
        """Way 2: Low-rank skew-symmetric perturbation I + BC^T - CB^T."""
        lr_r = self.pissa_config.low_rank_r
        
        # Get device and dtype from U buffer
        device = self.U.device
        dtype = self.U.dtype
        
        # Low-rank matrices B, C for U and V rotations
        self.B_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        self.B_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        
        if self.pissa_config.init_identity:
            # Initialize C to zero so that BC^T - CB^T = 0, resulting in Identity rotation
            self.C_U = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
            self.C_V = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
        else:
            self.C_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
            self.C_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
    
    def _init_way3(self):
        """Way 3: Exponential map of skew-symmetric matrix."""
        lr_r = self.pissa_config.low_rank_r
        
        # Get device and dtype from U buffer
        device = self.U.device
        dtype = self.U.dtype
        
        # Low-rank matrices B, C for generating skew-symmetric matrices
        self.B_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        self.B_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
        
        if self.pissa_config.init_identity:
            # Initialize C to zero so that BC^T - CB^T = 0, resulting in Identity rotation (exp(0)=I)
            self.C_U = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
            self.C_V = nn.Parameter(torch.zeros(self.r, lr_r, dtype=dtype, device=device))
        else:
            self.C_U = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
            self.C_V = nn.Parameter(torch.randn(self.r, lr_r, dtype=dtype, device=device) * 0.1)
    
    def get_rotation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the current rotation matrices R_U and R_V."""
        if self.pissa_config.method == "way0":
            return self.R_U, self.R_V
        
        elif self.pissa_config.method == "way1":
            # For Way 1, return the current active layer's rotation
            return self.current_givens_u(), self.current_givens_v()
        
        elif self.pissa_config.method == "way2":
            # R = I + BC^T - CB^T (skew-symmetric perturbation)
            R_U = (torch.eye(self.r, device=self.B_U.device, dtype=self.B_U.dtype) + 
                   self.B_U @ self.C_U.T - self.C_U @ self.B_U.T)
            R_V = (torch.eye(self.r, device=self.B_V.device, dtype=self.B_V.dtype) + 
                   self.B_V @ self.C_V.T - self.C_V @ self.B_V.T)
            return R_U, R_V
        
        elif self.pissa_config.method == "way3":
            # R = exp(BC^T - CB^T)
            skew_U = self.B_U @ self.C_U.T - self.C_U @ self.B_U.T
            skew_V = self.B_V @ self.C_V.T - self.C_V @ self.B_V.T
            
            # Cast to float32 for matrix_exp stability/compatibility
            orig_dtype = skew_U.dtype
            R_U = torch.matrix_exp(skew_U.float()).to(dtype=orig_dtype)
            R_V = torch.matrix_exp(skew_V.float()).to(dtype=orig_dtype)
            return R_U, R_V
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the rotational adapter.
        
        Following PiSSA formulation:
        Y = X * W = X * (W_residual + W_principal)
        where:
        - W_residual is stored in base_layer (NF4 quantized if quantize_residual=True)
        - W_principal = U @ R_U @ S @ R_V^T @ V^T (trainable via rotations)
        
        Handles quantized U/V by dequantizing before matmul operations.
        """
        # Base layer forward pass (contains W_residual - optionally quantized)
        result = self.base_layer(x)
        
        # Get current rotation matrices
        R_U, R_V = self.get_rotation_matrices()

        # Get U, V (dequantize if needed)
        # If U/V are Params4bit, dequantize them for use in matmul
        U_current = dequantize_params4bit(self.U) if hasattr(self.U, 'quant_state') else self.U
        V_current = dequantize_params4bit(self.V) if hasattr(self.V, 'quant_state') else self.V
        
        # Ensure all components are in same dtype as input for matmul compatibility
        # Only convert if dtype differs to avoid unnecessary overhead
        target_dtype = x.dtype
        # print("target_dtype:", target_dtype)
        # print("x.dtype:", x.dtype)
        
        if U_current.dtype != target_dtype:
            U_current = U_current.to(target_dtype)
        if V_current.dtype != target_dtype:
            V_current = V_current.to(target_dtype)
        if R_U.dtype != target_dtype:
            R_U = R_U.to(target_dtype)
        if R_V.dtype != target_dtype:
            R_V = R_V.to(target_dtype)
        # S is stored in FP32 for optimizer precision, convert to compute dtype
        # Note: .to() is differentiable, gradients flow back to self.S
        if self.S.dtype != target_dtype:
            S_current = self.S.to(target_dtype)
        else:
            S_current = self.S
    
        # Forward through the adaptation path: x @ V^T @ R_V @ S @ R_U^T @ U^T
        x_adapted = self.dropout(x)
        
        # Apply the rotational transformation chain
        # Note: V is [r, in_features], so V^T is [in_features, r]
        x_adapted = x_adapted @ V_current.T  # [batch, in_features] @ [in_features, r] -> [batch, r]
        x_adapted = x_adapted @ R_V.T        # [batch, r] @ [r, r] -> [batch, r]
        x_adapted = x_adapted * S_current    # [batch, r] * [r] -> [batch, r] (broadcasting)
        x_adapted = x_adapted @ R_U.T        # [batch, r] @ [r, r] -> [batch, r]
        x_adapted = x_adapted @ U_current.T  # [batch, r] @ [r, out_features] -> [batch, out_features]
        
        # Scale and add W_principal to W_residual result
        result = result + x_adapted * self.scaling
        
        return result

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for Way 0.
        
        Supports three regularization types:
        1. frobenius: ||R^T @ R - I||_F^2 (encourages orthogonality)
        2. determinant: (det(R) - 1)^2 (encourages det = +1)
        3. log_determinant: (log(det(R)) - 0)^2 (numerically stable det constraint)
        """
        # Early return if regularization is disabled to save compute
        if self.pissa_config.orthogonality_reg_weight == 0:
            return torch.tensor(0.0, device=self.R_U.device if hasattr(self, 'R_U') else next(self.parameters()).device)
        
        if self.pissa_config.method != "way0":
            # For ways 1, 2, 3: orthogonality is guaranteed by construction
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        
        reg_type = self.pissa_config.regularization_type
        
        if reg_type == "frobenius":
            # ||R_U^T @ R_U - I||_F^2 + ||R_V^T @ R_V - I||_F^2
            I = torch.eye(self.r, device=self.R_U.device, dtype=self.R_U.dtype)
            loss_u = torch.norm(self.R_U.T @ self.R_U - I, p='fro') ** 2
            loss_v = torch.norm(self.R_V.T @ self.R_V - I, p='fro') ** 2
            
        elif reg_type == "determinant":
            # (det(R_U) - 1)^2 + (det(R_V) - 1)^2
            # det doesn't support BF16, temporarily upcast to FP32
            det_u = torch.det(self.R_U.float())
            det_v = torch.det(self.R_V.float())
            loss_u = (det_u - 1.0) ** 2
            loss_v = (det_v - 1.0) ** 2
            
        elif reg_type == "log_determinant":
            # (log(det(R_U)) - 0)^2 + (log(det(R_V)) - 0)^2
            # More numerically stable than direct determinant
            # logdet doesn't support BF16, temporarily upcast to FP32
            log_det_u = torch.logdet(self.R_U.float())
            log_det_v = torch.logdet(self.R_V.float())
            loss_u = log_det_u ** 2
            loss_v = log_det_v ** 2
        
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")
        
        return (loss_u + loss_v) * self.pissa_config.orthogonality_reg_weight
    
    def step_phase(self):
        """Advance to next phase for Way 1 (sequential Givens training).
        
        This method:
        1. Merges current Givens rotation into U and V (in-place)
        2. Deletes current Givens layers to free VRAM
        3. ALWAYS instantiates next Givens layer (even after all cycles complete)
           - This ensures forward pass never breaks
           - Layers after training completes are just frozen (won't be trained)
           
        Returns:
            tuple: (params_before, params_after) for tracking parameter count changes
        """
        if self.pissa_config.method != "way1":
            return (0, 0)

        # Count trainable parameters before merging
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            # Get current rotation matrices before deletion
            R_U = self.current_givens_u()
            R_V = self.current_givens_v()
            
            # Merge rotations into U and V IN-PLACE
            # U_new = U_old @ R_U
            # V_new = R_V @ V_old
            self.U.copy_(self.U @ R_U)
            self.V.copy_(R_V @ self.V)
            
            # Delete current Givens layers to free VRAM
            delattr(self, 'current_givens_u')
            delattr(self, 'current_givens_v')
            
            # Advance to next layer
            self.current_layer_index = (self.current_layer_index + 1) % len(self.givens_pairings)
            
            # Check if we completed a full cycle
            if self.current_layer_index == 0:
                self.current_cycle += 1
                print(f"  ✓ Completed cycle {self.current_cycle}/{self.pissa_config.total_cycles}")
            
            # ALWAYS instantiate next Givens layer (prevents AttributeError in forward/validation)
            # Even after all cycles complete, we need layers for forward pass
            next_pairs = self.givens_pairings[self.current_layer_index]
            next_u = GivensRotationLayer(self.r, next_pairs)
            next_v = GivensRotationLayer(self.r, next_pairs)
            
            # Move to same device as U
            device = self.U.device
            next_u = next_u.to(device)
            next_v = next_v.to(device)
            
            # If we've completed all cycles, freeze these layers (they won't be trained further)
            if self.current_cycle >= self.pissa_config.total_cycles:
                next_u.thetas.requires_grad = False
                next_v.thetas.requires_grad = False
                print(f"  ⚠️ All {self.pissa_config.total_cycles} cycles complete - subsequent layers frozen")
            
            # Register as submodules
            self.add_module('current_givens_u', next_u)
            self.add_module('current_givens_v', next_v)
            
        # Count trainable parameters after instantiating next layer
        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (params_before, params_after)
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters in this adapter.
        
        Returns:
            Dictionary of parameter names to tensors
            
        Trainable components:
            - R_U, R_V: Rotation matrices (all 4 ways)
            - S: Singular values (if freeze_singular_values=False)
            - Givens angles: For Way 1
            - B_U, C_U, B_V, C_V: For Ways 2 & 3
            
        Frozen components:
            - base_layer.weight: W_residual (NF4 quantized)
            - U, V: Principal singular vectors (optionally quantized)
        """
        trainable = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        
        return trainable
    
    def get_frozen_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all frozen parameters and buffers.
        
        Returns:
            Dictionary of frozen parameter/buffer names to tensors
        """
        frozen = {}
        
        # Add frozen parameters
        for name, param in self.named_parameters():
            if not param.requires_grad:
                frozen[name] = param
        
        # Add buffers (always frozen)
        for name, buffer in self.named_buffers():
            frozen[name] = buffer
            
        return frozen
    
    def freeze_base_and_unfreeze_adapters(self):
        """
        Freeze base layer (W_residual, bias) and buffers (U, V), 
        unfreeze rotation parameters and optionally S.
        
        This method ensures correct freezing regardless of the rotation method:
        - Way0: R_U, R_V trainable
        - Way1: givens_u_layers.*.thetas, givens_v_layers.*.thetas trainable
        - Way2/3: B_U, C_U, B_V, C_V trainable
        - S: trainable if freeze_singular_values=False
        
        Base layer weight/bias and U/V buffers are always frozen.
        """
        for name, param in self.named_parameters():
            # Freeze base_layer weight and bias (W_residual)
            if name.startswith('base_layer.'):
                param.requires_grad = False
            else:
                # Everything else (rotation params, S if not frozen) should be trainable
                param.requires_grad = True
        
        # Note: U and V are buffers (not parameters), so they're automatically frozen
        # S is either a buffer (frozen) or parameter (trainable) based on config
    
    def print_parameter_summary(self):
        """Print a summary of trainable vs frozen parameters."""
        trainable = self.get_trainable_parameters()
        frozen = self.get_frozen_parameters()
        
        print(f"\n📊 Parameter Summary for {self.__class__.__name__}")
        print("=" * 60)
        
        print(f"🔥 Trainable Parameters ({len(trainable)} items):")
        total_trainable = 0
        for name, param in trainable.items():
            size = param.numel()
            total_trainable += size
            print(f"  • {name}: {tuple(param.shape)} ({size:,} params)")
        
        print(f"\n🧊 Frozen Parameters/Buffers ({len(frozen)} items):")
        total_frozen = 0
        for name, param in frozen.items():
            size = param.numel() if hasattr(param, 'numel') else len(param)
            total_frozen += size
            dtype = getattr(param, 'dtype', 'unknown')
            print(f"  • {name}: {tuple(param.shape)} ({size:,} params, {dtype})")
        
        print(f"\n📈 Summary:")
        print(f"  • Total trainable: {total_trainable:,}")
        print(f"  • Total frozen: {total_frozen:,}")
        print(f"  • Efficiency ratio: {total_trainable/(total_trainable+total_frozen):.4f}")
        print(f"  • Quantization: W_residual={'✅' if self.pissa_config.quantize_residual else '❌'}, U/V={'✅' if self.pissa_config.quantize_base_components else '❌'}")

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class RotationalPiSSATrainer:
    """Helper class for training rotational PiSSA models with Way 1 (sequential Givens)."""
    
    def __init__(self, model: nn.Module, config: RotationalPiSSAConfig):
        self.model = model
        self.config = config
        
        # Find all rotational adapters
        self.adapters = []
        for module in model.modules():
            if isinstance(module, RotationalLinearLayer):
                self.adapters.append(module)
    
    def should_step_phase(self, global_step: int) -> bool:
        """Check if we should advance to the next phase."""
        if self.config.method != "way1":
            return False
        
        return global_step % self.config.steps_per_phase == 0 and global_step > 0
    
    def step_phase(self):
        """Advance all adapters to next phase.
        
        Returns:
            tuple: (total_params_before, total_params_after) aggregated across all adapters
        """
        total_before = 0
        total_after = 0
        
        for adapter in self.adapters:
            before, after = adapter.step_phase()
            total_before += before
            total_after += after
        
        return (total_before, total_after)
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """Get total orthogonality regularization loss."""
        total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for adapter in self.adapters:
            total_loss = total_loss + adapter.get_orthogonality_loss()
        
        return total_loss
    
    # def is_training_complete(self) -> bool:
    #     """Check if training is complete for Way 1."""
    #     if self.config.method != "way1" or not self.adapters:
    #         return False
        
    #     return self.adapters[0].current_cycle >= self.config.total_cycles

# ============================================================================
# BATCHED SVD HELPER FOR FAST INITIALIZATION
# ============================================================================

def _create_adapter_from_svd(
    base_layer: nn.Linear,
    pissa_config: 'RotationalPiSSAConfig',
    adapter_name: str,
    U: torch.Tensor,  # Full U from SVD [out_features, k]
    S: torch.Tensor,  # Full S from SVD [k]
    V: torch.Tensor,  # Full V from SVD [k, in_features]
    original_dtype: torch.dtype,
    original_device: torch.device
) -> 'RotationalLinearLayer':
    """
    Create a RotationalLinearLayer from pre-computed SVD results.
    
    This is used by batched SVD processing to avoid redundant per-layer SVD computation.
    The SVD is computed once in a batch for all layers with the same shape, then this
    function creates adapters from those pre-computed results.
    """
    r = pissa_config.r
    
    # Create adapter instance WITHOUT calling _initialize_svd_components
    adapter = object.__new__(RotationalLinearLayer)
    nn.Module.__init__(adapter)
    
    adapter.base_layer = base_layer
    adapter.pissa_config = pissa_config
    adapter.adapter_name = adapter_name
    adapter.r = r
    adapter.in_features = base_layer.in_features
    adapter.out_features = base_layer.out_features
    
    # Split into principal and residual components
    U_principal = U[:, :r].contiguous()
    U_residual = U[:, r:].contiguous()
    S_principal = S[:r].contiguous()
    S_residual = S[r:].contiguous()
    V_principal = V[:r, :].contiguous()
    V_residual = V[r:, :].contiguous()
    
    # Compute W_residual = U_residual @ diag(S_residual) @ V_residual
    if U_residual.shape[1] > 0:
        W_residual = U_residual @ torch.diag(S_residual) @ V_residual
    else:
        W_residual = torch.zeros(adapter.out_features, adapter.in_features, 
                                  dtype=torch.float32, device=original_device)
    
    # Set base layer weight to W_residual (frozen)
    with torch.no_grad():
        adapter.base_layer.weight.data = W_residual.to(dtype=original_dtype, device=original_device)
        adapter.base_layer.weight.requires_grad = False
    
    # Store principal components as buffers
    adapter.register_buffer("U", U_principal.to(dtype=original_dtype, device=original_device))
    adapter.register_buffer("V", V_principal.to(dtype=original_dtype, device=original_device))
    
    # S: stored as FP32 if s_dtype_fp32 is enabled
    s_dtype = torch.float32 if pissa_config.s_dtype_fp32 else original_dtype
    S_converted = S_principal.to(dtype=s_dtype, device=original_device)
    if pissa_config.freeze_singular_values:
        adapter.register_buffer("S", S_converted)
    else:
        adapter.S = nn.Parameter(S_converted)
    
    # Initialize rotation matrices
    adapter._initialize_rotation_matrices()
    
    # Dropout
    if pissa_config.lora_dropout > 0:
        adapter.dropout = nn.Dropout(pissa_config.lora_dropout)
    else:
        adapter.dropout = nn.Identity()
    
    adapter.scaling = 1
    
    # Training state for Way 1
    if pissa_config.method == "way1":
        adapter.current_layer_index = 0
        adapter.current_cycle = 0
        adapter.step_count = 0
    
    return adapter

# ============================================================================
# MODEL REPLACEMENT UTILITIES
# ============================================================================

def replace_linear_with_rotational_pissa(
    model: nn.Module,
    pissa_config: RotationalPiSSAConfig,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    adapter_name: str = "default",
    freeze_base_model: bool = True
) -> Dict[str, RotationalLinearLayer]:
    """
    Replace Linear layers in a model with RotationalLinearLayer.
    
    This function implements the PiSSA methodology:
    1. For each target linear layer with weight W
    2. Computes SVD: W = U @ S @ V^T  
    3. Splits into principal (top-r) and residual components
    4. Updates original layer weight to W_res = U[:,r:] @ S[r:,r:] @ V[:,r:]^T
    5. Creates adapter with W_pri = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T
    6. Final output: Y = X * (W_res + W_pri)
    
    Args:
        model: The model to modify
        pissa_config: Configuration for the rotational adapters
        target_modules: List of module names to target (e.g., ["q_proj", "v_proj"])
        exclude_modules: List of module names to exclude (e.g., ["pooler", "classifier"])
        adapter_name: Name of the adapter
        freeze_base_model: If True, freeze all model params except adapter trainable params
    
    Returns:
        Dictionary mapping module names to created adapters
    """
    import gc
    
    adapters = {}
    layers_to_replace = []
    
    # First pass: identify all layers to replace (avoid modifying dict during iteration)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be excluded
            if exclude_modules is not None:
                should_exclude = any(excl in name for excl in exclude_modules)
                if should_exclude:
                    continue  # Skip this layer
            
            # Check if this module should be replaced
            if target_modules is None:
                should_replace = True
            else:
                should_replace = any(target in name for target in target_modules)
            
            if should_replace:
                layers_to_replace.append((name, module))
    
    print(f"  Found {len(layers_to_replace)} layers to replace")
    
    # ========== OPTIMIZED: Group layers by shape for parallel processing ==========
    # Group layers by (out_features, in_features) shape for efficient batching
    shape_groups = {}
    for name, module in layers_to_replace:
        shape = (module.out_features, module.in_features)
        shape_groups.setdefault(shape, []).append((name, module))
    
    print(f"  Grouped into {len(shape_groups)} shape groups: {[f'{s}: {len(g)} layers' for s, g in shape_groups.items()]}")
    
    # Create CUDA streams for parallel processing
    # NOTE: Disable streams on multi-GPU setups to avoid NCCL conflicts with DataParallel
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_streams = torch.cuda.is_available() and num_gpus == 1  # Only use streams on single GPU
    
    if use_streams:
        num_streams = 4
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        print(f"  Using {num_streams} CUDA streams for parallel SVD (single GPU mode)")
    else:
        streams = None
        if num_gpus > 1:
            print(f"  Multi-GPU detected ({num_gpus} GPUs) - using sequential processing to avoid NCCL conflicts")
    
    total_processed = 0
    
    # Process each shape group - use BATCHED SVD for maximum GPU utilization
    for shape, group_layers in shape_groups.items():
        batch_size = len(group_layers)
        print(f"  Processing shape {shape}: {batch_size} layers (batched SVD)...")
        
        if use_streams and batch_size > 1:
            # ========== ADAPTIVE BATCHED SVD with memory management ==========
            import time
            
            # Get dtype/device from first layer
            first_module = group_layers[0][1]
            original_dtype = first_module.weight.dtype
            original_device = first_module.weight.device
            
            # DIAGNOSTIC: Verify GPU is available and being used
            print(f"    Device: {original_device}, dtype: {original_dtype}")
            if not original_device.type == 'cuda':
                print(f"    WARNING: Weights are on {original_device}, not CUDA! Moving to GPU...")
                # Use current CUDA device (respects CUDA_VISIBLE_DEVICES)
                original_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            
            # Estimate memory required for this batch
            out_feat, in_feat = shape
            # Memory for stacked weights in FP32: batch_size * out_feat * in_feat * 4 bytes
            # SVD output (U, S, V) roughly doubles this
            bytes_per_matrix = out_feat * in_feat * 4  # FP32
            estimated_gb = (batch_size * bytes_per_matrix * 2) / (1024**3)  # Input + Output
            
            # Target 4GB per batch to be safe (leaves room for other ops)
            MAX_BATCH_GB = 4.0
            
            if estimated_gb > MAX_BATCH_GB:
                # Split into smaller sub-batches
                matrices_per_batch = max(1, int(batch_size * (MAX_BATCH_GB / estimated_gb)))
                num_sub_batches = (batch_size + matrices_per_batch - 1) // matrices_per_batch
                print(f"    Memory estimate: {estimated_gb:.2f} GB > {MAX_BATCH_GB} GB limit")
                print(f"    Splitting into {num_sub_batches} sub-batches of ~{matrices_per_batch} layers each")
            else:
                matrices_per_batch = batch_size
                num_sub_batches = 1
                print(f"    Memory estimate: {estimated_gb:.2f} GB (fits in single batch)")
            
            # Process in sub-batches
            all_U, all_S, all_V = [], [], []
            
            for sub_batch_idx in range(num_sub_batches):
                start_idx = sub_batch_idx * matrices_per_batch
                end_idx = min(start_idx + matrices_per_batch, batch_size)
                sub_batch = group_layers[start_idx:end_idx]
                sub_batch_size = len(sub_batch)
                
                if num_sub_batches > 1:
                    print(f"    Sub-batch {sub_batch_idx+1}/{num_sub_batches}: processing {sub_batch_size} layers...")
                
                # Stack weights for this sub-batch
                start_time = time.time()
                weight_batch = torch.stack([m.weight.data.to(original_device) for _, m in sub_batch])
                stack_time = time.time() - start_time
                
                if sub_batch_idx == 0:  # Only print details for first sub-batch
                    print(f"    Stacked {sub_batch_size} weights: shape={weight_batch.shape}, device={weight_batch.device}, dtype={weight_batch.dtype}")
                
                # SVD requires FP32 - convert but STAY ON GPU
                # VRAM optimization: explicitly free old tensor during conversion
                if weight_batch.dtype in (torch.bfloat16, torch.float16):
                    old_dtype = weight_batch.dtype
                    weight_batch_fp32 = weight_batch.float()  # Create FP32 copy
                    del weight_batch  # Free original BF16/FP16 tensor (saves ~50% memory)
                    weight_batch = weight_batch_fp32
                    if sub_batch_idx == 0:
                        print(f"    Converted {old_dtype} -> FP32: device={weight_batch.device}")
                
                # Ensure we're on CUDA before SVD
                assert weight_batch.device.type == 'cuda', f"weight_batch moved to {weight_batch.device}!"
                
                # Batched SVD for this sub-batch
                if sub_batch_idx == 0:
                    print(f"    Running batched SVD on {weight_batch.device}...")
                
                # NOTE: torch.cuda.synchronize() is for accurate timing only
                # PyTorch GPU ops are async - without sync, time.time() only measures CPU overhead
                # Set ENABLE_TIMING=False to skip sync and run ~5% faster (but no timing info)
                ENABLE_TIMING = False
                
                if ENABLE_TIMING:
                    torch.cuda.synchronize()  # Wait for previous ops before starting timer
                svd_start = time.time()
                
                U_batch, S_batch, V_batch = torch.linalg.svd(weight_batch, full_matrices=False)
                
                if ENABLE_TIMING:
                    torch.cuda.synchronize()  # Wait for SVD to complete before stopping timer
                svd_time = time.time() - svd_start
                
                if sub_batch_idx == 0 or num_sub_batches > 1:
                    print(f"    ✓ Sub-batch SVD complete in {svd_time:.2f}s (stack: {stack_time:.2f}s)")
                
                # Store results
                all_U.append(U_batch)
                all_S.append(S_batch)
                all_V.append(V_batch)
                
                del weight_batch
                torch.cuda.empty_cache()  # Free memory between sub-batches
            
            # Concatenate all sub-batch results
            U_batch = torch.cat(all_U, dim=0)
            S_batch = torch.cat(all_S, dim=0)
            V_batch = torch.cat(all_V, dim=0)
            
            del all_U, all_S, all_V
            
            if num_sub_batches == 1:
                print(f"    Result devices: U={U_batch.device}, S={S_batch.device}, V={V_batch.device}")
            
            # Now create adapters using pre-computed SVD results
            r = pissa_config.r
            
            for i, (name, module) in enumerate(group_layers):
                # Extract this layer's SVD components
                U_i = U_batch[i]  # [out_features, k]
                S_i = S_batch[i]  # [k]
                V_i = V_batch[i]  # [k, in_features]
                
                # Create adapter with pre-computed SVD (bypass _initialize_svd_components)
                adapted_layer = _create_adapter_from_svd(
                    module, pissa_config, adapter_name,
                    U_i, S_i, V_i, original_dtype, original_device
                )
                
                # Replace the module in model
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, adapted_layer)
                else:
                    setattr(model, child_name, adapted_layer)
                
                adapters[name] = adapted_layer
                del module
                
                total_processed += 1
                if total_processed % 20 == 0:
                    print(f"    [{total_processed}/{len(layers_to_replace)}] layers processed...")
            
            # Clean up batch tensors
            del U_batch, S_batch, V_batch
            
        else:
            # Fallback: sequential processing (single layer or CPU)
            for name, module in group_layers:
                adapted_layer = RotationalLinearLayer(module, pissa_config, adapter_name)
                
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, adapted_layer)
                else:
                    setattr(model, child_name, adapted_layer)
                
                adapters[name] = adapted_layer
                del module
                total_processed += 1
        
        # Cleanup once per shape group (not per layer!) - much faster
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"  ✓ All {total_processed} layers processed")
    
    # Final cleanup
    del layers_to_replace, shape_groups
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply freezing strategy if requested
    if freeze_base_model:
        # 1) Freeze ALL parameters in the model first
        for param in model.parameters():
            param.requires_grad = False
        
        # 2) Unfreeze adapter trainable parameters (method-agnostic)
        for adapter in adapters.values():
            adapter.freeze_base_and_unfreeze_adapters()
    
    return adapters

