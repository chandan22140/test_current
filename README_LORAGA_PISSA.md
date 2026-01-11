# LoRA-GA with Rotational PiSSA Implementation

This folder contains a modified version of the LoRA-GA code that uses **Rotational PiSSA** instead of PEFT's LoRA-GA implementation.

## Files Overview

### Core Training Script
- **`float_llama2-7b_metamath.py`**: Main training script adapted for Rotational PiSSA

### Supporting Modules
- **`data.py`**: Dataset loading utilities (**NO CHANGES** - identical to lora_ga/data.py)
- **`utils.py`**: Model initialization, training, and helper functions
- **`logTrainer.py`**: Custom trainer with logging and orthogonality regularization
- **`merge.py`**: Model checkpoint merging utilities

## Key Differences from lora_ga

### 1. **Import Changes**
```python
# ORIGINAL (lora_ga):
from peft import PeftModel, LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext

# MODIFIED (test_current):
from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    replace_linear_with_rotational_pissa,
)
```

### 2. **Configuration**
```python
# ORIGINAL:
peft_config = LoraGAConfig(
    target_modules=find_all_linear_modules(model=model),
    lora_alpha=config["a"],
    r=config["r"],
    iters=config["s"] // 2,
)

# MODIFIED:
pissa_config = RotationalPiSSAConfig(
    r=config["r"],
    lora_alpha=config["a"],
    method="way0",  # Direct optimization with regularization
    orthogonality_reg_weight=1e-3,
    init_identity=True,
    freeze_singular_values=False,
    quantize_residual=False,
    quantize_base_components=False,
)
```

### 3. **Model Adaptation**
```python
# ORIGINAL:
with LoraGAContext(model=model, named_grad=named_grad):
    model = get_peft_model(model=model, peft_config=peft_config)

# MODIFIED:
adapters = replace_linear_with_rotational_pissa(
    model=model,
    pissa_config=pissa_config,
    target_modules=find_all_linear_modules(model=model),
    adapter_name="default",
    freeze_base_model=True,
)
```

**Note on Gradient-Guided Initialization:**
- LoRA-GA uses `LoraGAContext` to initialize adapters based on estimated gradients
- Rotational PiSSA uses SVD decomposition which inherently captures weight structure
- The `estimate_gradient` function is kept for compatibility but is not strictly needed for PiSSA

### 4. **Checkpoint Saving**
```python
# ORIGINAL:
save_loraga_model_init(model=model, save_dir=save_dir)
save_loraga_model_final(model=model, save_dir=save_dir)

# MODIFIED:
torch.save({
    'model_state_dict': model.state_dict(),
    'pissa_config': pissa_config,
    'adapters': list(adapters.keys()),
}, os.path.join(save_dir, "init_checkpoint.pt"))
```

### 5. **Checkpoint Loading**
```python
# ORIGINAL:
model = PeftModel.from_pretrained(model, save_dir)

# MODIFIED:
checkpoint = torch.load(os.path.join(save_dir, "final_checkpoint.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
```

### 6. **Trainer Modifications**
```python
# Added in LogTrainer.__init__:
pissa_config = None,  # For orthogonality regularization

# Added method:
def _training_step_with_pissa_reg(self, model, inputs):
    # Adds orthogonality loss for way0 method
    # Logs ortho_loss, task_loss, total_loss
```

### 7. **Parameter Detection**
```python
# ORIGINAL (lora_ga):
if "lora_A" in name:
    self.orig_A[...] = param.detach().clone()
elif "lora_B" in name:
    self.orig_B[...] = param.detach().clone()

# MODIFIED (test_current):
if "R_U" in name or "B_U" in name:
    self.orig_A[...] = param.detach().clone()
elif "R_V" in name or "B_V" in name:
    self.orig_B[...] = param.detach().clone()
```

## Rotational PiSSA Parameters

The modified code uses **RotationalPiSSAConfig** with these key parameters:

- **`r`**: Rank (same as LoRA rank)
- **`lora_alpha`**: Scaling factor (same as LoRA alpha)
- **`method`**: Rotation parameterization ("way0", "way1", "way2", "way3")
  - **way0**: Direct optimization with orthogonality regularization
  - **way1**: Greedy sequential Givens rotations
  - **way2**: Low-rank skew-symmetric perturbation
  - **way3**: Exponential map of skew-symmetric matrix
- **`orthogonality_reg_weight`**: Weight for orthogonality regularization (way0 only)
- **`init_identity`**: Initialize rotation matrices as identity
- **`freeze_singular_values`**: Whether to freeze S matrix
- **`quantize_residual`**: NF4 quantize W_residual (like QLoRA)
- **`quantize_base_components`**: NF4 quantize U and V (not recommended)

## Mathematical Differences

### LoRA-GA Approach:
1. Estimate gradients on calibration data
2. Use gradients to initialize LoRA adapters (A, B matrices)
3. Train: `ΔW = B @ A` with gradient-guided initialization

### Rotational PiSSA Approach:
1. SVD decomposition: `W = U @ S @ V^T`
2. Split into principal and residual components
3. Train: `W_principal = U @ R_U @ S @ R_V^T @ V^T` via rotation matrices
4. Frozen: `W_residual` (optionally NF4 quantized)

**Key Insight**: PiSSA's SVD decomposition captures similar information to gradient-based initialization, but through eigenvalue decomposition instead of gradient estimation.

## Usage

```bash
# Run training with Rotational PiSSA
cd /home/chandan/DL_Quantization/backup_131/DL_Quantization/PiSSA/test_current
python float_llama2-7b_metamath.py --lora_alpha=8 --lora_rank=32 --sample_size=128 --seed=31
```

## Main Advantages of This Port

1. ✅ **No PEFT dependency**: Uses custom rotational PiSSA implementation
2. ✅ **Highlighted changes**: All modifications marked with `# CHANGED:` comments
3. ✅ **Compatible interface**: Maintains similar API to LoRA-GA
4. ✅ **Quantization support**: Optional NF4 quantization for residuals
5. ✅ **Orthogonality regularization**: Built-in support for way0 method

## Files Changed

| File | Changes | Complexity |
|------|---------|------------|
| `float_llama2-7b_metamath.py` | PEFT → Rotational PiSSA imports, config, adaptation | ⭐⭐⭐⭐⭐⭐⭐ |
| `data.py` | **None** (identical copy) | ⭐ |
| `utils.py` | Added `estimate_gradient`, pass `pissa_config` to trainer | ⭐⭐⭐⭐⭐⭐ |
| `logTrainer.py` | Detect rotational layers, add orthogonality loss | ⭐⭐⭐⭐⭐⭐⭐ |
| `merge.py` | PyTorch state dict loading instead of PEFT | ⭐⭐⭐⭐⭐⭐ |

## Notes

1. **Gradient Estimation**: The`estimate_gradient` function is kept for workflow compatibility but PiSSA doesn't strictly need it (SVD captures weight structure directly).

2. **Merging**: Full adapter merging requires manually computing `W_merged = W_residual + U @ R_U @ S @ R_V^T @ V^T` for each layer.

3. **Compatibility**: This implementation maintains the same training workflow as LoRA-GA while using a fundamentally different adapter mechanism.

4. **Method Selection**: Default uses "way0" (direct optimization). Other methods ("way1", "way2", "way3") can be selected via config.
