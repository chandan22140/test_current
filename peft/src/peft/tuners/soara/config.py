# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SOARAConfig(PeftConfig):
    """
    Configuration class for the SOARA (Subspace Orthogonal Adaptation via Rotational Alignment) tuner.

    SOARA adapts pretrained weights by decomposing them via SVD, then learning rotation matrices
    in the principal subspace while keeping the residual frozen (optionally NF4-quantized).

    Mathematical Framework:
        1. Original weight W is decomposed: W = U @ S @ V^T
        2. Split into principal (top-r) and residual components
        3. W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T (frozen, optionally NF4 quantized)
        4. W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T (trainable via rotations)
        5. Final output: Y = X @ (W_residual + W_principal)

    Rotation Parameterization Methods:
        - **v1** (SOARA-V1): Direct parameterization with orthogonality regularization
        - **V2** (SOARA-V2): Exact orthogonality via Givens rotations or Butterfly factorizations
        - **v3**: Low-rank skew-symmetric perturbation: R = I + BC^T - CB^T
        - **V4**: Exponential map of skew-symmetric matrix: R = exp(BC^T - CB^T)

    Args:
        r (`int`, *optional*, defaults to `16`):
            SOARA rank (number of principal singular components to adapt).
        target_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to apply SOARA to. Only `nn.Linear` layers are supported.
        method (`str`, *optional*, defaults to `"v1"`):
            Rotation parameterization method. One of `"v1"`, `"V2"`, `"v3"`, `"V4"`.
        soara_dropout (`float`, *optional*, defaults to `0.0`):
            Dropout probability applied to the adaptation path. Should generally be 0 for SOARA.
        orthogonality_reg_weight (`float`, *optional*, defaults to `1e-4`):
            Weight for orthogonality regularization loss (V1 only).
        regularization_type (`str`, *optional*, defaults to `"frobenius"`):
            Type of orthogonality regularization for V1. One of `"frobenius"`, `"determinant"`,
            `"log_determinant"`.
        n_givens_layers (`int`, *optional*):
            Number of Givens rotation layers for V2 (default: r-1).
        steps_per_phase (`int`, *optional*, defaults to `100`):
            Training steps per sequential Givens/butterfly phase (V2 only).
        total_cycles (`int`, *optional*, defaults to `3`):
            Total cycles through all sequential layers (V2 only).
        use_butterfly (`bool`, *optional*, defaults to `False`):
            Use butterfly factorization instead of Givens rotations (V2 only).
        butterfly_sequential (`bool`, *optional*, defaults to `False`):
            Train butterfly components sequentially (V2 only).
        butterfly_block_size (`int`, *optional*, defaults to `1`):
            Block size for butterfly factorization (V2 only).
        low_rank_r (`int`, *optional*, defaults to `4`):
            Low rank for B, C matrices in V3/V4 methods.
        init_identity (`bool`, *optional*, defaults to `True`):
            Initialize rotation matrices as identity.
        freeze_singular_values (`bool`, *optional*, defaults to `False`):
            Whether to freeze the principal singular values S.
        s_dtype_fp32 (`bool`, *optional*, defaults to `True`):
            Store singular values in FP32 for precision (prevents bfloat16 rounding issues).
        quantize_residual (`bool`, *optional*, defaults to `False`):
            Whether to NF4-quantize the residual weight matrix W_residual.
        quantize_base_components (`bool`, *optional*, defaults to `False`):
            Whether to NF4-quantize the principal U and V matrices.
        modules_to_save (`List[str]`, *optional*):
            List of modules apart from SOARA layers to be set as trainable and saved.
        layers_to_transform (`Union[List[int], int]`, *optional*):
            The layer indexes to transform.
        layers_pattern (`Optional[Union[List[str], str]]`, *optional*):
            The layer pattern name, used with `layers_to_transform`.
    """

    r: int = field(default=16, metadata={"help": "SOARA rank (number of principal SVD components)"})

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SOARA. "
                "Only nn.Linear layers are supported."
            )
        },
    )

    # Rotation parameterization method
    method: str = field(
        default="v1",
        metadata={
            "help": (
                "Rotation parameterization method. Options: "
                "'v1' (direct + regularization), 'V2' (Givens/butterfly), "
                "'v3' (skew-symmetric perturbation), 'V4' (matrix exponential)."
            )
        },
    )

    # Dropout
    soara_dropout: float = field(default=0.0, metadata={"help": "Dropout for SOARA adaptation path"})

    # V1 specific parameters
    orthogonality_reg_weight: float = field(
        default=1e-4,
        metadata={"help": "Weight for orthogonality regularization loss (V1 only)"},
    )
    regularization_type: str = field(
        default="frobenius",
        metadata={"help": "Regularization type for V1: 'frobenius', 'determinant', or 'log_determinant'"},
    )

    # V2 specific parameters
    n_givens_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of Givens rotation layers for V2 (default: r-1)"},
    )
    steps_per_phase: int = field(
        default=100,
        metadata={"help": "Training steps per sequential phase (V2 only)"},
    )
    total_cycles: int = field(
        default=3,
        metadata={"help": "Total cycles through all sequential layers (V2 only)"},
    )
    use_butterfly: bool = field(
        default=False,
        metadata={"help": "Use butterfly factorization instead of Givens rotations (V2 only)"},
    )
    butterfly_sequential: bool = field(
        default=False,
        metadata={"help": "Train butterfly components sequentially (V2 only)"},
    )
    butterfly_block_size: int = field(
        default=1,
        metadata={"help": "Block size for butterfly factorization (V2 only)"},
    )

    # V3/V4 specific parameters
    low_rank_r: int = field(
        default=4,
        metadata={"help": "Low rank for B, C matrices in V3/V4 methods"},
    )

    # General parameters
    init_identity: bool = field(
        default=True,
        metadata={"help": "Initialize rotation matrices as identity"},
    )
    freeze_singular_values: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the principal singular values S"},
    )
    s_dtype_fp32: bool = field(
        default=True,
        metadata={"help": "Store singular values in FP32 for optimizer precision"},
    )

    # Quantization parameters
    quantize_residual: bool = field(
        default=False,
        metadata={"help": "NF4-quantize the residual weight matrix W_residual"},
    )
    quantize_base_components: bool = field(
        default=False,
        metadata={"help": "NF4-quantize the principal U and V matrices"},
    )

    # Standard PEFT parameters
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from SOARA layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform. If specified, SOARA will only be applied to the specified layers."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different from `None`."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SOARA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # Validate method
        valid_methods = {"v1", "V2", "v3", "V4"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid SOARA method '{self.method}'. Must be one of {valid_methods}.")
        # Validate regularization type
        valid_reg_types = {"frobenius", "determinant", "log_determinant"}
        if self.regularization_type not in valid_reg_types:
            raise ValueError(
                f"Invalid regularization_type '{self.regularization_type}'. Must be one of {valid_reg_types}."
            )
        # Validate layers_pattern usage
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
