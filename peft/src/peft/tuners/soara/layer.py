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

"""
SOARA Layer Implementation

Implements the SOARA (Subspace Orthogonal Adaptation via Rotational Alignment) adapter layer.

The layer replaces a linear layer with:
    Y = X @ (W_residual + W_principal)
where:
    W_residual = U[:,r:] @ S[r:,r:] @ V[:,r:]^T  (frozen)
    W_principal = U[:,:r] @ R_U @ S[:r,:r] @ R_V^T @ V[:,:r]^T  (trainable via rotations)

Includes helper modules for rotation parameterization:
    - GivensRotationLayer: Parallel disjoint Givens rotations
    - ButterflyComponent: Single butterfly factorization level
    - ButterflyRotationLayer: Full butterfly composition (log(d) levels)
"""

from __future__ import annotations

import gc
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import SOARAConfig


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_givens_pairings(r: int, n_layers: int) -> list[list[tuple[int, int]]]:
    """
    Generate disjoint pairings for Givens rotations using round-robin tournament.

    Args:
        r: Dimension of rotation space.
        n_layers: Number of rotation layers to generate.

    Returns:
        List of rotation layers, each containing disjoint pairs.
    """
    if r <= 1:
        return []

    nodes = list(range(r))

    # Handle odd dimensions by adding dummy node
    if r % 2 == 1:
        nodes.append(-1)

    fixed_node = nodes[0]
    rotating_nodes = nodes[1:]

    all_pairings = []
    for _phase in range(min(n_layers, len(nodes) - 1)):
        current_pairs = []

        # Pair fixed node with first rotating node
        if rotating_nodes[0] != -1:
            current_pairs.append((fixed_node, rotating_nodes[0]))

        # Pair remaining nodes horizontally
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

    def __init__(self, dimension: int, pairs: list[tuple[int, int]]):
        super().__init__()
        self.dimension = dimension
        self.pairs = pairs

        p_indices = [p for p, q in pairs if p < dimension and q < dimension]
        q_indices = [q for p, q in pairs if p < dimension and q < dimension]

        self.register_buffer("p_indices", torch.tensor(p_indices, dtype=torch.long))
        self.register_buffer("q_indices", torch.tensor(q_indices, dtype=torch.long))

        self.thetas = nn.Parameter(torch.zeros(len(self.p_indices)))

    def forward(self) -> torch.Tensor:
        """Construct the rotation matrix from Givens rotations (vectorized)."""
        R = torch.eye(self.dimension, device=self.thetas.device, dtype=self.thetas.dtype)

        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)

        R.diagonal().scatter_(0, self.p_indices, cos_thetas)
        R.diagonal().scatter_(0, self.q_indices, cos_thetas)
        R[self.p_indices, self.q_indices] = -sin_thetas
        R[self.q_indices, self.p_indices] = sin_thetas

        return R

    def reset(self):
        """Reset all rotation angles to zero (identity rotation)."""
        with torch.no_grad():
            self.thetas.zero_()


class ButterflyComponent(nn.Module):
    """
    Single butterfly component B_tilde_b(d, k) from BOFT paper.

    Each block is parameterized as a Givens rotation (angle theta).

    Args:
        d: Full dimension of the orthogonal matrix R.
        k: Level size (power of 2). Determines block arrangement.
        block_size: Size b of the base orthogonal blocks (default 1).
    """

    def __init__(self, d: int, k: int, block_size: int = 1):
        super().__init__()
        self.d = d
        self.k = k
        self.block_size = block_size

        self.n_blocks = d // k
        self.rotations_per_block = k // 2
        self.total_rotations = self.n_blocks * self.rotations_per_block

        self.thetas = nn.Parameter(torch.zeros(self.total_rotations))
        self._precompute_indices()

    def _precompute_indices(self):
        p_indices = []
        q_indices = []

        for block_idx in range(self.n_blocks):
            base_idx = block_idx * self.k
            half_k = self.k // 2

            for i in range(half_k):
                p_indices.append(base_idx + i)
                q_indices.append(base_idx + i + half_k)

        self.register_buffer("p_indices", torch.tensor(p_indices, dtype=torch.long))
        self.register_buffer("q_indices", torch.tensor(q_indices, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        """Construct the butterfly component matrix."""
        R = torch.eye(self.d, device=self.thetas.device, dtype=self.thetas.dtype)

        cos_thetas = torch.cos(self.thetas)
        sin_thetas = torch.sin(self.thetas)

        R.diagonal().scatter_(0, self.p_indices, cos_thetas)
        R.diagonal().scatter_(0, self.q_indices, cos_thetas)
        R[self.p_indices, self.q_indices] = -sin_thetas
        R[self.q_indices, self.p_indices] = sin_thetas

        return R

    def reset(self):
        with torch.no_grad():
            self.thetas.zero_()


class ButterflyRotationLayer(nn.Module):
    """
    Full butterfly rotation matrix R(m, b) composing log2(d) butterfly components.

    R = B(d,d) @ B(d,d/2) @ ... @ B(d,2)

    Achieves O(d log d) parameters for a dense orthogonal matrix.

    Args:
        d: Dimension of the orthogonal matrix.
        block_size: Size b of base orthogonal blocks (default 1).
    """

    def __init__(self, d: int, block_size: int = 1):
        super().__init__()
        self.d = d
        self.block_size = block_size

        if d < 2 or (d & (d - 1)) != 0:
            d_padded = 2 ** math.ceil(math.log2(d))
            self.d_padded = d_padded
            self.needs_padding = True
        else:
            self.d_padded = d
            self.needs_padding = False

        self.m = int(math.log2(self.d_padded))

        self.components = nn.ModuleList()
        k = self.d_padded
        while k >= 2:
            self.components.append(ButterflyComponent(self.d_padded, k, block_size))
            k = k // 2

    def forward(self) -> torch.Tensor:
        """Compute the full butterfly rotation matrix R."""
        R = torch.eye(
            self.d_padded,
            device=self.components[0].thetas.device,
            dtype=self.components[0].thetas.dtype,
        )

        for component in self.components:
            R = R @ component()

        if self.needs_padding:
            R = R[: self.d, : self.d]

        return R

    def reset(self):
        for component in self.components:
            component.reset()

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# SOARA LAYER
# ============================================================================


class SOARALayer(BaseTunerLayer):
    """
    Base SOARA layer that tracks adapter metadata.

    This implements the BaseTunerLayer interface for SOARA, storing per-adapter
    rank, dropout, and references to the SOARA-specific modules.
    """

    # SOARA doesn't use ModuleDict-based adapter storage (unlike LoRA),
    # so adapter_layer_names is empty. We override set_adapter instead.
    adapter_layer_names = ()
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.soara_dropout = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"SOARA only supports nn.Linear layers, got {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def set_adapter(self, adapter_names, inference_mode=False):
        """Set active adapters, managing grad state for SOARA's per-adapter params."""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in self.r:
            is_active = adapter_name in adapter_names
            # Set grad state for all SOARA parameters for this adapter
            for attr_prefix in (
                "soara_S_", "soara_R_U_", "soara_R_V_",
                "soara_B_U_", "soara_B_V_", "soara_C_U_", "soara_C_V_",
                "soara_givens_u_", "soara_givens_v_",
                "soara_bf_u_", "soara_bf_v_",
                "soara_butterfly_u_", "soara_butterfly_v_",
            ):
                param = getattr(self, f"{attr_prefix}{adapter_name}", None)
                if param is not None:
                    if isinstance(param, nn.Parameter):
                        param.requires_grad_(is_active and not inference_mode)
                    elif isinstance(param, nn.Module):
                        param.requires_grad_(is_active and not inference_mode)

        self._active_adapter = adapter_names


class SOARALinear(nn.Module, SOARALayer):
    """
    SOARA adapter implemented as a linear layer replacement.

    Decomposes the original weight via SVD, freezes the residual, and learns
    rotation matrices R_U and R_V in the principal subspace.

    Forward computation:
        result = base_layer(x) + x @ V^T @ R_V @ S @ R_U^T @ U^T
    where base_layer contains W_residual and the second term is W_principal.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        soara_config: SOARAConfig,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        SOARALayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, soara_config)

    def update_layer(self, adapter_name: str, soara_config: SOARAConfig) -> None:
        """Initialize the SOARA adapter for the given layer."""
        # Cap rank to layer's maximum possible rank
        max_rank = min(self.in_features, self.out_features)
        r = min(soara_config.r, max_rank)
        self.r[adapter_name] = r

        # Dropout
        if soara_config.soara_dropout > 0.0:
            dropout_layer = nn.Dropout(p=soara_config.soara_dropout)
        else:
            dropout_layer = nn.Identity()
        self.soara_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        # Store config reference for forward/merge
        setattr(self, f"_soara_config_{adapter_name}", soara_config)

        # Perform SVD decomposition on base weight
        self._initialize_svd_components(adapter_name, r, soara_config)

        # Initialize rotation matrices based on selected method
        self._initialize_rotation_matrices(adapter_name, r, soara_config)

        self.set_adapter(self.active_adapters)

    def _initialize_svd_components(self, adapter_name: str, r: int, config: SOARAConfig) -> None:
        """Perform SVD on base layer weight and store principal/residual components."""
        with torch.no_grad():
            base_layer = self.get_base_layer()
            original_weight = base_layer.weight.data.clone()

            original_dtype = original_weight.dtype
            original_device = original_weight.device
            if original_weight.dtype in (torch.bfloat16, torch.float16):
                original_weight = original_weight.float()

            # SVD: W = U @ S @ V^T
            U, S, V = torch.linalg.svd(original_weight, full_matrices=False)

            del original_weight
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Split into principal (top-r) and residual components
            U_principal = U[:, :r].contiguous()
            U_residual = U[:, r:].contiguous()
            S_principal = S[:r].contiguous()
            S_residual = S[r:].contiguous()
            V_principal = V[:r, :].contiguous()
            V_residual = V[r:, :].contiguous()

            del U, S, V
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute residual: W_residual = U[:,r:] @ diag(S[r:]) @ V[r:,:]
            if U_residual.shape[1] > 0:
                S_residual_diag = torch.diag(S_residual)
                W_residual = U_residual @ S_residual_diag @ V_residual
                del S_residual_diag
            else:
                W_residual = torch.zeros_like(base_layer.weight.data)

            del U_residual, S_residual, V_residual
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Store W_residual in base layer (frozen)
            base_layer.weight.data = W_residual.to(dtype=original_dtype, device=original_device)
            base_layer.weight.requires_grad = False

            del W_residual
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Store principal U, V as frozen buffers
            self.register_buffer(
                f"soara_U_{adapter_name}",
                U_principal.to(dtype=original_dtype, device=original_device),
            )
            self.register_buffer(
                f"soara_V_{adapter_name}",
                V_principal.to(dtype=original_dtype, device=original_device),
            )

            # S: Use FP32 if s_dtype_fp32 is enabled
            s_dtype = torch.float32 if config.s_dtype_fp32 else original_dtype
            S_converted = S_principal.to(dtype=s_dtype, device=original_device)

            if config.freeze_singular_values:
                self.register_buffer(f"soara_S_{adapter_name}", S_converted)
            else:
                setattr(self, f"soara_S_{adapter_name}", nn.Parameter(S_converted))

    def _initialize_rotation_matrices(self, adapter_name: str, r: int, config: SOARAConfig) -> None:
        """Initialize rotation matrices R_U and R_V based on the selected method."""
        # Get device and dtype from U buffer
        U = getattr(self, f"soara_U_{adapter_name}")
        device = U.device
        dtype = U.dtype

        if config.method == "v1":
            self._init_v1(adapter_name, r, config, device, dtype)
        elif config.method == "V2":
            self._init_v2(adapter_name, r, config, device)
        elif config.method == "v3":
            self._init_v3(adapter_name, r, config, device, dtype)
        elif config.method == "V4":
            self._init_v4(adapter_name, r, config, device, dtype)

    def _init_v1(self, adapter_name, r, config, device, dtype):
        """V1: Direct parameterization with orthogonality regularization."""
        if config.init_identity:
            R_U = nn.Parameter(torch.eye(r, dtype=dtype, device=device))
            R_V = nn.Parameter(torch.eye(r, dtype=dtype, device=device))
        else:
            R_U_init = torch.randn(r, r, dtype=dtype, device=device)
            R_V_init = torch.randn(r, r, dtype=dtype, device=device)
            Q_U, _ = torch.linalg.qr(R_U_init)
            Q_V, _ = torch.linalg.qr(R_V_init)
            R_U = nn.Parameter(Q_U)
            R_V = nn.Parameter(Q_V)

        setattr(self, f"soara_R_U_{adapter_name}", R_U)
        setattr(self, f"soara_R_V_{adapter_name}", R_V)

    def _init_v2(self, adapter_name, r, config, device):
        """V2: Givens rotations or Butterfly factorization."""
        if config.use_butterfly:
            if config.butterfly_sequential:
                d_padded = 2 ** math.ceil(math.log2(r)) if (r & (r - 1)) != 0 else r
                k_values = []
                k = d_padded
                while k >= 2:
                    k_values.append(k)
                    k = k // 2

                setattr(self, f"_bf_k_values_{adapter_name}", k_values)
                setattr(self, f"_bf_d_padded_{adapter_name}", d_padded)
                setattr(self, f"_bf_needs_padding_{adapter_name}", d_padded != r)
                setattr(self, f"_bf_idx_{adapter_name}", 0)
                setattr(self, f"_bf_cycle_{adapter_name}", 0)

                first_k = k_values[0]
                block_size = config.butterfly_block_size
                self.add_module(
                    f"soara_bf_u_{adapter_name}",
                    ButterflyComponent(d_padded, first_k, block_size).to(device),
                )
                self.add_module(
                    f"soara_bf_v_{adapter_name}",
                    ButterflyComponent(d_padded, first_k, block_size).to(device),
                )
            else:
                block_size = config.butterfly_block_size
                self.add_module(
                    f"soara_butterfly_u_{adapter_name}",
                    ButterflyRotationLayer(r, block_size).to(device),
                )
                self.add_module(
                    f"soara_butterfly_v_{adapter_name}",
                    ButterflyRotationLayer(r, block_size).to(device),
                )
        else:
            n_layers = config.n_givens_layers or (r - 1)
            pairings = generate_givens_pairings(r, n_layers)
            setattr(self, f"_givens_pairings_{adapter_name}", pairings)
            setattr(self, f"_givens_idx_{adapter_name}", 0)
            setattr(self, f"_givens_cycle_{adapter_name}", 0)

            self.add_module(
                f"soara_givens_u_{adapter_name}",
                GivensRotationLayer(r, pairings[0]).to(device),
            )
            self.add_module(
                f"soara_givens_v_{adapter_name}",
                GivensRotationLayer(r, pairings[0]).to(device),
            )

    def _init_v3(self, adapter_name, r, config, device, dtype):
        """V3: Low-rank skew-symmetric perturbation I + BC^T - CB^T."""
        lr_r = config.low_rank_r

        setattr(
            self,
            f"soara_B_U_{adapter_name}",
            nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
        )
        setattr(
            self,
            f"soara_B_V_{adapter_name}",
            nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
        )

        if config.init_identity:
            setattr(
                self,
                f"soara_C_U_{adapter_name}",
                nn.Parameter(torch.zeros(r, lr_r, dtype=dtype, device=device)),
            )
            setattr(
                self,
                f"soara_C_V_{adapter_name}",
                nn.Parameter(torch.zeros(r, lr_r, dtype=dtype, device=device)),
            )
        else:
            setattr(
                self,
                f"soara_C_U_{adapter_name}",
                nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
            )
            setattr(
                self,
                f"soara_C_V_{adapter_name}",
                nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
            )

    def _init_v4(self, adapter_name, r, config, device, dtype):
        """V4: Exponential map of skew-symmetric matrix exp(BC^T - CB^T)."""
        lr_r = config.low_rank_r

        setattr(
            self,
            f"soara_B_U_{adapter_name}",
            nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
        )
        setattr(
            self,
            f"soara_B_V_{adapter_name}",
            nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
        )

        if config.init_identity:
            setattr(
                self,
                f"soara_C_U_{adapter_name}",
                nn.Parameter(torch.zeros(r, lr_r, dtype=dtype, device=device)),
            )
            setattr(
                self,
                f"soara_C_V_{adapter_name}",
                nn.Parameter(torch.zeros(r, lr_r, dtype=dtype, device=device)),
            )
        else:
            setattr(
                self,
                f"soara_C_U_{adapter_name}",
                nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
            )
            setattr(
                self,
                f"soara_C_V_{adapter_name}",
                nn.Parameter(torch.randn(r, lr_r, dtype=dtype, device=device) * 0.1),
            )

    def _get_rotation_matrices(
        self, adapter_name: str, config: SOARAConfig, r: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the current rotation matrices R_U and R_V."""
        if config.method == "v1":
            R_U = getattr(self, f"soara_R_U_{adapter_name}")
            R_V = getattr(self, f"soara_R_V_{adapter_name}")
            return R_U, R_V

        elif config.method == "V2":
            if config.use_butterfly:
                if config.butterfly_sequential:
                    bf_u = getattr(self, f"soara_bf_u_{adapter_name}")
                    bf_v = getattr(self, f"soara_bf_v_{adapter_name}")
                    R_U = bf_u()
                    R_V = bf_v()
                    needs_padding = getattr(self, f"_bf_needs_padding_{adapter_name}")
                    if needs_padding:
                        R_U = R_U[:r, :r]
                        R_V = R_V[:r, :r]
                    return R_U, R_V
                else:
                    butterfly_u = getattr(self, f"soara_butterfly_u_{adapter_name}")
                    butterfly_v = getattr(self, f"soara_butterfly_v_{adapter_name}")
                    return butterfly_u(), butterfly_v()
            else:
                givens_u = getattr(self, f"soara_givens_u_{adapter_name}")
                givens_v = getattr(self, f"soara_givens_v_{adapter_name}")
                return givens_u(), givens_v()

        elif config.method == "v3":
            B_U = getattr(self, f"soara_B_U_{adapter_name}")
            C_U = getattr(self, f"soara_C_U_{adapter_name}")
            B_V = getattr(self, f"soara_B_V_{adapter_name}")
            C_V = getattr(self, f"soara_C_V_{adapter_name}")

            I = torch.eye(r, device=B_U.device, dtype=B_U.dtype)
            R_U = I + B_U @ C_U.T - C_U @ B_U.T
            R_V = I + B_V @ C_V.T - C_V @ B_V.T
            return R_U, R_V

        elif config.method == "V4":
            B_U = getattr(self, f"soara_B_U_{adapter_name}")
            C_U = getattr(self, f"soara_C_U_{adapter_name}")
            B_V = getattr(self, f"soara_B_V_{adapter_name}")
            C_V = getattr(self, f"soara_C_V_{adapter_name}")

            skew_U = B_U @ C_U.T - C_U @ B_U.T
            skew_V = B_V @ C_V.T - C_V @ B_V.T

            orig_dtype = skew_U.dtype
            R_U = torch.matrix_exp(skew_U.float()).to(dtype=orig_dtype)
            R_V = torch.matrix_exp(skew_V.float()).to(dtype=orig_dtype)
            return R_U, R_V

        raise ValueError(f"Unknown method: {config.method}")

    def get_orthogonality_loss(self, adapter_name: str, config: SOARAConfig) -> torch.Tensor:
        """Compute orthogonality regularization loss (V1 only)."""
        device = next(self.parameters()).device

        if config.orthogonality_reg_weight == 0:
            return torch.tensor(0.0, device=device)

        if config.method != "v1":
            return torch.tensor(0.0, device=device)

        R_U = getattr(self, f"soara_R_U_{adapter_name}")
        R_V = getattr(self, f"soara_R_V_{adapter_name}")
        r = self.r[adapter_name]

        reg_type = config.regularization_type

        if reg_type == "frobenius":
            I = torch.eye(r, device=R_U.device, dtype=R_U.dtype)
            loss_u = torch.norm(R_U.T @ R_U - I, p="fro") ** 2
            loss_v = torch.norm(R_V.T @ R_V - I, p="fro") ** 2
        elif reg_type == "determinant":
            det_u = torch.det(R_U.float())
            det_v = torch.det(R_V.float())
            loss_u = (det_u - 1.0) ** 2
            loss_v = (det_v - 1.0) ** 2
        elif reg_type == "log_determinant":
            log_det_u = torch.logdet(R_U.float())
            log_det_v = torch.logdet(R_V.float())
            loss_u = log_det_u**2
            loss_v = log_det_v**2
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

        return (loss_u + loss_v) * config.orthogonality_reg_weight

    def step_phase(self, adapter_name: str, config: SOARAConfig) -> tuple[int, int]:
        """
        Advance to next phase for V2 (sequential Givens/butterfly training).

        Merges current rotation into U/V, deletes old layer, creates next one.

        Returns:
            (params_before, params_after) for tracking parameter counts.
        """
        if config.method != "V2":
            return (0, 0)

        r = self.r[adapter_name]

        if config.use_butterfly and not config.butterfly_sequential:
            return (0, 0)

        if config.use_butterfly and config.butterfly_sequential:
            return self._step_butterfly_sequential(adapter_name, config, r)

        return self._step_givens_sequential(adapter_name, config, r)

    def _step_givens_sequential(self, adapter_name, config, r):
        """Step through sequential Givens phases."""
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        U = getattr(self, f"soara_U_{adapter_name}")
        V = getattr(self, f"soara_V_{adapter_name}")

        with torch.no_grad():
            givens_u = getattr(self, f"soara_givens_u_{adapter_name}")
            givens_v = getattr(self, f"soara_givens_v_{adapter_name}")
            R_U = givens_u()
            R_V = givens_v()

            U.copy_(U @ R_U)
            V.copy_(R_V @ V)

            delattr(self, f"soara_givens_u_{adapter_name}")
            delattr(self, f"soara_givens_v_{adapter_name}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pairings = getattr(self, f"_givens_pairings_{adapter_name}")
            idx = getattr(self, f"_givens_idx_{adapter_name}")
            cycle = getattr(self, f"_givens_cycle_{adapter_name}")

            idx = (idx + 1) % len(pairings)
            if idx == 0:
                cycle += 1

            setattr(self, f"_givens_idx_{adapter_name}", idx)
            setattr(self, f"_givens_cycle_{adapter_name}", cycle)

            device = U.device
            next_u = GivensRotationLayer(r, pairings[idx]).to(device)
            next_v = GivensRotationLayer(r, pairings[idx]).to(device)

            self.add_module(f"soara_givens_u_{adapter_name}", next_u)
            self.add_module(f"soara_givens_v_{adapter_name}", next_v)

        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (params_before, params_after)

    def _step_butterfly_sequential(self, adapter_name, config, r):
        """Step through sequential butterfly phases."""
        params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)

        U = getattr(self, f"soara_U_{adapter_name}")
        V = getattr(self, f"soara_V_{adapter_name}")

        with torch.no_grad():
            bf_u = getattr(self, f"soara_bf_u_{adapter_name}")
            bf_v = getattr(self, f"soara_bf_v_{adapter_name}")
            R_U = bf_u()
            R_V = bf_v()

            needs_padding = getattr(self, f"_bf_needs_padding_{adapter_name}")
            if needs_padding:
                R_U = R_U[:r, :r]
                R_V = R_V[:r, :r]

            U.copy_(U @ R_U)
            V.copy_(R_V @ V)

            delattr(self, f"soara_bf_u_{adapter_name}")
            delattr(self, f"soara_bf_v_{adapter_name}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            k_values = getattr(self, f"_bf_k_values_{adapter_name}")
            d_padded = getattr(self, f"_bf_d_padded_{adapter_name}")
            idx = getattr(self, f"_bf_idx_{adapter_name}")
            cycle = getattr(self, f"_bf_cycle_{adapter_name}")

            idx = (idx + 1) % len(k_values)
            if idx == 0:
                cycle += 1

            setattr(self, f"_bf_idx_{adapter_name}", idx)
            setattr(self, f"_bf_cycle_{adapter_name}", cycle)

            device = U.device
            next_k = k_values[idx]
            block_size = config.butterfly_block_size

            next_u = ButterflyComponent(d_padded, next_k, block_size).to(device)
            next_v = ButterflyComponent(d_padded, next_k, block_size).to(device)

            self.add_module(f"soara_bf_u_{adapter_name}", next_u)
            self.add_module(f"soara_bf_v_{adapter_name}", next_v)

        params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (params_before, params_after)

    def get_delta_weight(self, adapter_name: str, config: SOARAConfig) -> torch.Tensor:
        """
        Compute the delta weight W_principal for the given adapter.

        W_principal = U @ R_U @ diag(S) @ R_V^T @ V
        """
        r = self.r[adapter_name]
        U = getattr(self, f"soara_U_{adapter_name}")
        V = getattr(self, f"soara_V_{adapter_name}")
        S = getattr(self, f"soara_S_{adapter_name}")

        R_U, R_V = self._get_rotation_matrices(adapter_name, config, r)

        # Ensure compatible dtypes
        target_dtype = U.dtype
        if R_U.dtype != target_dtype:
            R_U = R_U.to(target_dtype)
        if R_V.dtype != target_dtype:
            R_V = R_V.to(target_dtype)
        if S.dtype != target_dtype:
            S = S.to(target_dtype)

        # W_principal = U @ R_U @ diag(S) @ R_V^T @ V
        delta = U @ R_U @ torch.diag(S) @ R_V.T @ V

        return delta

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge active adapter weights into the base weight."""
        from peft.tuners.tuners_utils import check_adapters_to_merge

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.r:
                base_layer = self.get_base_layer()
                # We need the config to compute delta weight - store it during update_layer
                config = getattr(self, f"_soara_config_{active_adapter}", None)
                if config is None:
                    warnings.warn(
                        f"Could not find SOARA config for adapter '{active_adapter}'. Skipping merge."
                    )
                    continue

                delta_weight = self.get_delta_weight(active_adapter, config)

                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += delta_weight
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge previously merged adapter weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.r:
                config = getattr(self, f"_soara_config_{active_adapter}", None)
                if config is not None:
                    self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter, config)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.r:
                    continue

                config = getattr(self, f"_soara_config_{active_adapter}", None)
                if config is None:
                    continue

                r = self.r[active_adapter]
                U = getattr(self, f"soara_U_{active_adapter}")
                V = getattr(self, f"soara_V_{active_adapter}")
                S = getattr(self, f"soara_S_{active_adapter}")

                target_dtype = x.dtype

                U_current = U.to(target_dtype) if U.dtype != target_dtype else U
                V_current = V.to(target_dtype) if V.dtype != target_dtype else V
                S_current = S.to(target_dtype) if S.dtype != target_dtype else S

                R_U, R_V = self._get_rotation_matrices(active_adapter, config, r)
                if R_U.dtype != target_dtype:
                    R_U = R_U.to(target_dtype)
                if R_V.dtype != target_dtype:
                    R_V = R_V.to(target_dtype)

                dropout = self.soara_dropout[active_adapter]
                x_adapted = dropout(x)

                # Forward: x @ V^T @ R_V @ S @ R_U^T @ U^T
                x_adapted = x_adapted @ V_current.T
                x_adapted = x_adapted @ R_V.T
                x_adapted = x_adapted * S_current
                x_adapted = x_adapted @ R_U.T
                x_adapted = x_adapted @ U_current.T

                result = result + x_adapted

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "soara." + rep
