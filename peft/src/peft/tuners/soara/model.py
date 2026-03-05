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
SOARA Model

Integrates SOARA adapter layers into a pretrained model via the PEFT BaseTuner interface.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from ..tuners_utils import _maybe_include_all_linear_layers
from .config import SOARAConfig
from .layer import SOARALayer, SOARALinear


class SOARAModel(BaseTuner):
    """
    Creates a SOARA (Subspace Orthogonal Adaptation via Rotational Alignment) model
    from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SOARAConfig`]): The configuration of the SOARA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The SOARA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import SOARAConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = SOARAConfig(r=16, target_modules=["q_proj", "v_proj"])
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SOARAConfig`]): The configuration of the SOARA model.
    """

    prefix: str = "soara_"
    tuner_layer_cls = SOARALayer
    # Reuse LoRA target module mappings for common architectures
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peft_config: SOARAConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, SOARALinear):
            # Additional adapter on an existing SOARA layer
            target.update_layer(adapter_name, peft_config)
        else:
            # Create a new SOARA module to replace the target
            new_module = self._create_new_module(peft_config, adapter_name, target)
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        soara_config: SOARAConfig,
        adapter_name: str,
        target: nn.Module,
    ) -> SOARALinear:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        new_module = SOARALinear(
            target,
            adapter_name=adapter_name,
            soara_config=soara_config,
        )
        return new_module

    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute the total orthogonality regularization loss across all SOARA layers.

        This is only non-zero for V1 method with orthogonality_reg_weight > 0.

        Returns:
            Scalar tensor with the total regularization loss.
        """
        total_loss = torch.tensor(0.0)
        device = None

        for module in self.model.modules():
            if isinstance(module, SOARALinear):
                for adapter_name in module.active_adapters:
                    if adapter_name not in module.r:
                        continue
                    config = getattr(module, f"_soara_config_{adapter_name}", None)
                    if config is None:
                        continue
                    loss = module.get_orthogonality_loss(adapter_name, config)
                    if device is None:
                        device = loss.device
                        total_loss = total_loss.to(device)
                    total_loss = total_loss + loss

        return total_loss

    def step_all_phases(self) -> dict[str, tuple[int, int]]:
        """
        Step all SOARA layers to their next sequential phase (V2 only).

        Call this periodically during training (every `steps_per_phase` steps)
        for V2 method with sequential Givens or butterfly training.

        Returns:
            Dict mapping module names to (params_before, params_after) tuples.
        """
        results = {}
        for name, module in self.model.named_modules():
            if isinstance(module, SOARALinear):
                for adapter_name in module.active_adapters:
                    if adapter_name not in module.r:
                        continue
                    config = getattr(module, f"_soara_config_{adapter_name}", None)
                    if config is None:
                        continue
                    result = module.step_phase(adapter_name, config)
                    if result != (0, 0):
                        results[f"{name}.{adapter_name}"] = result
        return results
