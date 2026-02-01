from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
# CHANGED: Import rotational PiSSA layer instead of PEFT LoraLinear
# from peft.tuners.lora.layer import Linear as LoraLinear
from rotational_pissa_unified import RotationalLinearLayer

# include_keywords = ["block.0", "block.4"]
include_keywords = ["encoder.block.2", "encoder.block.3", "encoder.block.4"]  # for T5
# include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False


def get_forward_hook(name):
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )

    return hook


class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        # CHANGED: Add pissa_config for orthogonality regularization
        pissa_config = None,
        # CHANGED: Add s_lr_multiplier for separate S learning rate
        s_lr_multiplier: float = 10.0,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # CHANGED: Detect rotational PiSSA model instead of PEFT model
        self.is_peft = any(isinstance(m, RotationalLinearLayer) for m in model.modules())
        
        # CHANGED: Store pissa_config for orthogonality regularization
        self.pissa_config = pissa_config
        
        # CHANGED: Store s_lr_multiplier for separate S learning rate
        self.s_lr_multiplier = s_lr_multiplier
        
        if self.is_peft:
            # CHANGED: Get scaling from rotational PiSSA layer
            for name, module in model.named_modules():
                if isinstance(module, RotationalLinearLayer):
                    self.scaling = module.scaling
                    break
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0

    def create_optimizer(self):
        """Create optimizer with separate learning rate for S parameters.
        
        S (singular values) typically have values ~1-16, while R_U/R_V matrices
        have values ~0-1. With the same LR, S gets proportionally tiny updates.
        This method gives S a higher LR (s_lr_multiplier × base_lr).
        """
        if self.optimizer is not None:
            return self.optimizer
        
        # Separate S parameters from other trainable parameters
        s_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Match S parameters (e.g., "model.layers.0.self_attn.q_proj.S")
                if name.endswith('.S'):
                    s_params.append(param)
                else:
                    other_params.append(param)
        
        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        
        # Create optimizer with separate param groups
        optimizer_grouped_parameters = []
        
        if other_params:
            optimizer_grouped_parameters.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
            })
        
        if s_params:
            optimizer_grouped_parameters.append({
                'params': s_params,
                'lr': base_lr * self.s_lr_multiplier,  # Higher LR for S
                'weight_decay': 0.0,  # No weight decay for singular values
            })
            print(f"[LogTrainer] S params: {len(s_params)}, LR: {base_lr * self.s_lr_multiplier:.2e} ({self.s_lr_multiplier}x)")
            print(f"[LogTrainer] Other params: {len(other_params)}, LR: {base_lr:.2e}")
        
        # Use the optimizer class from args
        # Using eps=1e-10 instead of default 1e-8 for better precision with small gradients
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-10)
        
        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        if not do_log:
            # CHANGED: Add orthogonality regularization for rotational PiSSA
            if self.is_peft and self.pissa_config is not None:
                return self._training_step_with_pissa_reg(model, inputs, num_items_in_batch)
            
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except TypeError:
                # Fallback for older transformers versions that don't accept num_items_in_batch
                print("⚠️  Fallback for older transformers versions that don't accept num_items_in_batch")
                return super().training_step(model, inputs)

        print("⚠️  Fallback for do_log=True")
        # Original logging code (unchanged when do_log=True)
        if self.is_peft:
            if self.orig_A is None:
                self.orig_A = {}
                self.orig_B = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        # CHANGED: Adapt to rotational PiSSA parameter names
                        # For way0: R_U, R_V
                        # For way2/3: B_U, C_U, B_V, C_V
                        if "R_U" in name or "B_U" in name:
                            self.orig_A[name.split("R_U.")[0] if "R_U" in name else name.split("B_U.")[0]] = (
                                param.detach().clone()
                            )
                        elif "R_V" in name or "B_V" in name:
                            self.orig_B[name.split("R_V.")[0] if "R_V" in name else name.split("B_V.")[0]] = (
                                param.detach().clone()
                            )
                for name, module in model.named_modules():
                    if any([kw in name for kw in include_keywords]) and isinstance(
                        module, RotationalLinearLayer
                    ):
                        breakpoint()
                        hook = get_forward_hook(name)
                        module.register_forward_hook(hook)
        else:
            if self.orig_W is None:
                self.orig_W = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        self.orig_W[name] = param.detach().clone()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        with torch.no_grad():
            if (
                self.gradient_accumulation_counter
                % self.args.gradient_accumulation_steps
                == self.args.gradient_accumulation_steps - 1
            ):
                if self.is_peft:
                    # CHANGED: Log rotational PiSSA parameters instead of LoRA A/B
                    # This is complex and method-dependent, keeping simplified version
                    param_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and any(
                            [kw in name for kw in include_keywords]
                        ):
                            param_dict[name] = param
                    
                    # Log parameter norms
                    for name, param in param_dict.items():
                        if param.grad is not None:
                            wandb.log(
                                {
                                    f"param_norm/{name}": torch.norm(param).item(),
                                    f"grad_norm/{name}": torch.norm(param.grad).item(),
                                    "train/global_step": self.state.global_step,
                                },
                                commit=False,
                            )
                else:
                    W_dict = {}
                    for name, param in model.named_parameters():
                        if (
                            param.requires_grad
                            and any([kw in name for kw in include_keywords])
                            and len(param.shape) == 2
                        ):
                            W_dict[name] = param
                    for key in W_dict.keys():
                        W = W_dict[key]
                        W_grad = W.grad
                        W_0 = self.orig_W[key]
                        W_diff = W - W_0
                        W_diff_norm = torch.norm(W_diff).item()
                        W_norm = torch.norm(W).item()
                        W_grad_norm = torch.norm(W_grad).item()
                        U, S, V = torch.svd(W_diff.float())
                        top_1_ratio = S[0] / S.sum()
                        top_4_ratio = S[:4].sum() / S.sum()
                        wandb.log(
                            {
                                f"W_norm/{key}": W_norm,
                                f"W_grad_norm/{key}": W_grad_norm,
                                f"W_diff_norm/{key}": W_diff_norm,
                                "train/global_step": self.state.global_step,
                                f"W_top_1_ratio/{key}": top_1_ratio.item(),
                                f"W_top_4_ratio/{key}": top_4_ratio.item(),
                            }
                        )
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps

    # CHANGED: New method for training step with PiSSA regularization
    def _training_step_with_pissa_reg(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """Training step with orthogonality regularization for rotational PiSSA."""
        if torch.cuda.is_available():
             torch.cuda.reset_peak_memory_stats()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            if num_items_in_batch is None:
                loss = self.compute_loss(model, inputs)
            else:
                # Use num_items_in_batch if compute_loss supports/needs it?
                # Actually compute_loss signature is (model, inputs, return_outputs=False, num_items_in_batch=None)
                # But let's check if we can pass it safely.
                # Inspect compute_loss signature to be safe.
                import inspect
                if "num_items_in_batch" in inspect.getfullargspec(self.compute_loss).args:
                     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                else:
                     loss = self.compute_loss(model, inputs)

            # DEBUG: Print raw loss values
            # print(f"[DEBUG] raw_loss={loss.item():.6f}, loss_dtype={loss.dtype}")
            
            # Add orthogonality regularization for way0
            if self.pissa_config.method == "way0":
                ortho_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                for module in model.modules():
                    if isinstance(module, RotationalLinearLayer):
                        ortho_loss = ortho_loss + module.get_orthogonality_loss()
                
                # Combine losses
                total_loss = loss + ortho_loss
                
                # Log orthogonality loss
                if self.state.global_step % self.args.logging_steps == 0:
                    mem_metrics = {}
                    if torch.cuda.is_available():
                        mem_metrics = {
                            "gpu/peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                            "gpu/current_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                        }
                    
                    wandb.log({
                        "train/ortho_loss": ortho_loss.item(),
                        "train/task_loss": loss.item(),
                        "train/total_loss": total_loss.item(),
                        **mem_metrics,
                    }, commit=False)

                    # print(f"[DEBUG] ortho_loss={ortho_loss.item():.6f}, task_loss={loss.item():.6f}")
            else:
                total_loss = loss

        # DEBUG: Print total loss before any normalization
        # print(f"[DEBUG] total_loss_before_norm={total_loss.item():.6f}, GA_steps={self.args.gradient_accumulation_steps}")

        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()

        # FIX: Do backward with ORIGINAL loss, then divide for logging
        # This prevents bf16 precision loss from setting gradients to 0
        self.accelerator.backward(total_loss)

        # DEBUG: Check for NaN in parameters/gradients after backward
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"[NaN DETECTED] Gradient NaN in: {name}")
                if torch.isnan(p).any():
                    print(f"[NaN DETECTED] Parameter NaN in: {name}")

        # Return scaled loss for logging (after backward)
        return total_loss.detach() / self.args.gradient_accumulation_steps