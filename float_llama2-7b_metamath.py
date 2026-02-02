import torch
from fire import Fire

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CHANGED: Import rotational_pissa_unified instead of peft
# from peft import PeftModel, LoraGAConfig, get_peft_model
# from peft.utils.lora_ga_utils import (
#     estimate_gradient,
#     LoraGAContext,
#     save_loraga_model_init,
#     save_loraga_model_final,
# )
from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    replace_linear_with_rotational_pissa,
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from accelerate import Accelerator
from utils import (
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from data import DATASET_MAP
import wandb
import os


def main(lora_alpha=128, lora_rank=None, sample_size=128, seed=42, resume_from_checkpoint=None, track_grad_norm=False, method="way0", total_cycles=4, epochs=1, use_butterfly=False, butterfly_sequential=False):
    accelerator = Accelerator()
    model_id = "google/gemma-7b"
    model_type = "CausalLM"
    model_dtype = "bf16"
    dataset_name = "meta_math_full"

    # Default rank logic for butterfly
    if lora_rank is None:
        if use_butterfly:
            # Gemma-7b hidden size is 3072
            # Ideally fetch via AutoConfig but hardcoding for known model_id is safe enough or fetch dynamically
            from transformers import AutoConfig
            config_obj = AutoConfig.from_pretrained(model_id)
            lora_rank = config_obj.hidden_size
            if accelerator.is_local_main_process:
                print(f"🦋 Butterfly mode: defaulting rank to d_model={lora_rank}")
        else:
            lora_rank = 128
            
    config = dict(
        model=model_id.replace("/", "_"),
        d=dataset_name,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
        method=method,
        butterfly=use_butterfly,
        seq=butterfly_sequential,
    )
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    if butterfly_sequential:
         wandb_name = "butterfly_seq_" + wandb_name
    
    # wandb_name+="way1"  # Removed hardcoded suffix, now part of config
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="online",
            group="test",
            # CHANGED: Update project name to reflect rotational PiSSA
            project="Gemma SOARA",
        )
    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )
    if accelerator.is_local_main_process:
        print(model)

    # CHANGED: Use RotationalPiSSAConfig instead of LoraGAConfig
    pissa_config = RotationalPiSSAConfig(
        r=config["r"],
        lora_alpha=config["a"],
        # CHANGED: Add rotational PiSSA specific parameters
        # CHANGED: Use method argument
        method=method,
        total_cycles=total_cycles,  # For way1
        use_butterfly=use_butterfly,
        butterfly_sequential=butterfly_sequential,
        orthogonality_reg_weight=0,   
        init_identity=True,
        freeze_singular_values=False,
        quantize_residual=False,
        quantize_base_components=False,
    )

    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()

    if accelerator.is_local_main_process:
        print(pissa_config)
        print("model:", model)
        # print(val_set)
    
    # CHANGED: Use replace_linear_with_rotational_pissa instead of get_peft_model
    # Note: PiSSA doesn't use gradient-guided initialization like LoRA-GA
    # The SVD decomposition inherently captures the weight structure
    # 
    # Each rank does SVD on its own GPU (accelerator.device) to avoid contention.
    # SVD is deterministic - both ranks get identical results from identical inputs.
    # DeepSpeed will sync parameters during Trainer initialization.
    print(f"[Rank {accelerator.process_index}] Starting SVD initialization on {accelerator.device}...")
    adapters = replace_linear_with_rotational_pissa(
        model=model,
        pissa_config=pissa_config,
        target_modules=find_all_linear_modules(model=model),
        adapter_name="default",
        freeze_base_model=True,
        device=accelerator.device,  # Each rank uses its own GPU
    )
    print(f"[Rank {accelerator.process_index}] SVD initialization complete.")

    save_dir = os.path.join("./snapshot", wandb_name)
    if accelerator.is_local_main_process:
        print(model)
        # CHANGED: Custom save function instead of save_loraga_model_init
        # Save model using PyTorch's state dict
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'pissa_config': pissa_config,
            'adapters': list(adapters.keys()),
        }, os.path.join(save_dir, "init_checkpoint.pt"))
    print("finish replace_linear_with_rotational_pissa=================================================")
    
    # Note: No explicit barrier needed here - DeepSpeed Trainer handles 
    # synchronization internally when it wraps the model for distributed training.
    # The low kernel version (5.4.0) makes NCCL barriers unreliable anyway.

    model = train_text_to_text_model(
        run_name=os.path.join("peft_test", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        num_train_epochs=epochs,
        per_device_batch_size=1,
        real_batch_size=128,
        bf16=(model_dtype == "bf16"),
        eval_epochs=0.25,
        early_stopping_patience=3,
        max_length=1024,
        logging_steps=1,
        use_loraplus=False,
        loraplus_lr_ratio=None,
        learning_rate=2e-4,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        # CHANGED: Pass pissa_config for orthogonality regularization
        pissa_config=pissa_config,
        # CHANGED: Pass resume_from_checkpoint for resuming training
        resume_from_checkpoint=resume_from_checkpoint,
        training_args=dict(
            lr_scheduler_type="cosine",
            # If track_grad_norm is False, we disable max_grad_norm (set to 0) to skip clipping and sync overhead
            max_grad_norm=1.0 if track_grad_norm else 0.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
            torch_compile=False,  # Disabled: inductor adds significant memory overhead
        ),
    )
    if accelerator.is_local_main_process:
        # CHANGED: Custom save function instead of save_loraga_model_final
        torch.save({
            'model_state_dict': model.state_dict(),
            'pissa_config': pissa_config,
            'adapters': list(adapters.keys()),
        }, os.path.join(save_dir, "final_checkpoint.pt"))


if __name__ == "__main__":
    Fire(main)
