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


def main(lora_alpha=128, lora_rank=128, sample_size=128, seed=42):
    accelerator = Accelerator()
    model_id = "google/gemma-7b"
    model_type = "CausalLM"
    model_dtype = "bf16"
    dataset_name = "meta_math_full"
    config = dict(
        model=model_id.replace("/", "_"),
        d=dataset_name,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
    )
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    # wandb_name+="newarmup"
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
        method="way0",  # Using direct optimization with regularization
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
    
    # CHANGED: Use replace_linear_with_rotational_pissa instead of get_peft_model
    # Note: PiSSA doesn't use gradient-guided initialization like LoRA-GA
    # The SVD decomposition inherently captures the weight structure
    adapters = replace_linear_with_rotational_pissa(
        model=model,
        pissa_config=pissa_config,
        target_modules=find_all_linear_modules(model=model),
        adapter_name="default",
        freeze_base_model=True,
    )

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

    model = train_text_to_text_model(
        run_name=os.path.join("peft_test", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        num_train_epochs=1,
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
        training_args=dict(
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
            torch_compile=True,
        ),
    )
    if accelerator.is_local_main_process:
        # CHANGED: Custom save function instead of save_loraga_model_final
        torch.save({
            'model_state_dict': model.state_dict(),
            'pissa_config': pissa_config,
            'adapters': list(adapters.keys()),
        }, os.path.join(save_dir, "final_checkpoint.pt"))
        
        # CHANGED: Load model differently for rotational PiSSA
        # Instead of PeftModel.from_pretrained, we load the state dict
        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        # checkpoint = torch.load(os.path.join(save_dir, "final_checkpoint.pt"), weights_only=False)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # print(model)


if __name__ == "__main__":
    Fire(main)
