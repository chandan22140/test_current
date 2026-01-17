"""
Script to load DeBERTa-v3-base and print its architecture,
then calculate LoRA trainable params at rank 8.
"""
import torch
from transformers import AutoModel, AutoConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)

def count_lora_params(in_features, out_features, rank):
    """Count LoRA params for a single linear layer: A(in×r) + B(r×out)"""
    return in_features * rank + rank * out_features

def main():
    model_name = "microsoft/deberta-v3-base"
    
    print(f"Loading {model_name}...")

    # config = AutoConfig.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    
    
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )    
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Intermediate size: {config.intermediate_size}")
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(model)
    
    print("\n" + "="*80)
    print("LINEAR LAYERS (potential LoRA targets)")
    print("="*80)
    
    linear_layers = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            shape = f"({module.in_features}, {module.out_features})"
            params = module.in_features * module.out_features
            if module.bias is not None:
                params += module.out_features
            linear_layers[name] = {
                'in': module.in_features,
                'out': module.out_features,
                'params': params
            }
            print(f"  {name}: {shape} = {params:,} params")
            total_params += params
    
    print(f"\nTotal linear layer params: {total_params:,}")
    
    # Calculate LoRA params at rank 8
    rank = 8
    print("\n" + "="*80)
    print(f"PISSA TRAINABLE PARAMS (rank={rank})")
    print("="*80)
    
    # Target modules (same as train_glue_deberta.py)
    target_modules = [
        "query_proj", "key_proj", "value_proj",  # Q, K, V
        "attention.output.dense",                 # O projection
        "intermediate.dense",                     # FFN up
        "output.dense",                           # FFN down
    ]
    exclude_modules = ["pooler", "classifier"]  # Don't apply PiSSA to these
    
    pissa_targets = []
    pissa_total = 0
    fully_trainable = []
    fully_trainable_total = 0
    
    for name, info in linear_layers.items():
        # Check if excluded (pooler, classifier - fully trainable, no PiSSA)
        if any(excl in name for excl in exclude_modules):
            fully_trainable.append((name, info['in'], info['out'], info['params']))
            fully_trainable_total += info['params']
            print(f"  [FULL] {name}: {info['params']:,} params (fully trainable)")
            continue
        
        # Check if target for PiSSA
        if any(target in name for target in target_modules):
            lora_params = count_lora_params(info['in'], info['out'], rank)
            pissa_targets.append((name, info['in'], info['out'], lora_params))
            pissa_total += lora_params
            print(f"  [PISSA] {name}: ({info['in']}×{rank}) + ({rank}×{info['out']}) = {lora_params:,}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    total_model_params = sum(p.numel() for p in model.parameters())
    total_trainable = pissa_total + fully_trainable_total
    print(f"Total model params: {total_model_params:,}")
    print(f"PiSSA adapter params (r={rank}): {pissa_total:,}")
    print(f"Fully trainable params (pooler+classifier): {fully_trainable_total:,}")
    print(f"Total trainable: {total_trainable:,}")
    print(f"Percentage: {total_trainable / total_model_params * 100:.2f}%")
    print(f"\nLayers targeted: {len(pissa_targets)} PiSSA + {len(fully_trainable)} fully trainable")

if __name__ == "__main__":
    main()
