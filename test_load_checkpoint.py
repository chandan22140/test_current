"""
Script to load and test a trained checkpoint.
"""
import torch
import os
from utils import initialize_text_to_text_model

def main():
    # Model configuration
    model_id = "google/gemma-3-1b-it"
    model_type = "CausalLM"
    model_dtype = "bf16"
    
    # Checkpoint path
    checkpoint_path = "/home/chandan/DL_Quantization/backup_131/DL_Quantization/PiSSA/test_current/results/peft_test/model=llama_d=meta_math_a=8_r=32_s=128_sd=31/31/checkpoint-781"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Check if this is a HuggingFace Trainer checkpoint or custom checkpoint
    pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    final_checkpoint_path = os.path.join(checkpoint_path, "final_checkpoint.pt")
    
    if os.path.exists(pytorch_model_path):
        print("Found HuggingFace Trainer checkpoint (pytorch_model.bin)")
        
        # Initialize fresh model
        print(f"Loading base model: {model_id}")
        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        
        # Load the trained state dict
        print("Loading trained weights...")
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("Successfully loaded checkpoint!")
        print(model)
        
    elif os.path.exists(final_checkpoint_path):
        print("Found custom PiSSA checkpoint (final_checkpoint.pt)")
        
        # Initialize fresh model
        print(f"Loading base model: {model_id}")
        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        
        # Load the custom checkpoint
        print("Loading trained weights...")
        checkpoint = torch.load(final_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"PiSSA config: {checkpoint.get('pissa_config')}")
        print(f"Adapters: {checkpoint.get('adapters')}")
        print("Successfully loaded checkpoint!")
        print(model)
    
    else:
        print(f"No recognized checkpoint format found in {checkpoint_path}")
        print("Available files:")
        for f in os.listdir(checkpoint_path):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
