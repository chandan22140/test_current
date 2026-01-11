#!/usr/bin/env python3
"""
Training script for LLM with Rotational PiSSA on Mathematical Reasoning
========================================================================

This script demonstrates training language models on math reasoning tasks (GSM8K, MetaMathQA)
using all 4 rotational methods with proper answer extraction and validation.

Mathematical Answer Format:
- Training uses MetaMathQA with boxed answers: \boxed{numerical_value}
- Evaluation on GSM8K with majority voting (3 samples per question)
- Implements zero prompt loss masking to focus on answer generation

Supported Models:
- llama-2-7b, llama-2-13b
- mistral-7b-v0.1
- phi-2, phi-3-mini
- microsoft/phi-2 - 2.7B params
- Custom models via HuggingFace model_name_or_path

Supported Datasets:
- gsm8k (8,792 train / 1,319 test - grade school math)
- metamath (395,000 samples - augmented math reasoning)
- math (12,500 train / 5,000 test - competition math)

Usage:
    # Train on MetaMathQA with Llama-2-7B
    python train_llm_rotational.py --method way0 --dataset metamath --model meta-llama/Llama-2-7b-hf --epochs 3
    
    # Evaluate on GSM8K with majority voting
    python train_llm_rotational.py --method way1 --dataset gsm8k --model meta-llama/Llama-2-7b-hf --eval-only --majority-voting
    
    # Train all methods with 3 random seeds for reproducibility
    python train_llm_rotational.py --method all --dataset gsm8k --seeds 42 43 44 --epochs 5
"""

# Fix MKL threading issue for vLLM (MUST be set before any numpy/torch imports)
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import gc
import re
import json
import glob
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np
from huggingface_hub import login


# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set transformers verbosity to warning to reduce noise
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"   

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# print("🎯 Configured to use GPU 4.")

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
TRAIN_DATA_SIZE = 10000   # Match LoRA-GA paper (100k train samples)
VAL_DATA_SIZE = 1000 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# HuggingFace imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Datasets
from datasets import load_dataset

# WandB for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  WandB not available - logging disabled")

# Import our rotational PiSSA modules
from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    RotationalLinearLayer,
    replace_linear_with_rotational_pissa,
    RotationalPiSSATrainer
)
from vram_profiler import (
    profile_model_memory, 
    print_memory_report, 
    profile_vram_during_training,
    FLOPSProfiler,
    estimate_training_flops
)


# ============================================================================
# MODEL MERGING UTILITIES
# ============================================================================

def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: str,
    verify_weights: bool = True,
    original_norms: Optional[Dict[str, float]] = None
) -> str:
    """
    Merge rotational PiSSA adapters into base model weights and save.
    
    This function merges trained adapters back into the base model by computing:
        W_merged = W_residual + W_principal
        W_principal = U @ R_U @ diag(S) @ R_V^T @ V^T * scaling
    
    Args:
        model: Model with rotational PiSSA adapters
        tokenizer: Tokenizer to save with model
        output_dir: Directory to save merged model
        verify_weights: Whether to verify weight norms match original (for untrained models)
        original_norms: Optional dict of pre-computed original weight norms {layer_name: norm}
                       If provided, these will be used for verification instead of computing
                       from adapter components. This is more accurate since it captures the
                       original weights before PiSSA decomposition.
        
    Returns:
        Path to saved merged model
    """
    from pathlib import Path
    
    print(f"\n💾 Merging adapters into base model...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics for verification
    merged_count = 0
    verification_results = []
    
    # Use provided original norms or compute from adapter components
    if verify_weights and original_norms is None:
        print("  Computing original weight norms from adapter components...")
        original_norms = {}
        for name, module in model.named_modules():
            if isinstance(module, RotationalLinearLayer):
                with torch.no_grad():
                    # Compute what the original weight should be
                    # W_original = W_residual + U @ diag(S) @ V * scaling
                    U = module.U
                    V = module.V
                    S = module.S
                    scaling = module.scaling
                    W_residual = module.base_layer.weight.data
                    
                    # Initial principal component (before rotation)
                    W_principal_init = U @ torch.diag(S) @ V * scaling
                    W_original = W_residual + W_principal_init
                    
                    original_norms[name] = torch.norm(W_original).item()
    elif verify_weights and original_norms is not None:
        print(f"  Using provided original weight norms ({len(original_norms)} layers)")

    # Find all RotationalLinearLayer adapters and merge them
    for name, module in list(model.named_modules()):
        if isinstance(module, RotationalLinearLayer):
            # Get rotation matrices
            R_U, R_V = module.get_rotation_matrices()
            
            with torch.no_grad():
                # Get components
                W_residual = module.base_layer.weight.data
                dtype = W_residual.dtype
                device = W_residual.device

                U = module.U.to(dtype=dtype, device=device)  # [out_features, r]
                V = module.V.to(dtype=dtype, device=device)  # [r, in_features] - V is ALREADY V^T
                S = module.S.to(dtype=dtype, device=device)  # [r]
                scaling = module.scaling
                
                # Compute W_principal: U @ R_U @ diag(S) @ R_V^T @ V
                # Note: V is stored as [r, in_features], which is V^T in the SVD formula
                # So the computation is: U @ R_U @ diag(S) @ R_V^T @ V (no extra transpose!)
                # IMPORTANT: Do NOT apply scaling here! Scaling is only for training dynamics.
                # The original weight decomposition is: W = W_residual + W_principal (no scaling)
                
                # Ensure rotation matrices are in correct dtype
                R_U = R_U.to(dtype=dtype, device=device)
                R_V = R_V.to(dtype=dtype, device=device)
                
                diag_S = torch.diag(S)  # [r, r]
                rotated_S = R_U @ diag_S @ R_V.T  # [r, r]
                
                # Final composition
                W_principal = U @ rotated_S @ V  # [out_features, in_features]
                
                # Get W_residual from base layer
                W_residual = module.base_layer.weight.data
                
                # Merge: W_merged = W_residual + W_principal
                W_principal = W_principal.to(dtype=W_residual.dtype, device=W_residual.device)
                W_merged = W_residual + W_principal
                
                # Verification: Check if merged norm matches original (for untrained models)
                if verify_weights and name in original_norms:
                    merged_norm = torch.norm(W_merged).item()
                    original_norm = original_norms[name]
                    norm_diff = abs(merged_norm - original_norm)
                    rel_diff = norm_diff / original_norm * 100
                    
                    verification_results.append({
                        "name": name,
                        "original_norm": original_norm,
                        "merged_norm": merged_norm,
                        "diff": norm_diff,
                        "rel_diff_pct": rel_diff,
                    })
                
                # Create standard Linear layer with merged weights
                merged_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.base_layer.bias is not None,
                    device=W_merged.device,
                    dtype=W_merged.dtype
                )
                merged_linear.weight.data = W_merged
                if module.base_layer.bias is not None:
                    merged_linear.bias.data = module.base_layer.bias.data
                
                # Replace adapter with merged linear layer
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, merged_linear)
                else:
                    setattr(model, name, merged_linear)
                
                merged_count += 1
                
                # Cleanup to free VRAM
                del W_principal, W_residual, W_merged, rotated_S, diag_S
                if merged_count % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
    
    print(f"  ✓ Merged {merged_count} adapter layers")
    
    # Print verification results
    if verify_weights and verification_results:
        print(f"\n  📊 Weight Verification (untrained model - should be identical):")
        max_diff = max(r["rel_diff_pct"] for r in verification_results)
        avg_diff = sum(r["rel_diff_pct"] for r in verification_results) / len(verification_results)
        
        print(f"     Average relative difference: {avg_diff:.6f}%")
        print(f"     Max relative difference: {max_diff:.6f}%")
        
        if max_diff > 1.0:  # More than 1% difference (beyond numerical precision)
            print(f"     ⚠️  WARNING: Weights differ by more than 1% - possible training occurred")
            print(f"     Top 5 layers with largest differences:")
            top_5 = sorted(verification_results, key=lambda x: x["rel_diff_pct"], reverse=True)[:5]
            for r in top_5:
                print(f"       {r['name']}: {r['rel_diff_pct']:.6f}%")
        else:
            print(f"     ✅ Weights match original within numerical precision (<1%)")
            if max_diff > 0.1:
                print(f"     Note: Small differences ({max_diff:.6f}%) are due to SVD floating-point precision")
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save merged model in HuggingFace format
    print(f"\n  Saving merged model to {output_path}...")
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,  # Use safetensors
        max_shard_size="5GB"  # Save as single file if < 5GB, otherwise shard
    )
    tokenizer.save_pretrained(str(output_path))
    
    # Remove stale index file if it exists (from original model)
    # This prevents mismatches between index and actual weight files
    index_file = output_path / "model.safetensors.index.json"
    if index_file.exists():
        # Check if we actually have sharded files
        shard_files = glob.glob(str(output_path / "model-*-of-*.safetensors"))
        if not shard_files:
            # We have a single model.safetensors but index references shards - remove index
            print(f"  Removing stale shard index file (model saved as single file)")
            index_file.unlink()
    
    print(f"  ✓ Saved merged model to {output_path}")
    return str(output_path)


# ============================================================================
# MATHEMATICAL ANSWER EXTRACTION
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{} notation.
    
    Args:
        text: Generated text containing boxed answer
        
    Returns:
        Extracted answer string or None if not found
        
    Examples:
        "\boxed{42}" -> "42"
        "\boxed{3.14}" -> "3.14"
        "\boxed{\frac{1}{2}}" -> "0.5"
    """
    # Try to find \boxed{...}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    
    if matches:
        answer = matches[-1].strip()  # Take last boxed answer
        
        # Convert LaTeX fractions to decimals
        frac_pattern = r'\\frac\{(\d+)\}\{(\d+)\}'
        frac_match = re.search(frac_pattern, answer)
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            return str(numerator / denominator)
        
        # Convert dfrac (display fraction) to decimals
        dfrac_pattern = r'\\dfrac\{(\d+)\}\{(\d+)\}'
        dfrac_match = re.search(dfrac_pattern, answer)
        if dfrac_match:
            numerator = float(dfrac_match.group(1))
            denominator = float(dfrac_match.group(2))
            return str(numerator / denominator)
        
        # Remove LaTeX formatting
        answer = answer.replace('$', '').replace(',', '').strip()
        
        return answer
    
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract answer from GSM8K format: #### {number}
    
    Args:
        text: Generated text containing #### answer
        
    Returns:
        Extracted numerical answer string or None if not found
        
    Examples:
        "Step 1...\\n#### 42" -> "42"
        "The answer is 100.\\n#### 100" -> "100"
    """
    # GSM8K format: #### followed by number
    pattern = r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, text)
    
    if match:
        # Remove commas from number
        answer = match.group(1).replace(',', '').strip()
        return answer
    
    return None


def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract final numerical answer from text using multiple strategies.
    
    Fallback extraction when \boxed{} is not present:
    1. Look for "Final Answer", "Therefore", "The answer is"
    2. Extract last numerical value from last sentence
    3. Handle scientific notation, percentages, currencies
    
    Args:
        text: Generated text
        
    Returns:
        Numerical answer as float or None if not found
    """
    # Strategy 1: Look for explicit answer markers
    answer_markers = [
        r'Final Answer[:\s]+([+-]?\d+\.?\d*)',
        r'Therefore[,\s]+(?:the answer is\s+)?([+-]?\d+\.?\d*)',
        r'The answer is[:\s]+([+-]?\d+\.?\d*)',
        r'Answer[:\s]+([+-]?\d+\.?\d*)',
    ]
    
    for pattern in answer_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Strategy 2: Extract from boxed notation first
    boxed = extract_boxed_answer(text)
    if boxed:
        try:
            # Clean and convert
            cleaned = re.sub(r'[^\d\.\-\+]', '', boxed)
            return float(cleaned)
        except ValueError:
            pass
    
    # Strategy 3: Extract last number from last sentence
    sentences = text.split('.')
    for sentence in reversed(sentences):
        # Find all numbers (including decimals, negatives)
        numbers = re.findall(r'[+-]?\d+\.?\d*', sentence)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                continue
    
    return None


def normalize_answer(answer: Union[str, float, int]) -> str:
    """
    Normalize answer to standard format for comparison.
    
    Args:
        answer: Raw answer (string, float, or int)
        
    Returns:
        Normalized string representation
    """
    if answer is None:
        return "INVALID"
    
    # Convert to string
    if isinstance(answer, (int, float)):
        answer_str = str(answer)
    else:
        answer_str = str(answer)
    
    # Remove whitespace and common formatting
    answer_str = answer_str.strip().lower()
    answer_str = answer_str.replace(',', '').replace('$', '').replace('%', '')
    
    # Try to convert to float for numerical comparison
    try:
        num = float(answer_str)
        # Round to 2 decimal places for comparison
        return f"{num:.2f}"
    except ValueError:
        print(answer, ":Non-numerical answer - return as-is:", answer_str)
        return answer_str


def majority_vote(answers: List[str]) -> str:
    """
    Select most frequent answer from multiple samples.
    
    Args:
        answers: List of normalized answers from multiple generations
        
    Returns:
        Most frequent answer
    """
    if not answers:
        return "INVALID"
    
    # Filter out invalid answers
    valid_answers = [a for a in answers if a != "INVALID"]
    
    if not valid_answers:
        return "INVALID"
    
    # Count occurrences
    counter = Counter(valid_answers)
    most_common = counter.most_common(1)[0][0]
    
    return most_common


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MathTrainingConfig:
    """Configuration for LLM training on math reasoning tasks."""
    
    # Model configuration
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"  # Smaller 2.7B model for faster testing (was: meta-llama/Llama-2-7b-hf)
    use_flash_attention: bool = False  # Requires flash-attn package
    load_in_8bit: bool = False  # Use 8-bit quantization for base model
    
    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 32  # Effective batch = batch_size * grad_accum = 32
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0  # Paper uses weight decay of 0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True  # Enable gradient checkpointing (LoRA-GA uses this)
    
    # Sequence length
    max_seq_length: int = 1024  # Maximum context length (paper uses T=1024)
    max_answer_length: Optional[int] = None  # No limit on answer generation length
    
    # Rotational PiSSA configuration
    method: str = "way0"  # way0, way1, way2, way3, or 'all'
    pissa_rank: int = 16  # Paper uses rank r=16
    pissa_alpha: float = 16.0  # Paper uses alpha=16
    lora_dropout: float = 0.0
    orthogonality_weight: float = 1e-3
    regularization_type: str = "frobenius"
    steps_per_phase: int = 0   # For way1
    total_cycles: int = 3      # For way1
    low_rank_r: int = 4        # For way2/way3
    quantize_residual: bool = False
    quantize_base_components: bool = False
    
    # Target modules for adaptation (for LLMs)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",    # Query projection in attention
        "k_proj",    # Key projection in attention
        "v_proj",    # Value projection in attention
        "o_proj",    # Output projection in attention
        "gate_proj", # Gate projection in FFN (Llama)
        "up_proj",   # Up projection in FFN
        "down_proj", # Down projection in FFN
    ])
    
    # Data configuration
    dataset: str = "gsm8k"  # gsm8k, metamath, math
    data_path: str = "./data"
    preprocessing_num_workers: int = 4
    
    # Answer extraction configuration
    zero_prompt_loss: bool = True  # Mask loss for input prompt
    filter_disclaimers: bool = True  # Filter training samples with disclaimers #practically doesnt affect gsm8k dataset
    evaluation_strategy: str = "single"  # "single" (greedy, fast) or "voting" (majority vote, robust)
    num_eval_samples: int = 3  # Number of samples per question for voting (only used if evaluation_strategy="voting")
    eval_temperature: float = 0.8  # Temperature for sampling during voting
    eval_top_p: float = 0.95  # Top-p for sampling during voting
    use_vllm: bool = True  # Use vLLM for faster batched inference
    eval_batch_size: int = 4  # Batch size for vLLM evaluation
    
    # Logging configuration
    use_wandb: bool = True
    project_name: str = "rotational-pissa-llm-math"
    experiment_name: Optional[str] = None
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Device configuration
    device: str = "cuda"
    
    # Freezing strategy
    freeze_backbone: bool = True
    train_embeddings: bool = False
    train_lm_head: bool = False
    
    # Reproducibility
    seed: int = 42
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ============================================================================
# DATASET
# ============================================================================

class MathReasoningDataset(Dataset):
    """Dataset for mathematical reasoning tasks."""
    
    def __init__(
        self,
        config: MathTrainingConfig,
        tokenizer: AutoTokenizer,
        split: str = "train",
    ):
        self.train_config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Filter if training
        if split == "train" and config.filter_disclaimers:
            self._filter_disclaimers()
        
        print(f"✓ Loaded {len(self.data)} {split} examples from {config.dataset}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load and preprocess dataset."""
        dataset_name = self.train_config.dataset.lower()
        
        if dataset_name == "gsm8k":
            # Load GSM8K from HuggingFace
            if self.split == "full":
                # Load both train and test splits and combine them
                train_dataset = load_dataset("gsm8k", "main", split="train")
                test_dataset = load_dataset("gsm8k", "main", split="test")
                
                # Combine datasets
                from datasets import concatenate_datasets
                dataset = concatenate_datasets([train_dataset, test_dataset])
            else:
                split_name = "train" if self.split == "train" else "test"
                dataset = load_dataset("gsm8k", "main", split=split_name)
            
            # Convert to our format
            data = []
            for item in dataset:
                question = item["question"]
                answer = item["answer"]
                
                # Extract numerical answer from GSM8K format
                # GSM8K answers are like: "Step 1...\n#### 42"
                if "####" in answer:
                    numerical_answer = answer.split("####")[-1].strip()  
                    data.append({
                        "question": question,
                        "answer": answer,
                        "numerical_answer": numerical_answer,
                    })
                else:
                    print("⚠️ invalid question in gsm8k without ####:", question, answer)
            
            return data
        
        elif dataset_name == "metamath":
            # Load MetaMathQA from HuggingFace (following LoRA-GA paper: GSM category only)
            # Load once and cache both train/eval splits
            if not hasattr(self.__class__, '_metamath_cache'):
                print(f"  Loading MetaMathQA dataset (filtering GSM category)...")
                dataset = load_dataset("meta-math/MetaMathQA", split="train")
                dataset = dataset.shuffle(seed=42)  # Shuffle with fixed seed for reproducibility
                
                # Filter for GSM category only and collect samples (matching LoRA-GA paper)
                # Train: 100k samples, Eval: 10k samples
                train_samples = []
                eval_samples = []
                
                from tqdm import tqdm
                print(f"  Filtering GSM category samples (using consistent prompt format)...")
                for item in tqdm(dataset, desc="Processing MetaMathQA"):
                    # Only use GSM category (matching LoRA-GA paper)
                    if "GSM" not in item.get("type", ""):
                        continue
                    
                    question = item["query"]
                    answer = item["response"]
                    
                    # Use consistent prompt format (matching _format_prompt)
                    prompt = self._format_prompt(question)
                    combined_text = prompt + " " + answer + self.tokenizer.eos_token
                    
                    # Filter by token length (max_seq_length from config)
                    tokens = self.tokenizer(combined_text, add_special_tokens=True)["input_ids"]
                    if len(tokens) > self.train_config.max_seq_length:
                        continue
                    
                    # Extract numerical answer - try GSM8K format first, then boxed notation
                    numerical_answer = extract_gsm8k_answer(answer) or extract_boxed_answer(answer) or ""
                    
                    # Skip if no numerical answer found (quality control)
                    if not numerical_answer:
                        continue

                    if "####" not in answer:
                        answer = answer + f"\n#### {numerical_answer}"

                    sample = {
                        "question": question,
                        "answer": answer,
                        "numerical_answer": numerical_answer,
                    }
                    
                    # Split: first 100k for train, next 10k for eval (matching LoRA-GA)
                    if len(train_samples) < TRAIN_DATA_SIZE:
                        train_samples.append(sample)
                    elif len(eval_samples) < VAL_DATA_SIZE:
                        eval_samples.append(sample)
                    
                    # Stop once we have enough samples
                    if len(train_samples) >= TRAIN_DATA_SIZE and len(eval_samples) >= VAL_DATA_SIZE:
                        break
                
                # Cache the splits
                self.__class__._metamath_cache = {
                    "train": train_samples,
                    "test": eval_samples,
                }
                print(f"  ✓ Collected {len(train_samples)} train + {len(eval_samples)} eval GSM category samples")
            
            # Return cached data for requested split
            data = self.__class__._metamath_cache[self.split]
            return data
        
        elif dataset_name == "math":
            # Load MATH dataset from HuggingFace
            split_name = "train" if self.split == "train" else "test"
            dataset = load_dataset("hendrycks/competition_math", split=split_name)
            
            data = []
            for item in dataset:
                question = item["problem"]
                answer = item["solution"]
                
                # Extract numerical answer
                numerical_answer = extract_boxed_answer(answer) or ""
                
                data.append({
                    "question": question,
                    "answer": answer,
                    "numerical_answer": numerical_answer,
                })
            
            return data
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _filter_disclaimers(self):
        """Filter out training samples with disclaimers."""
        original_len = len(self.data)
        
        disclaimer_patterns = [
            r'^As an AI',
            r'^Sorry',
            r'^I apologize',
            r'^I cannot',
            r'^I\'m sorry',
        ]
        
        filtered_data = []
        for item in self.data:
            answer = item["answer"]
            
            # Check if answer starts with disclaimer
            has_disclaimer = any(
                re.match(pattern, answer, re.IGNORECASE)
                for pattern in disclaimer_patterns
            )
            
            if not has_disclaimer:
                filtered_data.append(item)
        
        self.data = filtered_data
        
        filtered_count = original_len - len(self.data)
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} samples with disclaimers ({filtered_count/original_len*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Format prompt (following instruction-tuning format)
        prompt = self._format_prompt(item["question"])
        answer = item["answer"]
        
        # Tokenize
        # We'll handle loss masking in collate_fn
        return {
            "prompt": prompt,
            "answer": answer,
            "numerical_answer": item["numerical_answer"],
            "question": item["question"],
        }
    
    
    def _format_prompt(self, question: str) -> str:
        """Format question as instruction prompt with GSM8K answer format specification."""
        # Use simple format matching LoRA-GA baseline
        prompt = f"Q: {question}\nA: "
        return prompt


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, config: MathTrainingConfig) -> Dict[str, torch.Tensor]:
    """
    Collate batch with proper padding and loss masking.
    
    Implements zero prompt loss masking (following LoRA-GA approach):
    - Tokenize combined prompt + answer + EOS as one sequence (context-aware tokenization)
    - Calculate prompt length separately for masking
    - Loss for input prompt is set to -100 (ignored)
    - Loss for answer generation is computed normally
    """
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # Combine prompt + answer + EOS token (following LoRA-GA)
    # This ensures proper context-aware tokenization
    combined_texts = [
        prompt + " " + answer + tokenizer.eos_token
        for prompt, answer in zip(prompts, answers)
    ]
    # print("combined_texts[0]:", combined_texts[0])
    
    # Tokenize combined text in one pass (more accurate than separate tokenization)
    combined_encodings = tokenizer(
        combined_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=config.max_seq_length,
        padding=False,
        return_tensors=None,
    )
    
    # print("combined_encodings[0]:" , combined_encodings[0])
    # print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    # Calculate prompt lengths for masking (tokenize prompts to get their lengths)
    # Note: We need to tokenize prompts separately ONLY to know where to mask
    prompt_encodings = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    
    # Build batch tensors
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i in range(len(batch)):
        input_ids = combined_encodings["input_ids"][i]
        attention_mask = combined_encodings["attention_mask"][i]
        prompt_length = len(prompt_encodings["input_ids"][i])
        
        # Create labels with prompt masking
        if config.zero_prompt_loss:
            # Mask prompt tokens (set to -100)
            labels = [-100] * prompt_length + input_ids[prompt_length:]
        else:
            # Compute loss on everything
            labels = input_ids.copy()
        
        # Ensure labels length matches input
        if len(labels)!=len(input_ids):
            labels = labels[:len(input_ids)]
            print("⚠️  len(labels)!=len(input_ids)")
            
        
        # Add to lists
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    
    # Pad sequences
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class MathDataCollator:
    """Custom data collator for math reasoning tasks."""
    
    def __init__(self, tokenizer: AutoTokenizer, config: MathTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return collate_fn(batch, self.tokenizer, self.config)



# ============================================================================
# TRAINER CALLBACKS
# ============================================================================

class RotationalPiSSACallback(TrainerCallback):
    """Callback for handling Rotational PiSSA specific logic."""
    
    def __init__(self, rotational_trainer: Optional[RotationalPiSSATrainer], config: MathTrainingConfig):
        self.rotational_trainer = rotational_trainer
        self.config = config
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.rotational_trainer and self.config.method == "way1":
            if self.rotational_trainer.should_step_phase(state.global_step):
                print(f"  🔄 Stepping to next Givens phase at step {state.global_step}")
                self.rotational_trainer.step_phase()
        return control


# ============================================================================
# TRAINER
# ============================================================================

class LLMMathTrainer:
    """Trainer for LLM on math reasoning tasks with Rotational PiSSA."""
    
    def __init__(self, config: MathTrainingConfig):
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        self.device = self._setup_device()
        print(f"✓ Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Setup datasets
        print(f"\n📚 Loading Datasets...")
        print(f"  Training dataset: {config.dataset}")
        self.train_dataset = MathReasoningDataset(config, self.tokenizer, split="train")
        
        # For intermediate evaluation during training: use validation split from same dataset
        if config.dataset == "metamath":
            # Use 10k validation samples from MetaMathQA (GSM category)
            intermediate_eval_dataset = MathReasoningDataset(config, self.tokenizer, split="test")
            print(f"  Intermediate eval: {len(intermediate_eval_dataset)} samples from MetaMathQA validation")
        else:
            # For GSM8K/MATH, use validation split
            intermediate_eval_dataset = MathReasoningDataset(config, self.tokenizer, split="test")
        
        # Sample subset for faster intermediate evals (saves VRAM)
        if len(intermediate_eval_dataset) > VAL_DATA_SIZE:
            import random
            eval_indices = random.sample(range(len(intermediate_eval_dataset)), VAL_DATA_SIZE)
            eval_indices.sort()
            self.eval_dataset = torch.utils.data.Subset(intermediate_eval_dataset, eval_indices)
            print(f"  Using {len(self.eval_dataset)} samples for intermediate evaluation steps")
        else:
            self.eval_dataset = intermediate_eval_dataset
        
        # For FINAL evaluation: always use GSM8K test set (1,319 samples) - matching LoRA-GA paper
        print(f"\n🎯 Final Evaluation Dataset: GSM8K test set")
        original_dataset = config.dataset
        config.dataset = "gsm8k"  # Temporarily switch to GSM8K
        self.final_eval_dataset = MathReasoningDataset(config, self.tokenizer, split="full")
        config.dataset = original_dataset  # Restore original dataset
        print(f"  Final evaluation will use {len(self.final_eval_dataset)} samples from GSM8K test set")
        
        # Initialize WandB (skip if already initialized by sweep)
        if config.use_wandb and HAS_WANDB:
            if wandb.run is None:
                self._init_wandb()
            else:
                print(f"✓ Using existing WandB run: {wandb.run.name}")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer with proper configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,  # Allow custom tokenizer code from model repo (required for some models)
            use_fast=True,           # Use fast Rust-based tokenizer (faster than Python version)
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("⚠️ tokenizer.pad_token is already:", tokenizer.pad_token)
        
        # Set padding side to right for training (will switch to left during generation)
        tokenizer.padding_side = "right"
        
        print(f"✓ Loaded tokenizer: {self.config.model_name_or_path}")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f" ⚠️  Initial Padding side: {tokenizer.padding_side}")
        
        return tokenizer
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        exp_name = self.config.experiment_name or f"{self.config.dataset}_{self.config.method}_seed{self.config.seed}"
        
        wandb.init(
            project=self.config.project_name,
            name=exp_name,
            config=self.config.__dict__,
        )
    
    def create_model(self, method: str) -> Tuple[nn.Module, Optional[RotationalPiSSATrainer]]:
        """Create model with rotational PiSSA adapters."""
        print(f"\n{'='*70}")
        print(f"Creating model with method: {method}")
        print(f"{'='*70}")
        
        # Load base model
        print(f"Loading base model: {self.config.model_name_or_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,  # Use BF16 for better memory efficiency and stability
            load_in_8bit=self.config.load_in_8bit,
            device_map=None,  # Manual device placement
            trust_remote_code=True,
        )
        
        # Capture original norms before any modification (for later verification)
        print("  Capturing original weight norms...")
        self.original_norms = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.original_norms[name] = torch.norm(module.weight).item()
        
        
        # Move to device if not using 8-bit (8-bit handles device placement automatically)
        if not self.config.load_in_8bit:
            model = model.to(self.device)
        
        # Print model architecture
        print("\n📐 Model Architecture:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Configure rotational PiSSA
        pissa_config = RotationalPiSSAConfig(
            r=self.config.pissa_rank,
            lora_alpha=self.config.pissa_alpha,
            lora_dropout=self.config.lora_dropout,
            method=method,
            orthogonality_reg_weight=self.config.orthogonality_weight,
            regularization_type=self.config.regularization_type,
            steps_per_phase=self.config.steps_per_phase,
            total_cycles=self.config.total_cycles,
            low_rank_r=self.config.low_rank_r,
            quantize_residual=self.config.quantize_residual,
            quantize_base_components=self.config.quantize_base_components,
        )
        
        # Replace linear layers with rotational PiSSA
        print(f"\n🔄 Applying Rotational PiSSA ({method})...")
        
        # Clear VRAM before replacement
        gc.collect()
        torch.cuda.empty_cache()
        
        adapters = replace_linear_with_rotational_pissa(
            model,
            pissa_config,
            target_modules=self.config.target_modules,
            freeze_base_model=self.config.freeze_backbone,
        )
        
        # Clear VRAM after replacement
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"  ✓ Created {len(adapters)} rotational adapters")
        
        # Optionally unfreeze embeddings and LM head
        if self.config.train_embeddings:
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = True
            print(f"  ✓ Embeddings unfrozen")
        else:
            print(f"  ✓ Embeddings frozen")
        
        if self.config.train_lm_head:
            for param in model.get_output_embeddings().parameters():
                param.requires_grad = True
            print(f"  ✓ LM head unfrozen")
        else:
            print(f"  ✓ LM head frozen")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_pct = trainable_params / total_params * 100
        
        print(f"\n📊 Parameter Counts:")
        print(f"  Total:     {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Frozen:    {total_params-trainable_params:,} ({(total_params-trainable_params)/1e6:.2f}M)")
        print(f"  Trainable: {trainable_pct:.2f}%")
        
        # Create rotational trainer for all methods (needed for loss calculation in Way 0)
        # and phase stepping in Way 1
        rotational_trainer = RotationalPiSSATrainer(model, pissa_config)
        if method == "way1":
            print(f"  ✓ Created RotationalPiSSATrainer for sequential Givens")
        elif method == "way0":
            print(f"  ✓ Created RotationalPiSSATrainer for orthogonality regularization")
        
        # Profile VRAM
        print(f"\n🔍 VRAM Profiling:")
        vram_results = profile_model_memory(model, device=str(self.device))
        print_memory_report(vram_results)
        
        return model, rotational_trainer
    
    def train_single_method(self, method: str) -> Dict:
        """Train with a single rotational method using HuggingFace Trainer."""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING: {method.upper()}")
        print(f"{'='*70}\n")
        
        # Create model
        model, rotational_trainer = self.create_model(method)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"{method}_seed{self.config.seed}"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type="cosine",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps" if self.config.eval_steps > 0 else "no",
            save_strategy="steps" if self.config.save_checkpoints else "no",
            save_total_limit=3,
            load_best_model_at_end=True if self.config.save_checkpoints else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,  # CRITICAL: Only compute loss, don't store logits (saves ~12GB VRAM)
            bf16=True,
            dataloader_num_workers=self.config.preprocessing_num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb and HAS_WANDB else "none",
            run_name=f"{self.config.dataset}_{method}_seed{self.config.seed}",
            seed=self.config.seed,
        )
        
        
        # Create data collator
        data_collator = MathDataCollator(self.tokenizer, self.config)
        
        # Create custom trainer to handle orthogonality loss
        class CustomTrainer(Trainer):
            def __init__(self, rotational_trainer=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rotational_trainer = rotational_trainer
                
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Standard loss
                if num_items_in_batch is not None:
                     loss_or_outputs = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
                else:
                     loss_or_outputs = super().compute_loss(model, inputs, return_outputs)
                
                if return_outputs:
                    loss, outputs = loss_or_outputs
                else:
                    loss = loss_or_outputs
                    outputs = None
                
                # Add orthogonality loss if applicable
                if self.rotational_trainer:
                    ortho_loss = self.rotational_trainer.get_orthogonality_loss()
                    # Add to main loss
                    loss = loss + ortho_loss.to(loss.device)
                    
                return (loss, outputs) if return_outputs else loss

        # Create trainer
        trainer = CustomTrainer(
            rotational_trainer=rotational_trainer,
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[
                RotationalPiSSACallback(rotational_trainer, self.config),
            ],
        )
        
        print(f"📋 Training Configuration:")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.effective_batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Warmup ratio: {self.config.warmup_ratio}")
        
        # Train
        train_result = trainer.train()
        
        # Prepare for evaluation and saving
        # We extract state to CPU first so we can free GPU memory if needed for vLLM
        trainable_state = {
            name: param.cpu() for name, param in model.named_parameters() if param.requires_grad
        }
        
        best_eval_loss = 0.0
        if hasattr(trainer.state, 'best_metric'):
            best_eval_loss = trainer.state.best_metric
        else:
            print("⚠️  trainer.state has no attr best_metric")

        # Final evaluation logic
        final_accuracy = 0.0
        
        if self.config.evaluation_strategy == "voting":
            print(f"\n🗳️  Final Evaluation with Majority Voting ({self.config.num_eval_samples} samples)...")
            final_accuracy = self._evaluate_with_voting(model)
            print(f"  Final Accuracy (majority voting): {final_accuracy:.2f}%")
            
        elif self.config.evaluation_strategy == "single":
            print(f"\n✅ Final Evaluation with Single-Pass Greedy Decoding...")
            # print("self.config.use_vllm:", self.config.use_vllm)
            if self.config.use_vllm:
                print(f"  Using vLLM for fast batched inference (batch_size={self.config.eval_batch_size})...")
                
                # 1. Save merged model first (needed for vLLM loading)
                merged_model_path = self._save_merged_model(model, method)
                
                # 2. CRITICAL: Free VRAM before starting vLLM
                # vLLM needs significant memory for KV cache and model weights
                # We must delete the current model and trainer references
                print("  🗑️  Clearing GPU memory for vLLM...")
                del model, trainer, rotational_trainer
                gc.collect()
                torch.cuda.empty_cache()
                
                # 3. Run vLLM evaluation (loads model from disk into fresh memory)
                final_accuracy = self._evaluate_with_vllm(merged_model_path)
            else:
                print("⚠️  Unhandled case")
                
            print(f"  Final Accuracy (single-pass): {final_accuracy:.2f}%")
        else:
            print("⚠️  invalid self.config.evaluation_strategy")

        
        # Save final model (using pre-extracted state)
        if self.config.save_checkpoints:
            final_model_dir = self.output_dir / f"{method}_seed{self.config.seed}_final"
            final_model_dir.mkdir(parents=True, exist_ok=True)
            
            save_dict = {
                "method": method,
                "final_accuracy": final_accuracy,
                "best_eval_loss": best_eval_loss,
                "trainable_state_dict": trainable_state,
                "config": self.config.__dict__,
                "train_result": {
                    "global_step": train_result.global_step,
                    "training_loss": train_result.training_loss,
                },
            }
            
            torch.save(save_dict, final_model_dir / "model.pt")
            print(f"✓ Saved final model to {final_model_dir}/model.pt")
            print(f"  Final Accuracy: {final_accuracy:.2f}%")
            print(f"  Trainable parameters saved: {len(trainable_state)}")
        
        return {
            "method": method,
            "best_eval_loss": best_eval_loss,
            "final_accuracy": final_accuracy,
            "seed": self.config.seed,
            "train_result": train_result,

        }
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        print("⚠️ _compute_metrics called ..")
        """
        Compute evaluation metrics.
        
        Note: With prediction_loss_only=True, we don't receive logits here.
        The Trainer automatically computes eval_loss during evaluation.
        For actual accuracy (which requires generation), use _evaluate_single_pass() or _evaluate_with_voting().
        
        Args:
            eval_pred: EvalPrediction object (empty when prediction_loss_only=True)
            
        Returns:
            Empty dictionary (loss is computed automatically by Trainer)
        """
        # When prediction_loss_only=True, Trainer computes loss automatically
        # and we don't need to do anything here
        return {}
    
    def _evaluate_single_pass(self, model: nn.Module) -> float:
        """
        Evaluate with single-pass greedy decoding (LoRA-GA style).
        
        Fast evaluation matching the LoRA-GA paper's approach:
        - Single greedy generation per question
        - No sampling or majority voting
        - Uses GSM8K #### format extraction
        """
        model.eval()
        
        correct = 0
        total = 0
        
        # Switch to left padding for generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        print(f"  ⚠️  Switched to left padding for generation")
        
        # Use full eval dataset for final evaluation
        eval_dataset = self.final_eval_dataset
        
        # Create temporary dataloader for evaluation
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.preprocessing_num_workers,
            collate_fn=MathDataCollator(self.tokenizer, self.config),
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Get original data for this batch
                batch_start = batch_idx * self.config.batch_size
                batch_end = min(batch_start + self.config.batch_size, len(eval_dataset))
                batch_data = [eval_dataset[i] for i in range(batch_start, batch_end)]
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                numerical_answers = [item["numerical_answer"] for item in batch_data]
                
                # Single greedy generation
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_answer_length or 1024,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids[:, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Extract and compare answers
                for generated_text, true_answer in zip(generated_texts, numerical_answers):
                    # Try GSM8K format first (#### number)
                    pred_answer = extract_gsm8k_answer(generated_text)
                    
                    # Fallback to boxed format for MetaMathQA
                    if pred_answer is None:
                        pred_answer = extract_numerical_answer(generated_text)
                    
                    pred_norm = normalize_answer(pred_answer)
                    true_norm = normalize_answer(true_answer)
                    
                    if pred_norm == true_norm:
                        correct += 1
                    
                    total += 1
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        
        accuracy = correct / total * 100 if total > 0 else 0.0
        return accuracy
    
    def _save_merged_model(self, model: nn.Module, method: str) -> str:
        """
        Save merged model (wrapper around standalone save_merged_model function).
        
        Args:
            model: Model with rotational PiSSA adapters
            method: Training method name (for output directory naming)
            
        Returns:
            Path to saved merged model
        """
        merged_model_dir = self.output_dir / f"{method}_seed{self.config.seed}_merged"
        return save_merged_model(
            model=model,
            tokenizer=self.tokenizer,
            output_dir=str(merged_model_dir),
            verify_weights=True,  # Don't verify for trained models
            original_norms=self.original_norms
        )
    
    def _evaluate_with_vllm(self, model_path: str) -> float:
        """
        Evaluate using vLLM for fast batched inference (matching LoRA-GA paper).
        
        Args:
            model_path: Path to merged model checkpoint
            
        Returns:
            Accuracy percentage
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("⚠️  vLLM not installed. Falling back to standard evaluation.")
            print("   Install with: pip install vllm")
            # Fallback to standard evaluation
            return self._evaluate_single_pass(None)
        
        print(f"\n🚀 Loading model with vLLM...")
        
        # Calculate safe GPU memory utilization based on available free memory
        # Free: 25.0 GiB, Total: 39.39 GiB -> Use ~60% of total to be safe
        llm = LLM(
            model_path,
            dtype="bfloat16",
            seed=self.config.seed,
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs to save memory
            gpu_memory_utilization=0.60,  # Reduced to 60% (~23.6 GiB) to fit in 25 GiB free
            max_model_len=2048,  # Reduce from default 4096 to save KV cache memory
        )
        
        
        sampling_params = SamplingParams(
            top_p=self.config.eval_top_p, 
            temperature=self.config.eval_temperature,
            max_tokens=1024
        )
        
        correct = 0
        total = 0
        
        # Get prompts from eval dataset
        eval_dataset = self.final_eval_dataset
        prompts = []
        true_answers = []
        
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            prompt = item["prompt"]
            prompts.append(prompt)
            true_answers.append(item["numerical_answer"])
        
        print(f"  Evaluating on {len(prompts)} samples with batch_size={self.config.eval_batch_size}...")
        print(f"\n  📝 First prompt example:")
        print(f"  {prompts[0]}...")
        print(f"  True answer: {true_answers[0]}")
        
        # Batch inference with vLLM
        from tqdm import tqdm
        
        # Debug: print first few examples
        debug_count = 0
        max_debug = 60
        
        for idx in tqdm(range(0, len(prompts), self.config.eval_batch_size), desc="vLLM Inference"):
            batch_prompts = prompts[idx:idx + self.config.eval_batch_size]
            batch_answers = true_answers[idx:idx + self.config.eval_batch_size]
            
            # Generate with vLLM
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
            
            
            
            # Extract and compare answers
            count = 0
            for true_answer, output in zip(batch_answers, outputs):
                
                generated_text = output.outputs[0].text
                
                # Try GSM8K format first (#### number)
                pred_answer = extract_gsm8k_answer(generated_text)
                
                # Fallback to numerical extraction
                if pred_answer is None:
                    pred_answer = extract_numerical_answer(generated_text)
                
                pred_norm = normalize_answer(pred_answer)
                true_norm = normalize_answer(true_answer)
                
                # Debug: Print first few examples
                if debug_count < max_debug:
                    print(f"\n{'='*70}")
                    print(f"Example {debug_count + 1}:")
                    print(f"batch_prompts[count]: {batch_prompts[count]}...")
                    print(f"Generated: {generated_text}...")
                    print(f"Extracted pred: {pred_answer}")
                    print(f"Normalized pred: {pred_norm}")
                    print(f"True answer: {true_answer}")
                    print(f"Normalized true: {true_norm}")
                    print(f"Match: {pred_norm == true_norm}")
                    debug_count += 1
                count+=1
                if pred_norm == true_norm:
                    correct += 1
                
                total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        # Save results
        results_file = self.output_dir / "gsm8k_results.txt"
        with open(results_file, "a") as f:
            f.write(f"{model_path},eval_seed={self.config.seed},temperature={self.config.eval_temperature}    {accuracy/100:.4f}\n")
        
        return accuracy
    
    def _evaluate_with_voting(self, model: nn.Module) -> float:
        """
        Evaluate with majority voting (multiple samples per question).
        
        More robust but slower evaluation:
        - Generate multiple samples per question (default 3)
        - Use majority voting to select final answer
        - Uses both GSM8K #### format and boxed format extraction
        """
        model.eval()
        
        correct = 0
        total = 0
        
        # Switch to left padding for generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        print(f"  Switch to left padding for generation ⚠️  Padding side: {self.tokenizer.padding_side}")

        # Use full eval dataset for final evaluation
        eval_dataset = self.final_eval_dataset
        
        # Create temporary dataloader for evaluation
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.preprocessing_num_workers,
            collate_fn=MathDataCollator(self.tokenizer, self.config),
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Get original data for this batch
                batch_start = batch_idx * self.config.batch_size
                batch_end = min(batch_start + self.config.batch_size, len(eval_dataset))
                batch_data = [eval_dataset[i] for i in range(batch_start, batch_end)]
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                numerical_answers = [item["numerical_answer"] for item in batch_data]
                
                batch_size = input_ids.shape[0]
                
                # Generate multiple samples per question
                all_predictions = [[] for _ in range(batch_size)]
                
                for sample_idx in range(self.config.num_eval_samples):
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_answer_length or 1024,
                        do_sample=True,
                        temperature=self.config.eval_temperature,
                        top_p=self.config.eval_top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    # Decode
                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids[:, input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )
                    
                    # Extract answers (try GSM8K format first, then boxed)
                    for i, generated_text in enumerate(generated_texts):
                        pred_answer = extract_gsm8k_answer(generated_text)
                        if pred_answer is None:
                            pred_answer = extract_numerical_answer(generated_text)
                        pred_norm = normalize_answer(pred_answer)
                        all_predictions[i].append(pred_norm)
                
                # Majority voting
                for predictions, true_answer in zip(all_predictions, numerical_answers):
                    final_pred = majority_vote(predictions)
                    true_norm = normalize_answer(true_answer)
                    
                    if final_pred == true_norm:
                        correct += 1
                    
                    total += 1
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        print(f"  ✓ Restored original padding side")
        
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        return accuracy
    
    def _save_checkpoint(self, model: nn.Module, method: str, epoch: int, accuracy: float):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"{method}_seed{self.config.seed}_epoch{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (only trainable parameters to save space)
        trainable_state = {
            name: param for name, param in model.named_parameters() if param.requires_grad
        }
        
        torch.save({
            "epoch": epoch,
            "method": method,
            "accuracy": accuracy,
            "trainable_state_dict": trainable_state,
            "config": self.config.__dict__,
        }, checkpoint_dir / "checkpoint.pt")
        
        print(f"✓ Saved checkpoint to {checkpoint_dir}")
    
    def train_all_methods(self) -> Dict:
        """Train all 4 methods and compare."""
        methods = ["way0", "way1", "way2", "way3"]
        results = {}
        
        for method in methods:
            print(f"\n{'='*70}")
            print(f"🔄 Training method: {method}")
            print(f"{'='*70}\n")
            
            result = self.train_single_method(method)
            results[method] = result
            
            # Clear VRAM between methods
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict):
        """Print comparison of all methods."""
        print(f"\n{'='*70}")
        print(f"📊 COMPARISON OF ALL METHODS")
        print(f"{'='*70}\n")
        
        print(f"{'Method':<10} {'Best Acc':<12} {'Final Acc (voting)':<20}")
        print(f"{'-'*50}")
        
        for method, result in results.items():
            print(f"{method:<10} {result['best_accuracy']:>10.2f}% {result['final_accuracy']:>18.2f}%")
        
        print(f"\n{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LLM on Math Reasoning with Rotational PiSSA")
    
    # Model and dataset
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="HuggingFace model name or path (default: meta-llama/Llama-2-7b-hf )")
    parser.add_argument("--dataset", type=str, default="metamath",
                       choices=["gsm8k", "metamath", "math"],
                       help="Dataset to use (train on metamath, test on gsm8k)")
    
    # Method
    parser.add_argument("--method", type=str, default="way0",
                       choices=["way0", "way1", "way2", "way3", "all"],
                       help="Rotational method")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs (paper uses 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per device (paper uses 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                       help="Gradient accumulation steps (paper uses 32, effective batch=32)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate (paper uses 2e-5)")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Maximum sequence length (paper uses 1024)")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                       help="Disable gradient checkpointing (enabled by default for memory efficiency)")
    
    # PiSSA
    parser.add_argument("--rank", type=int, default=16,
                       help="PiSSA rank")
    parser.add_argument("--alpha", type=float, default=16.0,
                       help="PiSSA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                       help="LoRA dropout")
    parser.add_argument("--orthogonality-weight", type=float, default=1e-3,
                       help="Orthogonality regularization weight")
    parser.add_argument("--regularization-type", type=str, default="log_determinant",
                       choices=["frobenius", "determinant", "log_determinant"],
                       help="Regularization type for way0")
    parser.add_argument("--steps-per-phase", type=int, default=0,
                       help="Steps per phase for way1 (0 = auto)")
    parser.add_argument("--total-cycles", type=int, default=2,
                       help="Total cycles for way1")
    parser.add_argument("--low-rank-r", type=int, default=4,
                       help="Low rank for way2/way3")
    parser.add_argument("--quantize-residual", action="store_true",
                       help="NF4 quantize residual")
    parser.add_argument("--quantize-base-components", action="store_true",
                       help="NF4 quantize base components")
    
    # Evaluation
    parser.add_argument("--eval-only", action="store_true",
                       help="Skip training and only run evaluation")
    parser.add_argument("--evaluation-strategy", type=str, default="single",
                       choices=["single", "voting"],
                       help="Evaluation strategy: 'single' (greedy, fast) or 'voting' (majority vote, robust)")
    parser.add_argument("--num-eval-samples", type=int, default=3,
                       help="Number of samples for majority voting (only used with --evaluation-strategy voting)")
    parser.add_argument("--use-vllm", action="store_true",
                       help="Use vLLM for faster inference during evaluation (requires merged model)")
    parser.add_argument("--eval-temperature", type=float, default=0.8,
                       help="Temperature for sampling during evaluation")
    parser.add_argument("--eval-top-p", type=float, default=0.95,
                       help="Top-p for sampling during evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=4,
                       help="Batch size for vLLM evaluation")
    parser.add_argument("--max-answer-length", type=int, default=None,
                       help="Maximum number of new tokens to generate (default: None = 1024)")
    
    # Other
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--merged-model-path", type=str, default=None,
                       help="Path to merged model for evaluation (required if --eval-only)")
    
    args = parser.parse_args()
    
    # Create config
    config = MathTrainingConfig(
        model_name_or_path=args.model,
        dataset=args.dataset,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        pissa_rank=args.rank,
        pissa_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        orthogonality_weight=args.orthogonality_weight,
        regularization_type=args.regularization_type,
        steps_per_phase=args.steps_per_phase,
        total_cycles=args.total_cycles,
        low_rank_r=args.low_rank_r,
        quantize_residual=args.quantize_residual,
        quantize_base_components=args.quantize_base_components,
        evaluation_strategy=args.evaluation_strategy,
        num_eval_samples=args.num_eval_samples,
        use_vllm=True,
        eval_batch_size=args.eval_batch_size,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        max_answer_length=args.max_answer_length,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )
    
    print("train_config:", config)
    
    
    
    trainer = LLMMathTrainer(config)
    
    if args.eval_only:
        if not args.merged_model_path:
            print("❌ Error: --merged-model-path is required for --eval-only")
            return
        
        print(f"\n🚀 Running evaluation only on {args.merged_model_path}...")
        if args.use_vllm:
            accuracy = trainer._evaluate_with_vllm(args.merged_model_path)
        else:
            print("⚠️  Standard evaluation requires loading model first. Please use --use-vllm for eval-only mode.")
            return
            
        print(f"Final Accuracy: {accuracy:.2f}%")
        return

    if args.method == "all":
        trainer.train_all_methods()
    else:
        trainer.train_single_method(config.method)


if __name__ == "__main__":
    main()
