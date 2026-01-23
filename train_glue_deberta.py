"""
GLUE Benchmark Training with DeBERTaV3-base + Rotational PiSSA
================================================================

Trains DeBERTaV3-base on all GLUE tasks using Rotational PiSSA (all 4 ways).
Reports results on development set with 5-seed averaging.

Tasks: MNLI, SST-2, CoLA, QQP, QNLI, RTE, MRPC, STS-B
Methods: way0, way1, way2, way3

Usage:
    python train_glue_deberta.py --task sst2 --method way0 --seed 42
    python train_glue_deberta.py --task all --method all --seeds 42,1234,2024,7890,5555
"""

import os
import sys
import argparse
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

# Import Rotational PiSSA modules
from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    RotationalLinearLayer,
    replace_linear_with_rotational_pissa,
    RotationalPiSSATrainer,
)

# ============================================================================
# GLUE TASK CONFIGURATIONS
# ============================================================================

GLUE_TASKS = {
    "mnli": {
        "num_labels": 3,
        "metric": "accuracy",  # Report m/mm accuracy
        "keys": ("premise", "hypothesis"),
        "is_regression": False,
        "validation_key": "validation_matched",  # Also has validation_mismatched
    },
    "sst2": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("sentence", None),
        "is_regression": False,
        "validation_key": "validation",
    },
    "cola": {
        "num_labels": 2,
        "metric": "matthews",
        "keys": ("sentence", None),
        "is_regression": False,
        "validation_key": "validation",
    },
    "qqp": {
        "num_labels": 2,
        "metric": "acc_f1",
        "keys": ("question1", "question2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "qnli": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("question", "sentence"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "rte": {
        "num_labels": 2,
        "metric": "accuracy",
        "keys": ("sentence1", "sentence2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "mrpc": {
        "num_labels": 2,
        "metric": "acc_f1",
        "keys": ("sentence1", "sentence2"),
        "is_regression": False,
        "validation_key": "validation",
    },
    "stsb": {
        "num_labels": 1,
        "metric": "spearman",
        "keys": ("sentence1", "sentence2"),
        "is_regression": True,
        "validation_key": "validation",
    },
}


@dataclass
class GLUEConfig:
    """Configuration for GLUE benchmark training."""
    # Model
    model_name: str = "microsoft/deberta-v3-base"
    
    # Task
    task: str = "sst2"
    
    # Rotational PiSSA
    method: str = "way0"
    pissa_rank: int = 8
    pissa_alpha: float = 16.0
    orthogonality_weight: float = 1e-4
    low_rank_r: int = 4
    total_cycles: int = 3  # For way1: how many full cycles through all layers
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 32
    max_seq_length: int = 256
    warmup_ratio: float = 0.06
    
    # Seeds for averaging
    seed: int = 42
    
    # Logging
    use_wandb: bool = True
    project_name: str = "glue-deberta-rotational-pissa"
    output_dir: str = "./glue_results"
    logging_steps: int = 100
    logging_steps: int = 100
    max_steps: int = -1
    track_grad_norm: bool = False  # If False, skip computing/logging grad norm for speed
    
    # Model architecture
    no_pooler: bool = False  # If True, use CLS token directly instead of pooler
    
    # Device
    device: str = "auto"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_glue_dataset(task: str, tokenizer, max_seq_length: int = 128):
    """Load and tokenize a GLUE dataset."""
    task_config = GLUE_TASKS[task]
    key1, key2 = task_config["keys"]
    
    # Load dataset
    dataset = load_dataset("glue", task)
    
    def tokenize_function(examples):
        if key2 is None:
            return tokenizer(
                examples[key1],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
        else:
            return tokenizer(
                examples[key1],
                examples[key2],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
    
    # Get columns to remove (all text columns except label)
    columns_to_remove = [col for col in dataset["train"].column_names 
                         if col not in ["label", "labels"]]
    
    # Tokenize - preserve label column
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    
    # Rename label -> labels for consistency with HuggingFace Trainer
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    
    return tokenized


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(task: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute task-specific metrics."""
    task_config = GLUE_TASKS[task]
    metric_type = task_config["metric"]
    is_regression = task_config["is_regression"]
    
    if is_regression:
        # STS-B: Spearman and Pearson correlation
        spearman = spearmanr(predictions, labels)[0]
        pearson = pearsonr(predictions, labels)[0]
        return {"spearman": spearman, "pearson": pearson, "corr": (spearman + pearson) / 2}
    
    # Classification: convert logits to predictions
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    if metric_type == "accuracy":
        return {"accuracy": accuracy_score(labels, predictions)}
    
    elif metric_type == "matthews":
        return {"matthews": matthews_corrcoef(labels, predictions)}
    
    elif metric_type == "acc_f1":
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary")
        return {"accuracy": acc, "f1": f1, "acc_f1": (acc + f1) / 2}
    
    return {}


# ============================================================================
# TRAINER CLASS
# ============================================================================

class GLUETrainer:
    """Trainer for GLUE tasks with Rotational PiSSA."""
    
    def __init__(self, config: GLUEConfig):
        self.config = config
        self.task_config = GLUE_TASKS[config.task]
        
        # Set seed
        self._set_seed(config.seed)
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"\n{'='*60}")
        print(f"Task: {config.task.upper()}")
        print(f"Method: {config.method}")
        print(f"Seed: {config.seed}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load tokenizer and model
        self._load_model()
        
        # Load data
        self._load_data()
        
        # Apply Rotational PiSSA
        self._apply_rotational_pissa()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_model(self):
        """Load DeBERTaV3 model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.task_config["num_labels"],
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=model_config,
        )
        
        # Remove pooler if requested - use CLS token directly
        if self.config.no_pooler:
            print("  Removing pooler - using [CLS] token directly for classification")
            # Replace pooler with identity (just passes through the hidden state)
            class IdentityPooler(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense = torch.nn.Identity()
                    self.dropout = torch.nn.Dropout(0)
                def forward(self, hidden_states):
                    # Just return the [CLS] token (first token)
                    return hidden_states[:, 0]
            
            self.model.pooler = IdentityPooler(model_config)
            # Reinitialize classifier to match hidden_size directly
            self.model.classifier = torch.nn.Linear(
                model_config.hidden_size, 
                self.task_config["num_labels"]
            )
            print(f"  New classifier: Linear({model_config.hidden_size}, {self.task_config['num_labels']})")
        
        self.model.to(self.device)
    
    def _load_data(self):
        """Load GLUE dataset."""
        print(f"Loading dataset: {self.config.task}")
        
        dataset = load_glue_dataset(
            self.config.task,
            self.tokenizer,
            self.config.max_seq_length,
        )
        
        # Data collator
        collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        
        # Create dataloaders (with parallel loading for speed)
        self.train_loader = DataLoader(
            dataset["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )
        
        val_key = self.task_config["validation_key"]
        self.val_loader = DataLoader(
            dataset[val_key],
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )
        
        # For MNLI, also load mismatched validation
        if self.config.task == "mnli":
            self.val_loader_mm = DataLoader(
                dataset["validation_mismatched"],
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=4,
                pin_memory=True,
            )
        
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Val samples: {len(dataset[val_key])}")
    
    def _apply_rotational_pissa(self):
        """Apply Rotational PiSSA to the model."""
        print(f"\nApplying Rotational PiSSA (method={self.config.method}, rank={self.config.pissa_rank})")
        
        # Configure PiSSA
        pissa_config = RotationalPiSSAConfig(
            r=self.config.pissa_rank,
            lora_alpha=self.config.pissa_alpha,
            method=self.config.method,
            orthogonality_reg_weight=self.config.orthogonality_weight,
            low_rank_r=self.config.low_rank_r,
            total_cycles=self.config.total_cycles,
            init_identity=True,
            freeze_singular_values=False,
            s_dtype_fp32=True,
        )
        
        # Target modules for DeBERTa:
        # - query_proj, key_proj, value_proj: attention Q, K, V projections
        # - attention.output.dense: attention O projection  
        # - intermediate.dense: FFN up-projection (768 -> 3072)
        # - output.dense: FFN down-projection (3072 -> 768)
        # We use "dense" which matches all dense layers in encoder
        # BUT we exclude pooler.dense and classifier (these need full training)
        # target_modules = ["query_proj", "key_proj", "value_proj"]
        # All encoder dense layers (not pooler)

        target_modules = [
            "query_proj", "key_proj", "value_proj",  # Q, K, V
            "attention.output.dense",                 # O projection
            "intermediate.dense",                     # FFN up
            "output.dense",                           # FFN down (careful: not pooler)
        ]
        exclude_modules = ["pooler", "classifier"]  # Don't apply PiSSA to these
        
        # Replace linear layers (excludes pooler and classifier)
        self.adapters = replace_linear_with_rotational_pissa(
            self.model,
            pissa_config,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            freeze_base_model=True,
        )
        
        # Create trainer for orthogonality loss
        self.rotational_trainer = RotationalPiSSATrainer(self.model, pissa_config)
        self.pissa_config = pissa_config
        
        # CRITICAL: Unfreeze classifier (and pooler if present) - they're randomly initialized!
        # They need FULL training, not PiSSA adaptation
        print("  Making classifier (and pooler if present) fully trainable (no PiSSA):")
        for name, param in self.model.named_parameters():
            if "classifier" in name or ("pooler" in name and not self.config.no_pooler):
                param.requires_grad = True
                print(f"    Unfreezing: {name} ({param.numel():,} params)")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
    def train(self) -> Dict[str, float]:
        """Train the model and return final metrics."""
        # Initialize wandb (skip if already running from sweep agent)
        if self.config.use_wandb and wandb.run is None:
            run_name = f"{self.config.task}_{self.config.method}_seed{self.config.seed}"
            wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={
                    "task": self.config.task,
                    "method": self.config.method,
                    "seed": self.config.seed,
                    "rank": self.config.pissa_rank,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                },
            )
        # If wandb.run exists (from sweep), enable logging
        elif wandb.run is not None:
            self.config.use_wandb = True
        
        best_metric = -float("inf")
        best_results = {}
        
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, avg_grad_norm = self._train_epoch(epoch)
            
            # Evaluate
            results = self._evaluate()
            
            # Get primary metric
            if self.task_config["metric"] == "accuracy":
                primary_metric = results.get("accuracy", 0)
            elif self.task_config["metric"] == "matthews":
                primary_metric = results.get("matthews", 0)
            elif self.task_config["metric"] == "acc_f1":
                primary_metric = results.get("acc_f1", 0)
            elif self.task_config["metric"] == "spearman":
                primary_metric = results.get("corr", 0)
            else:
                primary_metric = results.get("accuracy", 0)
            
            if primary_metric > best_metric:
                best_metric = primary_metric
                best_results = results.copy()
            
            # Log to wandb
            if self.config.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "avg_grad_norm": avg_grad_norm,
                    **{f"val_{k}": v for k, v in results.items()},
                    "best_metric": best_metric,
                }
                wandb.log(log_dict)
            
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Loss: {train_loss:.4f} - Metric: {primary_metric:.4f} - Grad Norm: {avg_grad_norm:.4f}")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return best_results
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_grad_norm = 0
        num_steps = 0
        num_logging_steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Add orthogonality regularization for way0
            if self.pissa_config.method == "way0":
                ortho_loss = self.rotational_trainer.get_orthogonality_loss()
                loss = loss + ortho_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            # Always clip for stability, but optionally track norm
            # If tracking is disabled, we skip the expensive .item() synchronization
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimization: Only sync to CPU (item()) if we are going to log
            is_logging_step = (num_steps % self.config.logging_steps == 0)
            
            if is_logging_step:
                # Efficiently handle grad norm tracking
                if self.config.track_grad_norm:
                    grad_norm_val = grad_norm.item()
                    loss_val = loss.item() # Sync loss too if tracking is robust
                    total_grad_norm += grad_norm_val
                else:
                    grad_norm_val = 0.0
                    # Note: We skip total_grad_norm accumulation if not tracking to avoid mix of real/zero values
                
                # loss.item() is also a sync, but usually we want loss logged.
                # Assuming user prioritized grad norm sync removal.
                # We will still sync loss for wandb.
                loss_val = loss.item() 
                num_logging_steps += 1

            num_steps += 1
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update Rotational PiSSA phase (for Way 1)
            if self.pissa_config.method == "way1" and self.rotational_trainer.should_step_phase(num_steps):
                self.rotational_trainer.step_phase()
            
            # Accumulate loss on GPU to avoid sync
            total_loss += loss.detach()
            
            # Log per-step metrics to wandb
            if self.config.use_wandb and is_logging_step:
                wandb.log({
                    "train/loss": loss_val,
                    "train/grad_norm": grad_norm_val,
                    "train/lr": self.scheduler.get_last_lr()[0],
                })
            
            # Stop if max_steps reached
            if self.config.max_steps > 0 and num_steps >= self.config.max_steps:
                break
        
        avg_loss = total_loss.item() / len(self.train_loader)
        avg_grad_norm = total_grad_norm / max(1, num_logging_steps)
        
        return avg_loss, avg_grad_norm
    
    def _evaluate(self, loader=None) -> Dict[str, float]:
        """Evaluate on validation set."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = self.model(**batch)
                
                if self.task_config["is_regression"]:
                    predictions = outputs.logits.squeeze(-1)
                else:
                    predictions = outputs.logits
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        results = compute_metrics(self.config.task, predictions, labels)
        
        # For MNLI, also evaluate on mismatched
        if self.config.task == "mnli" and loader == self.val_loader:
            mm_results = self._evaluate(self.val_loader_mm)
            results["accuracy_mm"] = mm_results["accuracy"]
            results["m_mm"] = (results["accuracy"] + results["accuracy_mm"]) / 2
        
        return results


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_single_experiment(config: GLUEConfig) -> Dict[str, float]:
    """Run a single experiment and return results."""
    print("Running experiment with config:", config)
    trainer = GLUETrainer(config)
    return trainer.train()


def main():
    parser = argparse.ArgumentParser(description="GLUE Benchmark with DeBERTaV3 + Rotational PiSSA")
    
    # Task and method
    parser.add_argument("--task", type=str, default="cola",
                        choices=list(GLUE_TASKS.keys()),
                        help="GLUE task to run")
    parser.add_argument("--method", type=str, default="way0",
                        choices=["way0", "way1", "way2", "way3"],
                        help="Rotational PiSSA method")
    
    # Model and training
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--orthogonality_weight", type=float, default=1e-4)
    parser.add_argument("--total_cycles", type=int, default=3,
                        help="For way1: how many full cycles through all layers")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit number of steps per epoch for debugging")
    parser.add_argument("--no-pooler", action="store_true",
                        help="Remove pooler, use [CLS] token directly for classification")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./glue_results")
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Single task, single method
    config = GLUEConfig(
        task=args.task,
        method=args.method,
        model_name=args.model,
        pissa_rank=args.rank,
        orthogonality_weight=args.orthogonality_weight,
        total_cycles=args.total_cycles,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs= args.epochs if args.task != "stsb" else 35,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        no_pooler=args.no_pooler,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
    )
    
    results = run_single_experiment(config)
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()
