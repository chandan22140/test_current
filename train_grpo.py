"""
GRPO training script for google/gemma-3-270m on MetaMathQA (full).

- Training flow follows llama3_1_(8b)_grpo.py (GRPO with reward functions)
- PEFT model setup follows float_llama2-7b_metamath.py (RotationalPiSSA)
- Supports vLLM inference via TRL's GRPOConfig(use_vllm=True)
"""

import re
import os
import torch
from fire import Fire
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import GRPOConfig, GRPOTrainer

from rotational_pissa_unified import (
    RotationalPiSSAConfig,
    replace_linear_with_rotational_pissa,
)
from utils import find_all_linear_modules
# CHANGED: Import get_chat_template
from get_chat_template import get_chat_template
import wandb


# ─────────────────────── Prompt & Format Templates ───────────────────────

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


# ─────────────────────── Answer Extraction Helpers ───────────────────────

def extract_xml_answer(text: str) -> str:
    """Extract content between <answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract answer after #### from MetaMathQA responses."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_metamath_answer(text: str) -> str | None:
    """Extract answer from MetaMathQA 'The answer is: X' format."""
    if "The answer is:" in text:
        return text.split("The answer is:")[-1].strip().rstrip(".")
    return extract_hash_answer(text)


# ─────────────────────── Dataset Loading ─────────────────────────────────

def get_metamath_questions(max_samples: int = None, max_tokens: int = 512) -> Dataset:
    """Load MetaMathQA full dataset and format for GRPO training."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

    processed = []
    for sample in dataset:
        answer = extract_metamath_answer(sample["response"])
        if answer is None:
            continue

        # Build the prompt in chat format
        prompt = [
            {"role": "user", "content": SYSTEM_PROMPT.strip() + "\n\n" + sample["query"]},
        ]

        # Filter out overly long samples
        flat_text = sample["query"] + " " + sample["response"]
        if len(tokenizer(flat_text)["input_ids"]) >= max_tokens:
            continue

        processed.append({
            "prompt": prompt,
            "answer": answer,
        })

        if max_samples is not None and len(processed) >= max_samples:
            break

    print(f"Loaded {len(processed)} samples from MetaMathQA (full)")
    return Dataset.from_list(processed)


# ─────────────────────── Reward Functions ────────────────────────────────

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward +2.0 for correct answer, 0.0 otherwise."""
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward +0.5 if extracted answer is a number."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.replace(".", "", 1).replace("-", "", 1).isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward +0.5 for strict XML format compliance."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward +0.5 for soft XML format compliance."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Fine-grained reward for partial XML formatting."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward based on XML tag counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ─────────────────────── Main ────────────────────────────────────────────

def main(
    # Model
    model_id: str = "google/gemma-3-270m",
    model_dtype: str = "bf16",
    # Dataset
    dataset_name: str = "meta-math/MetaMathQA",
    max_samples: int = None,  # None = use full dataset
    max_tokens: int = 512,
    # PEFT (RotationalPiSSA) — matches float_llama2-7b_metamath.py
    lora_rank: int = 128,
    lora_alpha: int = 128,
    method: str = "way0",
    total_cycles: int = 4,
    use_butterfly: bool = False,
    butterfly_sequential: bool = False,
    # GRPO Training
    max_seq_length: int = 1024,
    max_completion_length: int = 768,
    num_train_epochs: int = 1,
    max_steps: int = -1,
    per_device_train_batch_size: int = 4,  # Must be divisible by num_generations
    gradient_accumulation_steps: int = 1,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.1,
    max_grad_norm: float = 0.1,
    logging_steps: int = 1,
    save_steps: int = 250,
    # vLLM
    use_vllm: bool = True,
    vllm_gpu_memory_utilization: float = 0.5,
    # Misc
    output_dir: str = "outputs",
    seed: int = 42,
    report_to: str = "wandb",
    wandb_project: str = "Gemma-GRPO-SOARA",
):
    """
    GRPO + RotationalPiSSA training on MetaMathQA.
    Supports vLLM for fast generation during GRPO rollouts.
    """
    # ── wandb ──
    config = dict(
        model=model_id.replace("/", "_"),
        d=dataset_name.replace("/", "_"),
        a=lora_alpha,
        r=lora_rank,
        method=method,
        butterfly=use_butterfly,
        seq=butterfly_sequential,
    )
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    if butterfly_sequential:
        wandb_name = "butterfly_seq_" + wandb_name

    if report_to == "wandb":
        wandb.init(name=wandb_name, mode="online", project=wandb_project)

    # ── Load model & tokenizer ──
    torch_dtype = torch.bfloat16 if model_dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",  # Small model, eager is fine
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # CHANGED: Apply chat template
    print("Applying Gemma 3 chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded model: {model_id}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Default rank for butterfly mode ──
    if use_butterfly and lora_rank == 128:
        config_obj = AutoConfig.from_pretrained(model_id)
        lora_rank = config_obj.hidden_size
        print(f"🦋 Butterfly mode: defaulting rank to d_model={lora_rank}")

    # ── Apply RotationalPiSSA PEFT (from float_llama2-7b_metamath.py) ──
    pissa_config = RotationalPiSSAConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        method=method,
        total_cycles=total_cycles,
        use_butterfly=use_butterfly,
        butterfly_sequential=butterfly_sequential,
        orthogonality_reg_weight=0,
        init_identity=True,
        freeze_singular_values=False,
        quantize_residual=False,
        quantize_base_components=False,
    )
    print(f"PEFT config: {pissa_config}")

    target_modules = find_all_linear_modules(model=model)
    print(f"Target modules: {target_modules}")

    device = next(model.parameters()).device
    adapters = replace_linear_with_rotational_pissa(
        model=model,
        pissa_config=pissa_config,
        target_modules=target_modules,
        adapter_name="default",
        freeze_base_model=True,
        device=device,
    )
    print("RotationalPiSSA initialization complete.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    # ── Save init checkpoint ──
    save_dir = os.path.join("./snapshot", wandb_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pissa_config": pissa_config,
            "adapters": list(adapters.keys()),
        },
        os.path.join(save_dir, "init_checkpoint.pt"),
    )

    # ── Load dataset ──
    dataset = get_metamath_questions(max_samples=max_samples, max_tokens=max_tokens)

    # ── GRPO Config ──
    grpo_kwargs = dict(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-10,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_grad_norm=max_grad_norm,
        report_to=report_to,
        output_dir=output_dir,
        save_steps=save_steps,
        bf16=(model_dtype == "bf16"),
        seed=seed,
    )

    # epochs vs steps
    if max_steps > 0:
        grpo_kwargs["max_steps"] = max_steps
    else:
        grpo_kwargs["num_train_epochs"] = num_train_epochs

    # vLLM inference support
    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization

    training_args = GRPOConfig(**grpo_kwargs)

    # ── Train ──
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # ── Save final checkpoint ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pissa_config": pissa_config,
            "adapters": list(adapters.keys()),
        },
        os.path.join(save_dir, "final_checkpoint.pt"),
    )
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    Fire(main)
