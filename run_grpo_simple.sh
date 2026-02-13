#!/bin/bash
# Simple single-GPU GRPO training WITHOUT vLLM server
# For small models like gemma-3-270m, regular generation is fast enough
#
# Usage: bash run_grpo_simple.sh [gpu_id]

GPU="${1:-0}"

echo "==========================================="
echo " GRPO Training (no vLLM) + RotationalPiSSA"
echo "==========================================="
echo "GPU: $GPU"
echo "==========================================="

CUDA_VISIBLE_DEVICES=$GPU python3 train_grpo.py \
    --model_id "google/gemma-3-270m" \
    --max_samples None \
    --max_tokens 512 \
    --num_train_epochs 1 \
    --num_generations 4 \
    --max_completion_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --logging_steps 1 \
    --save_steps 250 \
    --use_vllm False \
    --lora_rank 8 \
    --lora_alpha 8 \
    --method way0 \
    --total_cycles 4 \
    --seed 42 \
    --report_to wandb \
    --wandb_project "Gemma-GRPO-SOARA" \
    --output_dir "outputs/grpo_gemma270m"
