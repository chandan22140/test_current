#!/bin/bash
# Launch GRPO training with vLLM acceleration
# GPU 0: vLLM inference server
# GPU 1: Training process
#
# Usage: bash run_grpo_vllm.sh [model_id] [output_dir]
#
# Default: google/gemma-3-270m

MODEL_ID="${1:-google/gemma-3-270m}"
OUTPUT_DIR="${2:-outputs/grpo_gemma270m_vllm}"
VLLM_PORT=8000
VLLM_GPU=0
TRAIN_GPU=1

echo "==========================================="
echo " GRPO Training with vLLM + RotationalPiSSA"
echo "==========================================="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "vLLM Server: GPU $VLLM_GPU (port $VLLM_PORT)"
echo "Training:    GPU $TRAIN_GPU"
echo "==========================================="

# Step 1: Start vLLM server on GPU 0 in the background
echo "[1/2] Starting vLLM server on GPU $VLLM_GPU..."
CUDA_VISIBLE_DEVICES=$VLLM_GPU trl vllm-serve \
    --model "$MODEL_ID" \
    --port $VLLM_PORT \
    --gpu_memory_utilization 0.3 &
VLLM_PID=$!

# Give server time to start
echo "Waiting for vLLM server to initialize..."
sleep 10

# Check if server started
for i in $(seq 1 30); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs above."
        exit 1
    fi
    echo "Waiting for vLLM server... ($i/30)"
    sleep 5
done

# Step 2: Run training on GPU 1
echo "[2/2] Starting GRPO + RotationalPiSSA training on GPU $TRAIN_GPU..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPU python3 train_grpo.py \
    --model_id "$MODEL_ID" \
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
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.3 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --method way0 \
    --total_cycles 4 \
    --seed 42 \
    --report_to wandb \
    --wandb_project "Gemma-GRPO-SOARA" \
    --output_dir "$OUTPUT_DIR"
TRAIN_EXIT=$?

# Cleanup: kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done! Exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
