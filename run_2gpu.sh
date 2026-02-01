#!/bin/bash

# usage: ./run_2gpu.sh [args...]
# e.g. ./run_2gpu.sh --method way1 --total_cycles 4 --epochs 1

export CUDA_VISIBLE_DEVICES=0,1

# Launch with accelerate
# --multi_gpu: Enable multi-GPU support
# --num_processes 2: Use 2 GPUs
# --mixed_precision bf16: Optional but recommended if supported
accelerate launch \
    --config_file deepspeed_config.yaml \
    float_llama2-7b_metamath.py --method way1 --total_cycles 7 --epochs 3 --use-butterfly --butterfly-sequential "$@"
