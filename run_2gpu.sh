#!/bin/bash

# usage: ./run_2gpu.sh [args...]
# e.g. ./run_2gpu.sh --method way1 --total_cycles 4 --epochs 1

export CUDA_VISIBLE_DEVICES=0,1

# NCCL workarounds for kernel < 5.5.0 (current: 5.4.0)
# P2P and IB disabled to use shared memory instead of problematic kernel features
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# Extended timeout (30 min instead of 10 min) for slow operations
export NCCL_TIMEOUT=1800000

# Launch with accelerate
# --multi_gpu: Enable multi-GPU support
# --num_processes 2: Use 2 GPUs
# --mixed_precision bf16: Optional but recommended if supported
accelerate launch \
    --config_file deepspeed_config.yaml \
    float_llama2-7b_metamath.py --method way1 --total_cycles 7 --epochs 3 --use-butterfly --butterfly-sequential "$@"
