#!/bin/bash
# ==============================================================================
# SOARA Ablation Study Runner
# ==============================================================================
# This script runs the ablation study for different SOARA configurations
# and generates t-SNE visualizations for embedding analysis.
#
# Usage:
#   ./run_ablation.sh [--sweep | --train | --tsne | --all]
#
# Options:
#   --sweep   Create wandb sweeps for all configurations
#   --train   Run direct training (no sweep) for all configurations
#   --tsne    Run t-SNE analysis on existing checkpoints
#   --all     Run both training and t-SNE analysis
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="${SCRIPT_DIR}/yaml_soara_ablation"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/soara_ablation"
TSNE_OUTPUT_DIR="${SCRIPT_DIR}/tsne_plots"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TSNE_OUTPUT_DIR"

# -----------------------------------------------------------------------------
# SWEEP MODE: Create and run wandb sweeps
# -----------------------------------------------------------------------------
run_sweeps() {
    echo "=============================================="
    echo "Creating wandb sweeps for SOARA ablation"
    echo "=============================================="
    
    # Way0 with different ranks
    for rank in 2 4 8 16; do
        yaml_file="${YAML_DIR}/sweep_fgvc_way0_r${rank}.yaml"
        if [ -f "$yaml_file" ]; then
            echo ""
            echo "Creating sweep for way0 rank=${rank}..."
            wandb sweep "$yaml_file" 2>&1 | tee "${OUTPUT_DIR}/sweep_way0_r${rank}.log"
        else
            echo "Warning: $yaml_file not found, skipping..."
        fi
    done
    
    # Way1 butterfly sequential
    yaml_file="${YAML_DIR}/sweep_fgvc_way1_bf_seq.yaml"
    if [ -f "$yaml_file" ]; then
        echo ""
        echo "Creating sweep for way1 butterfly sequential..."
        wandb sweep "$yaml_file" 2>&1 | tee "${OUTPUT_DIR}/sweep_way1_bf_seq.log"
    else
        echo "Warning: $yaml_file not found, skipping..."
    fi
    
    echo ""
    echo "=============================================="
    echo "Sweeps created! Check wandb dashboard for sweep IDs."
    echo "Run agents with: wandb agent <username/project/sweep_id>"
    echo "=============================================="
}

# -----------------------------------------------------------------------------
# DIRECT TRAINING MODE: Train without sweeps (fixed hyperparameters)
# -----------------------------------------------------------------------------
run_direct_training() {
    echo "=============================================="
    echo "Running direct training for SOARA ablation"
    echo "=============================================="
    
    # Fixed hyperparameters from sweep results
    LR=0.00090625
    WEIGHT_DECAY=0.000018778
    ORTHO_WEIGHT=0.0018789
    EPOCHS=10
    BATCH_SIZE=32
    
    # Way0 with different ranks
    for rank in 2 4 8 16; do
        echo ""
        echo "Training way0 rank=${rank}..."
        python train_vit_rotational.py \
            --model google/vit-base-patch16-224 \
            --dataset fgvc_aircraft \
            --method way0 \
            --rank $rank \
            --learning-rate $LR \
            --weight-decay $WEIGHT_DECAY \
            --orthogonality-weight $ORTHO_WEIGHT \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --output-dir "${OUTPUT_DIR}/way0_r${rank}" \
            --experiment-name "ablation_way0_r${rank}" \
            2>&1 | tee "${OUTPUT_DIR}/train_way0_r${rank}.log"
    done
    
    # Way1 butterfly sequential
    echo ""
    echo "Training way1 butterfly sequential..."
    python train_vit_rotational.py \
        --model google/vit-base-patch16-224 \
        --dataset fgvc_aircraft \
        --method way1 \
        --rank 768 \
        --use-butterfly \
        --butterfly-sequential \
        --total-cycles 3 \
        --learning-rate 0.001 \
        --weight-decay 0.001 \
        --lr-ratio-s 10.0 \
        --epochs 30 \
        --batch-size $BATCH_SIZE \
        --output-dir "${OUTPUT_DIR}/way1_bf_seq" \
        --experiment-name "ablation_way1_bf_seq" \
        2>&1 | tee "${OUTPUT_DIR}/train_way1_bf_seq.log"
    
    echo ""
    echo "=============================================="
    echo "Training complete! Checkpoints in: ${OUTPUT_DIR}"
    echo "=============================================="
}

# -----------------------------------------------------------------------------
# t-SNE ANALYSIS MODE: Generate visualizations from checkpoints
# -----------------------------------------------------------------------------
run_tsne_analysis() {
    echo "=============================================="
    echo "Running t-SNE analysis on trained models"
    echo "=============================================="
    
    # Collect checkpoint paths
    CHECKPOINTS=""
    
    for rank in 2 4 8 16; do
        ckpt_dir="${OUTPUT_DIR}/way0_r${rank}"
        if [ -d "$ckpt_dir" ]; then
            ckpt_file=$(find "$ckpt_dir" -name "*_best_model.pth" 2>/dev/null | head -1)
            if [ -n "$ckpt_file" ]; then
                CHECKPOINTS="$CHECKPOINTS way0_r${rank}=$ckpt_file"
            fi
        fi
    done
    
    ckpt_dir="${OUTPUT_DIR}/way1_bf_seq"
    if [ -d "$ckpt_dir" ]; then
        ckpt_file=$(find "$ckpt_dir" -name "*_best_model.pth" 2>/dev/null | head -1)
        if [ -n "$ckpt_file" ]; then
            CHECKPOINTS="$CHECKPOINTS way1_bf_seq=$ckpt_file"
        fi
    fi
    
    if [ -z "$CHECKPOINTS" ]; then
        echo "Error: No checkpoints found in ${OUTPUT_DIR}"
        echo "Please run training first with: ./run_ablation.sh --train"
        exit 1
    fi
    
    echo "Found checkpoints:$CHECKPOINTS"
    echo ""
    
    python tsne_analysis.py \
        --checkpoints $CHECKPOINTS \
        --output-dir "$TSNE_OUTPUT_DIR" \
        --n-classes 5 \
        --max-samples 500 \
        --perplexity 30
    
    echo ""
    echo "=============================================="
    echo "t-SNE analysis complete! Plots in: ${TSNE_OUTPUT_DIR}"
    echo "=============================================="
}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
case "${1:-}" in
    --sweep)
        run_sweeps
        ;;
    --train)
        run_direct_training
        ;;
    --tsne)
        run_tsne_analysis
        ;;
    --all)
        run_direct_training
        run_tsne_analysis
        ;;
    *)
        echo "SOARA Ablation Study Runner"
        echo ""
        echo "Usage: $0 [--sweep | --train | --tsne | --all]"
        echo ""
        echo "Options:"
        echo "  --sweep   Create wandb sweeps for all configurations"
        echo "  --train   Run direct training for all configurations"  
        echo "  --tsne    Run t-SNE analysis on existing checkpoints"
        echo "  --all     Run training and t-SNE analysis"
        echo ""
        echo "Example workflow:"
        echo "  1. Run sweeps: $0 --sweep"
        echo "  2. Start agents: wandb agent <username/project/sweep_id>"
        echo "  3. After training, run t-SNE: $0 --tsne"
        echo ""
        echo "Alternative (direct training):"
        echo "  $0 --all"
        ;;
esac
