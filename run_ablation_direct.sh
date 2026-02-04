#!/bin/bash
# ==============================================================================
# SOARA Ablation - Direct Training with Fixed Hyperparameters
# ==============================================================================
# Runs all ablation configurations with tuned hyperparameters.
#
# Usage: ./run_ablation_direct.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/soara_ablation"

mkdir -p "$OUTPUT_DIR"

# Fixed hyperparameters from sweep
LR=0.00090625
WEIGHT_DECAY=0.000018778
ORTHO_WEIGHT=0.0018789
EPOCHS=10
BATCH_SIZE=32
MODEL="google/vit-base-patch16-224"
DATASET="fgvc_aircraft"

echo "=============================================="
echo "SOARA Ablation Study - Direct Training"
echo "=============================================="
echo "Hyperparameters:"
echo "  LR: $LR"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Ortho Weight: $ORTHO_WEIGHT"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "=============================================="

# Way0 with different ranks
for rank in 2 4 8 16; do
    echo ""
    echo ">>> Training way0 rank=${rank}..."
    echo ""
    
    python train_vit_rotational.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --method way0 \
        --rank $rank \
        --learning-rate $LR \
        --weight-decay $WEIGHT_DECAY \
        --orthogonality-weight $ORTHO_WEIGHT \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir "${OUTPUT_DIR}" \
        --experiment-name "ablation_way0_r${rank}" \
        2>&1 | tee "${OUTPUT_DIR}/train_way0_r${rank}.log"
    
    echo ""
    echo ">>> Completed way0 rank=${rank}"
    echo ""
done

# Way1 butterfly sequential
echo ""
echo ">>> Training way1 butterfly sequential..."
echo ""

python train_vit_rotational.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
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
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "ablation_way1_bf_seq" \
    2>&1 | tee "${OUTPUT_DIR}/train_way1_bf_seq.log"

echo ""
echo "=============================================="
echo "All training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "To run t-SNE analysis:"
echo "  python tsne_analysis.py --checkpoint-dir ${OUTPUT_DIR} --output-dir ./tsne_plots"
echo "=============================================="
