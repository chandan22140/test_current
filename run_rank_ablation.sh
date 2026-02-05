#!/bin/bash
# Run rank ablation study across 5 datasets
# Fixed hyperparameters for fair comparison

# Fixed hyperparameters
LR=0.001
WEIGHT_DECAY=0.001
ORTHO_WEIGHT=0.04
EPOCHS=10
BATCH_SIZE=32
METHOD="way0"

# Datasets and ranks
DATASETS=("fgvc_aircraft")
RANKS=(2 4 8 16)

OUTPUT_BASE="~/test_current/outputs/rank_ablation"
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "  RANK ABLATION STUDY"
echo "  Datasets: ${DATASETS[@]}"
echo "  Ranks: ${RANKS[@]}"
echo "  LR=$LR, WD=$WEIGHT_DECAY, Ortho=$ORTHO_WEIGHT"
echo "=============================================="

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========== Dataset: $dataset =========="
    
    for rank in "${RANKS[@]}"; do
        exp_name="${dataset}_r${rank}"
        output_dir="${OUTPUT_BASE}/${exp_name}"
        log_file="${OUTPUT_BASE}/train_${exp_name}.log"
        
        echo ""
        echo ">>> Training: $exp_name"
        echo "    Output: $output_dir"
        echo "    Log: $log_file"
        
        # Skip if already completed
        if [ -f "${output_dir}/way0_best_model.pth" ]; then
            echo "    Already completed, skipping..."
            continue
        fi
        
        python train_vit_rotational.py \
            --dataset "$dataset" \
            --method "$METHOD" \
            --rank "$rank" \
            --learning-rate "$LR" \
            --weight-decay "$WEIGHT_DECAY" \
            --orthogonality-weight "$ORTHO_WEIGHT" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$output_dir" \
            --experiment-name "$exp_name" \
            2>&1 | tee "$log_file"
        
        echo "    Completed: $exp_name"
    done
done

echo ""
echo "=============================================="
echo "  ALL TRAINING COMPLETE!"
echo "=============================================="
echo "Logs saved to: $OUTPUT_BASE"
