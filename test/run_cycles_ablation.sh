#!/bin/bash
# Run num_cycles ablation study on fgvc_aircraft
# Testing convergence with varying cycles for way1 method

# Fixed hyperparameters
RANK=16
LR=0.001
WEIGHT_DECAY=0.001
ORTHO_WEIGHT=0.04
EPOCHS=20
BATCH_SIZE=32
METHOD="way1"
DATASET="fgvc_aircraft"

# Cycles to test
CYCLES=(1 3 5 7)

OUTPUT_BASE="/home/chandan/test_current/outputs/cycles_ablation"
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "  NUM_CYCLES ABLATION STUDY"
echo "  Dataset: $DATASET"
echo "  Method: $METHOD"
echo "  Rank: $RANK"
echo "  Cycles: ${CYCLES[@]}"
echo "  Epochs: $EPOCHS"
echo "  LR=$LR, WD=$WEIGHT_DECAY, Ortho=$ORTHO_WEIGHT"
echo "=============================================="

for cycles in "${CYCLES[@]}"; do
    exp_name="${DATASET}_${METHOD}_cycles${cycles}"
    output_dir="${OUTPUT_BASE}/${exp_name}"
    log_file="${OUTPUT_BASE}/train_${exp_name}.log"
    
    echo ""
    echo ">>> Training: $exp_name"
    echo "    Cycles: $cycles"
    echo "    Output: $output_dir"
    echo "    Log: $log_file"
    
    # Skip if already completed
    if [ -f "${output_dir}/${METHOD}_best_model.pth" ]; then
        echo "    Already completed, skipping..."
        continue
    fi
    
    echo ""
    echo "    === Configuration ==="
    echo "    Total Cycles: $cycles"
    echo "    Method: $METHOD"
    echo "    Rank: $RANK"
    echo "    Epochs: $EPOCHS"
    
    # Calculate num_steps_per_phase based on epochs and cycles
    # Assuming fgvc_aircraft dataset - we'll let the script calculate it
    # The script will print the actual num_steps_per_phase used
    
    python train_vit_rotational.py \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --rank "$RANK" \
        --total-cycles "$cycles" \
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

echo ""
echo "=============================================="
echo "  ALL TRAINING COMPLETE!"
echo "=============================================="
echo "Logs saved to: $OUTPUT_BASE"
