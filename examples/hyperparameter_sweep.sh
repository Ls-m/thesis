#!/bin/bash

# Hyperparameter Sweep Example Script
# This script demonstrates how to use config overrides for systematic hyperparameter tuning

echo "Starting hyperparameter sweep..."

# Create results directory for this sweep
SWEEP_DIR="results/hyperparameter_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

# Learning rate sweep
echo "=== Learning Rate Sweep ==="
for lr in 0.01 0.001 0.0001 0.00001; do
    echo "Testing learning rate: $lr"
    python src/train.py \
        --override training.learning_rate=$lr \
        --override training.max_epochs=5 \
        --override logging.experiment_name=lr_sweep_$lr \
        --override logging.log_dir="$SWEEP_DIR/lr_$lr" \
        > "$SWEEP_DIR/lr_${lr}_output.log" 2>&1 &
    
    # Limit concurrent jobs to avoid resource exhaustion
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n  # Wait for any job to complete
    fi
done

# Wait for all learning rate jobs to complete
wait

echo "=== Batch Size Sweep ==="
for bs in 32 64 128 256; do
    echo "Testing batch size: $bs"
    python src/train.py \
        --override training.batch_size=$bs \
        --override training.max_epochs=5 \
        --override logging.experiment_name=bs_sweep_$bs \
        --override logging.log_dir="$SWEEP_DIR/bs_$bs" \
        > "$SWEEP_DIR/bs_${bs}_output.log" 2>&1 &
    
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n
    fi
done

wait

echo "=== Model Architecture Sweep ==="
for model in CNN1D LSTM CNN_LSTM; do
    echo "Testing model: $model"
    python src/train.py \
        --override model.name=$model \
        --override training.max_epochs=5 \
        --override logging.experiment_name=model_sweep_$model \
        --override logging.log_dir="$SWEEP_DIR/model_$model" \
        > "$SWEEP_DIR/model_${model}_output.log" 2>&1 &
    
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n
    fi
done

wait

echo "=== Dropout Rate Sweep ==="
for dropout in 0.1 0.2 0.3 0.4 0.5; do
    echo "Testing dropout: $dropout"
    python src/train.py \
        --override model.dropout=$dropout \
        --override training.max_epochs=5 \
        --override logging.experiment_name=dropout_sweep_$dropout \
        --override logging.log_dir="$SWEEP_DIR/dropout_$dropout" \
        > "$SWEEP_DIR/dropout_${dropout}_output.log" 2>&1 &
    
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n
    fi
done

wait

echo "=== Combined Best Parameters Test ==="
# Test combination of potentially good parameters
python src/train.py \
    --override training.learning_rate=0.001 \
    --override training.batch_size=128 \
    --override model.name=CNN_LSTM \
    --override model.dropout=0.2 \
    --override training.max_epochs=10 \
    --override logging.experiment_name=combined_best \
    --override logging.log_dir="$SWEEP_DIR/combined_best" \
    > "$SWEEP_DIR/combined_best_output.log" 2>&1

echo "Hyperparameter sweep completed!"
echo "Results saved in: $SWEEP_DIR"
echo ""
echo "To analyze results, check the log files in each subdirectory:"
echo "  - Training logs: $SWEEP_DIR/*/logs/"
echo "  - Output logs: $SWEEP_DIR/*_output.log"
echo ""
echo "You can also use TensorBoard to visualize the results:"
echo "  tensorboard --logdir=$SWEEP_DIR"
