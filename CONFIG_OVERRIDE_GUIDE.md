# Config Override Guide

This guide explains how to override any configuration field from the command line when running your training script.

## Overview

The config override system allows you to modify any configuration value without editing the YAML config file. This is particularly useful for:

- Hyperparameter tuning
- Quick experiments
- Running different model configurations
- Adjusting training parameters for different environments

## Basic Usage

### Command Line Syntax

```bash
python src/train.py --override key.subkey=value
```

### Multiple Overrides

You can specify multiple overrides in a single command:

```bash
python src/train.py --override training.learning_rate=0.001 --override model.name=CNN1D --override training.batch_size=64
```

## Configuration Structure

Your config file has the following main sections:

```yaml
data:           # Data loading and preprocessing settings
preprocessing:  # Signal processing parameters
model:          # Model architecture settings
training:       # Training hyperparameters
cross_validation: # Cross-validation settings
logging:        # Logging and checkpointing
evaluation:     # Evaluation metrics and settings
hardware:       # Hardware acceleration settings
```

## Override Examples

### 1. Training Parameters

```bash
# Change learning rate
python src/train.py --override training.learning_rate=0.001

# Modify batch size and epochs
python src/train.py --override training.batch_size=64 --override training.max_epochs=100

# Change optimizer settings
python src/train.py --override training.optimizer=adam --override training.weight_decay=1e-5

# Adjust early stopping patience
python src/train.py --override training.patience=20
```

### 2. Model Configuration

```bash
# Switch model architecture
python src/train.py --override model.name=CNN1D

# Adjust model parameters
python src/train.py --override model.hidden_size=256 --override model.num_layers=4

# Change dropout rate
python src/train.py --override model.dropout=0.3
```

### 3. Data and Preprocessing

```bash
# Change normalization method
python src/train.py --override preprocessing.normalization=min_max

# Adjust filter parameters
python src/train.py --override preprocessing.bandpass_filter.low_freq=0.1 --override preprocessing.bandpass_filter.high_freq=1.5

# Modify downsampling rate
python src/train.py --override preprocessing.downsample.target_rate=32

# Change data split ratio
python src/train.py --override training.val_split=0.15
```

### 4. Hardware Settings

```bash
# Force CPU usage
python src/train.py --override hardware.accelerator=cpu

# Use specific GPU
python src/train.py --override hardware.accelerator=gpu --override hardware.devices=1

# Change precision
python src/train.py --override hardware.precision=16
```

### 5. Logging and Evaluation

```bash
# Change experiment name
python src/train.py --override logging.experiment_name=my_experiment

# Adjust number of saved checkpoints
python src/train.py --override logging.save_top_k=5

# Change evaluation metrics
python src/train.py --override evaluation.metrics=[mse,mae,pearson_correlation]

# Modify plot samples
python src/train.py --override evaluation.plot_samples=10
```

## Data Type Conversion

The system automatically converts string values to appropriate Python types:

| Input String | Converted Type | Example |
|--------------|----------------|---------|
| `123` | Integer | `training.max_epochs=50` |
| `0.001` | Float | `training.learning_rate=0.001` |
| `true`/`false` | Boolean | `some_flag=true` |
| `none`/`null` | None | `optional_param=none` |
| `[a,b,c]` | List | `evaluation.metrics=[mse,mae]` |
| `text` | String | `model.name=CNN1D` |

## Advanced Usage

### Nested Configuration Overrides

You can override deeply nested configuration values:

```bash
# Override nested filter parameters
python src/train.py --override preprocessing.bandpass_filter.order=4

# Modify nested hardware settings
python src/train.py --override hardware.accelerator=gpu
```

### Creating New Configuration Keys

You can even create new configuration keys that don't exist in the original config:

```bash
# Add new parameters
python src/train.py --override custom.new_param=value --override debug.verbose=true
```

### Print Configuration Only

To see the final configuration without running training:

```bash
python src/train.py --print-config --override training.learning_rate=0.001
```

## Common Use Cases

### 1. Quick Hyperparameter Sweep

```bash
# Test different learning rates
python src/train.py --override training.learning_rate=0.01 --override logging.experiment_name=lr_001
python src/train.py --override training.learning_rate=0.001 --override logging.experiment_name=lr_0001
python src/train.py --override training.learning_rate=0.0001 --override logging.experiment_name=lr_00001
```

### 2. Model Architecture Comparison

```bash
# Test different models
python src/train.py --override model.name=CNN1D --override logging.experiment_name=cnn1d_test
python src/train.py --override model.name=LSTM --override logging.experiment_name=lstm_test
python src/train.py --override model.name=CNN_LSTM --override logging.experiment_name=cnn_lstm_test
```

### 3. Quick Debug Run

```bash
# Fast debug run with minimal epochs
python src/train.py --override training.max_epochs=1 --override training.batch_size=32 --override logging.experiment_name=debug_run
```

### 4. Production Run with Optimized Settings

```bash
# Full training run with optimized parameters
python src/train.py \
  --override training.max_epochs=200 \
  --override training.learning_rate=0.0005 \
  --override training.batch_size=256 \
  --override model.hidden_size=256 \
  --override model.dropout=0.1 \
  --override logging.experiment_name=production_run_v1
```

## Error Handling

The system gracefully handles errors:

- **Invalid format**: Warns and skips malformed overrides
- **Non-existent paths**: Creates new configuration keys
- **Type conflicts**: Warns and skips overrides that conflict with existing structure

Example error cases:
```bash
# Invalid format (missing =)
python src/train.py --override invalid_format
# Warning: Failed to apply override 'invalid_format': Invalid override format

# Trying to override a non-dict value
python src/train.py --override training.learning_rate.invalid=value
# Warning: Failed to apply override: learning_rate is not a dictionary
```

## Tips and Best Practices

1. **Use descriptive experiment names** when overriding parameters:
   ```bash
   --override logging.experiment_name=lr_001_bs_128_dropout_03
   ```

2. **Test with `--print-config`** first to verify your overrides:
   ```bash
   python src/train.py --print-config --override your.parameter=value
   ```

3. **Group related overrides** for readability:
   ```bash
   # Model-related overrides
   python src/train.py \
     --override model.name=CNN1D \
     --override model.hidden_size=256 \
     --override model.dropout=0.3
   ```

4. **Use shell scripts** for complex parameter sweeps:
   ```bash
   #!/bin/bash
   for lr in 0.01 0.001 0.0001; do
     python src/train.py --override training.learning_rate=$lr --override logging.experiment_name=lr_$lr
   done
   ```

## Help and Documentation

To see all available command line options:

```bash
python src/train.py --help
```

This will show the help message including config override examples.
