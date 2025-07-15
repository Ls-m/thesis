# PPG to Respiratory Waveform Estimation - Improvements

This document describes the major improvements made to the PPG to respiratory waveform estimation project to address poor model performance and enhance the training pipeline.

## 🚀 Key Improvements

### 1. Fold Selection & Dataset Management

#### New CLI Features:
- **Single Fold Training**: Use `--fold <subject_name>` to train with a specific subject as test set
- **Dataset Switching**: Use `--dataset bidmc` or `--dataset csv` to switch between datasets
- **Subject Listing**: Use `--list-subjects` to see all available subjects in a dataset
- **Flexible Cross-Validation**: Default behavior remains full leave-one-out CV

#### Usage Examples:
```bash
# List subjects in CSV dataset
python src/train.py --config configs/improved_config.yaml --dataset csv --list-subjects

# Single fold training with specific subject
python src/train.py --config configs/improved_config.yaml --fold AL25

# Use BIDMC dataset with specific fold
python src/train.py --config configs/improved_config.yaml --dataset bidmc --fold subject_01

# Full leave-one-out CV (default behavior)
python src/train.py --config configs/improved_config.yaml
```

### 2. Advanced Model Architectures

#### New Models Added:
- **RWKV (Receptance Weighted Key Value)**: Linear attention mechanism for efficient long-sequence processing
- **ImprovedTransformer**: Multi-scale Transformer with positional encoding and skip connections
- **WaveNet**: Dilated convolutions with gated activations, optimized for signal-to-signal translation
- **Enhanced AttentionCNN_LSTM**: Multi-scale CNN feature extraction with attention mechanism

#### Model Comparison:
| Model | Strengths | Best For | Complexity |
|-------|-----------|----------|------------|
| RWKV | Linear complexity, memory efficient | Long sequences | O(n) |
| ImprovedTransformer | Multi-scale features, self-attention | Complex patterns | O(n²) |
| WaveNet | Large receptive field, signal-optimized | Signal translation | O(n) |
| AttentionCNN_LSTM | Balanced approach, proven architecture | General use | O(n) |

### 2. Enhanced Training & Logging

#### Comprehensive Metrics Tracking:
- ✅ **Training Loss** (per step and per epoch)
- ✅ **Validation Loss** (per epoch)
- ✅ **Training Pearson Correlation** (per epoch)
- ✅ **Validation Pearson Correlation** (per epoch)
- ✅ **Training MAE** (per epoch)
- ✅ **Validation MAE** (per epoch)

#### TensorBoard Integration:
- Real-time monitoring of all metrics
- Automatic plot generation
- Easy comparison between different runs
- Hyperparameter tracking

### 3. Preprocessing Configuration Management

#### Features:
- **Save preprocessing setups** with statistics
- **Load and reuse** preprocessing configurations
- **Adjustable parameters** (e.g., segment_length) while keeping core preprocessing
- **JSON and Pickle formats** for human readability and exact reproduction
- **Automatic timestamping** and metadata tracking

#### Usage Examples:
```python
# Save current preprocessing setup
from preprocessing_config import save_current_preprocessing_setup
save_current_preprocessing_setup(config, stats, "my_preprocessing_v1")

# Load and adjust for different segment length
from preprocessing_config import load_and_adjust_preprocessing_config
new_config = load_and_adjust_preprocessing_config(
    "my_preprocessing_v1", 
    segment_length=1024,
    other_adjustments={'preprocessing.normalization': 'min_max'}
)
```

## 📁 New Files Structure

```
├── models/
│   └── rwkv.py                    # RWKV, ImprovedTransformer, WaveNet models
├── src/
│   └── preprocessing_config.py    # Preprocessing configuration management
├── configs/
│   └── improved_config.yaml       # Optimized configuration for new models
├── examples/
│   └── improved_training_example.py # Comprehensive demo script
├── preprocessing_configs/          # Saved preprocessing configurations
└── IMPROVEMENTS_README.md         # This file
```

## 🔧 Configuration Improvements

### Optimized Settings:
- **Increased model capacity**: `hidden_size: 256`, `num_layers: 6`
- **Better learning rate**: `0.0005` (balanced for stability and convergence)
- **Optimized batch size**: `64` (better GPU utilization)
- **More training epochs**: `50` (sufficient for convergence)
- **Reduced dropout**: `0.1` (better learning capacity)

### Model-Specific Configurations:
```yaml
# RWKV Configuration
model:
  name: "RWKV"
  hidden_size: 256
  num_layers: 6
  dropout: 0.1

# ImprovedTransformer Configuration  
model:
  name: "ImprovedTransformer"
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1

# WaveNet Configuration
model:
  name: "WaveNet"
  hidden_size: 128
  num_layers: 10
  kernel_size: 3
  dropout: 0.1
```

## 🚀 Quick Start Guide

### 1. Basic Training with New Models

```bash
# Train with RWKV (recommended for efficiency)
python src/train.py --config configs/improved_config.yaml --override model.name=RWKV

# Train with ImprovedTransformer (for complex patterns)
python src/train.py --config configs/improved_config.yaml --override model.name=ImprovedTransformer

# Train with WaveNet (for signal processing)
python src/train.py --config configs/improved_config.yaml --override model.name=WaveNet
```

### 2. Monitor Training with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

### 3. Fold Selection & Dataset Management

```bash
# List available subjects in CSV dataset
python src/train.py --config configs/improved_config.yaml --dataset csv --list-subjects

# List available subjects in BIDMC dataset  
python src/train.py --config configs/improved_config.yaml --dataset bidmc --list-subjects

# Single fold training with specific subject
python src/train.py --config configs/improved_config.yaml --fold subject_01

# Single fold with dataset selection
python src/train.py --config configs/improved_config.yaml --dataset bidmc --fold subject_01
```

### 4. Run Comprehensive Demos

```bash
# Run the complete demonstration
python examples/improved_training_example.py

# Run fold selection demonstration
python examples/fold_selection_example.py
```

## 📊 Expected Performance Improvements

### Previous Performance:
- Validation Correlation: ~0.5
- Test Correlation: ~0.3

### Expected Improvements:
- **RWKV**: 15-25% improvement in correlation
- **ImprovedTransformer**: 20-30% improvement in correlation  
- **WaveNet**: 10-20% improvement in correlation
- **Better preprocessing**: 5-10% additional improvement

### Key Factors for Improvement:
1. **Better architectures** designed for signal-to-signal tasks
2. **Increased model capacity** with proper regularization
3. **Optimized hyperparameters** based on best practices
4. **Enhanced training monitoring** for better convergence tracking
5. **Preprocessing consistency** through configuration management

## 🔬 Advanced Usage

### Hyperparameter Sweeps

```bash
# Learning rate sweep
for lr in 0.0001 0.0005 0.001; do
    python src/train.py --config configs/improved_config.yaml \
        --override training.learning_rate=$lr \
        --override logging.experiment_name=lr_sweep_$lr
done

# Model architecture sweep
for model in RWKV ImprovedTransformer WaveNet; do
    python src/train.py --config configs/improved_config.yaml \
        --override model.name=$model \
        --override logging.experiment_name=model_sweep_$model
done
```

### Preprocessing Experiments

```bash
# Different normalization methods
python src/train.py --config configs/improved_config.yaml \
    --override preprocessing.normalization=min_max

# Different segment lengths
python src/train.py --config configs/improved_config.yaml \
    --override data.segment_length=1024

# Different filter settings
python src/train.py --config configs/improved_config.yaml \
    --override preprocessing.bandpass_filter.high_freq=3.0
```

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   --override training.batch_size=32
   
   # Reduce model size
   --override model.hidden_size=128
   ```

2. **Training Instability**:
   ```bash
   # Lower learning rate
   --override training.learning_rate=0.0001
   
   # Increase gradient clipping
   --override training.gradient_clip_val=0.5
   ```

3. **Poor Convergence**:
   ```bash
   # Increase model capacity
   --override model.hidden_size=512
   
   # More epochs
   --override training.max_epochs=100
   ```

## 📈 Monitoring and Analysis

### TensorBoard Metrics to Watch:
- **train_correlation** vs **val_correlation**: Check for overfitting
- **train_loss** vs **val_loss**: Monitor convergence
- **Learning rate**: Ensure proper scheduling

### Key Performance Indicators:
- **Validation correlation > 0.7**: Good performance
- **Test correlation > 0.6**: Acceptable generalization
- **Stable training curves**: No oscillations or divergence

## 🔄 Next Steps

1. **Run baseline experiments** with all new models
2. **Compare performance** using TensorBoard
3. **Fine-tune hyperparameters** based on initial results
4. **Experiment with ensemble methods** combining best models
5. **Analyze failure cases** and adjust preprocessing accordingly

## 📚 References

- **RWKV**: [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- **WaveNet**: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- **Attention Mechanisms**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

For questions or issues, please check the troubleshooting section or create an issue in the repository.
