# PPG to Respiratory Waveform Estimation

This project implements deep learning models for estimating respiratory waveforms from photoplethysmography (PPG) signals using PyTorch Lightning and TensorBoard for clean logging and monitoring.

## Project Structure

```
thesis/
├── configs/
│   └── config.yaml              # Configuration file
├── csv/                         # Dataset folder (29 CSV files)
│   ├── AL25.csv
│   ├── BL24.csv
│   └── ...
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── cnn1d.py                # 1D CNN models
│   ├── lstm.py                 # LSTM models
│   └── cnn_lstm.py             # Hybrid CNN-LSTM models
├── src/                        # Source code
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── dataset.py              # PyTorch dataset classes
│   ├── lightning_module.py     # PyTorch Lightning module
│   ├── train.py                # Training script
│   └── test.py                 # Evaluation script
├── logs/                       # TensorBoard logs
├── results/                    # Training and evaluation results
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **Multiple Model Architectures**: CNN1D, LSTM, CNN-LSTM hybrid models
- **Leave-One-Out Cross-Validation**: Robust evaluation using subject-wise splits
- **Advanced Preprocessing**: Bandpass filtering, downsampling, normalization, segmentation
- **PyTorch Lightning**: Clean, scalable training framework
- **TensorBoard Integration**: Comprehensive logging and monitoring
- **Early Stopping**: Validation-based early stopping to prevent overfitting
- **Ensemble Predictions**: Average predictions across all folds
- **Comprehensive Evaluation**: Pearson correlation, MSE, MAE metrics with visualizations

## Dataset

The dataset consists of 29 CSV files (one per subject) with the following columns:
- `PPG`: Photoplethysmography signal (input)
- `NASAL CANULA`: Respiratory signal (ground truth)
- Sampling frequency: 256 Hz (except for one file)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Data settings**: Input/output columns, sampling rates, segment parameters
- **Preprocessing**: Filter parameters, downsampling, normalization methods
- **Model architecture**: Model type, hidden size, layers, dropout
- **Training**: Batch size, learning rate, epochs, early stopping patience
- **Hardware**: GPU/CPU usage, precision settings

## Usage

### Training

#### Standard Training (with potential data leakage)
Run leave-one-out cross-validation training:

```bash
python src/train.py --config configs/config.yaml --seed 42
```

#### Subject-wise Training (recommended - no data leakage)
Run training with proper subject-wise splitting to prevent data leakage:

```bash
# Full cross-validation with subject-wise splitting
python src/train_subject_wise.py --config configs/config.yaml

# Single fold with specific test subject
python src/train_subject_wise.py --config configs/config.yaml --fold subject_01

# Use BIDMC dataset
python src/train_subject_wise.py --config configs/config.yaml --dataset bidmc

# Custom validation split (30% of training subjects for validation)
python src/train_subject_wise.py --config configs/config.yaml --val-split 0.3

# List available subjects
python src/train_subject_wise.py --list-subjects --dataset bidmc
```

**⚠️ Important**: The subject-wise training script (`train_subject_wise.py`) is recommended as it ensures proper separation of subjects across train/validation/test sets, preventing data leakage that can lead to overly optimistic results.

For detailed information about subject-wise splitting, see [SUBJECT_WISE_SPLITTING_README.md](SUBJECT_WISE_SPLITTING_README.md).

#### Demonstration
To see how subject-wise splitting works and validate that there's no data leakage:

```bash
python src/demo_subject_wise_splitting.py
```

Options:
- `--config`: Path to configuration file (default: `configs/config.yaml`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--override`: Override any config value from command line (can be used multiple times)
- `--print-config`: Print final configuration and exit without training

### Config Overrides

You can override any configuration parameter from the command line without editing the YAML file:

```bash
# Override single parameters
python src/train.py --override training.learning_rate=0.001

# Override multiple parameters
python src/train.py --override training.learning_rate=0.001 --override model.name=CNN1D --override training.batch_size=64

# Override nested parameters
python src/train.py --override preprocessing.bandpass_filter.low_freq=0.1 --override preprocessing.bandpass_filter.high_freq=1.5

# Print configuration without training
python src/train.py --print-config --override training.learning_rate=0.001
```

#### Common Override Examples

**Training Parameters:**
```bash
python src/train.py --override training.learning_rate=0.001 --override training.max_epochs=100
python src/train.py --override training.batch_size=256 --override training.optimizer=adam
```

**Model Configuration:**
```bash
python src/train.py --override model.name=CNN1D --override model.hidden_size=256
python src/train.py --override model.dropout=0.3 --override model.num_layers=4
```

**Hardware Settings:**
```bash
python src/train.py --override hardware.accelerator=cpu
python src/train.py --override hardware.precision=16 --override hardware.devices=2
```

**Quick Debug Run:**
```bash
python src/train.py --override training.max_epochs=1 --override training.batch_size=32 --override logging.experiment_name=debug_test
```

For detailed information about config overrides, see [CONFIG_OVERRIDE_GUIDE.md](CONFIG_OVERRIDE_GUIDE.md).

### Evaluation

Evaluate trained models and generate visualizations:

```bash
cd src
python test.py --config ../configs/config.yaml --results ../results/complete_cv_results.pkl
```

Options:
- `--config`: Path to configuration file
- `--results`: Path to cross-validation results file
- `--output_dir`: Output directory for evaluation results (default: `results/evaluation`)

## Model Architectures

### 1. CNN1D
- Encoder-decoder architecture with 1D convolutions
- Batch normalization and dropout for regularization
- Configurable number of layers and hidden size

### 2. ResidualCNN1D
- Residual connections for improved gradient flow
- Multiple residual blocks with skip connections

### 3. LSTM
- Bidirectional LSTM for temporal modeling
- Input/output projection layers
- Configurable hidden size and number of layers

### 4. BiLSTM
- Bidirectional LSTM with attention mechanism
- Self-attention for improved feature selection

### 5. CNN_LSTM
- Hybrid architecture combining CNN feature extraction with LSTM temporal modeling
- CNN layers extract local features, LSTM captures temporal dependencies

### 6. AttentionCNN_LSTM
- Multi-scale CNN feature extraction
- Self-attention mechanism for improved performance
- Residual connections between LSTM and attention outputs

## Data Preprocessing Pipeline

1. **Loading**: Read CSV files and extract PPG and respiratory signals
2. **Filtering**: Apply bandpass filter (0.1-2.0 Hz) to remove noise
3. **Downsampling**: Reduce sampling rate from 256 Hz to 64 Hz
4. **Normalization**: Z-score normalization for stable training
5. **Segmentation**: Create overlapping windows (8 seconds with 50% overlap)

## Cross-Validation Strategy

- **Leave-One-Out**: Each subject is used as test set once
- **Train/Validation Split**: 80/20 split of remaining subjects for training
- **Early Stopping**: Monitor validation correlation for early stopping
- **Ensemble**: Average predictions across all folds for final evaluation

## Metrics

- **Pearson Correlation**: Primary metric for signal similarity
- **Mean Squared Error (MSE)**: L2 loss between predictions and targets
- **Mean Absolute Error (MAE)**: L1 loss for robust evaluation
- **Root Mean Squared Error (RMSE)**: Square root of MSE

## Outputs

### Training Outputs
- Model checkpoints in `logs/checkpoints/`
- TensorBoard logs in `logs/`
- Individual fold results in `results/fold_*_results.pkl`
- Complete cross-validation results in `results/complete_cv_results.pkl`

### Evaluation Outputs
- Correlation scatter plot
- Sample prediction visualizations
- Per-subject correlation bar chart
- Comprehensive results summary
- Ensemble predictions and metrics

## Monitoring Training

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- Correlation metrics over time
- Learning rate schedules
- Model architecture graphs

## Customization

### Adding New Models

1. Create a new model file in `models/` directory
2. Implement your model class inheriting from `nn.Module`
3. Add the model to `models/__init__.py`
4. Update the configuration file to use your new model

### Modifying Preprocessing

Edit the `DataPreprocessor` class in `src/data_utils.py`:
- Add new filtering methods
- Implement different normalization techniques
- Modify segmentation strategies

### Custom Metrics

Add new metrics to the Lightning module in `src/lightning_module.py`:
- Implement metric calculation in validation/test steps
- Log metrics using `self.log()`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Slow Training**: Increase `num_workers` in data loaders
3. **Poor Convergence**: Adjust learning rate or model architecture
4. **Data Loading Errors**: Check CSV file format and column names

### Performance Tips

- Use GPU acceleration when available
- Enable mixed precision training (`precision: 16` in config)
- Increase batch size for better GPU utilization
- Use multiple workers for data loading

## Results Interpretation

- **Correlation > 0.7**: Excellent performance
- **Correlation 0.5-0.7**: Good performance
- **Correlation < 0.5**: Poor performance, consider model/data improvements

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ppg_respiratory_estimation,
  title={PPG to Respiratory Waveform Estimation using Deep Learning},
  author={Elham Farang},
  year={2025},
  howpublished={\url{https://github.com/Ls-m/thesis}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please contact [elham.fr80@gmail.com] or open an issue on GitHub.
