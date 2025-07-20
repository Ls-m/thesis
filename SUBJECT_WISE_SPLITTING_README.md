# Subject-wise Data Splitting Implementation

This document explains the subject-wise data splitting implementation for PPG to respiratory waveform estimation, which ensures proper separation of subjects across training, validation, and test sets to prevent data leakage.

## 🎯 Problem Statement

The original implementation had a potential data leakage issue where segments from the same subject could appear in both training and validation sets. This happens because:

1. **Segment-based splitting**: The original code segments each subject's data into overlapping windows
2. **Random validation split**: It then randomly splits these segments into train/validation sets
3. **Data leakage**: Segments from the same subject can end up in both training and validation sets

This leads to overly optimistic validation performance since the model sees similar patterns from the same subject during both training and validation.

## ✅ Solution: Subject-wise Splitting

The new implementation ensures that **all data from a given subject goes entirely into either the training, validation, or test set, never split across multiple sets**.

### Key Features

- **True subject independence**: No subject appears in multiple sets
- **Configurable validation split**: Control what fraction of training subjects to use for validation
- **Reproducible splits**: Uses random seeds for consistent results
- **Backward compatibility**: Maintains compatibility with existing code
- **Comprehensive validation**: Built-in checks to ensure no data leakage

## 📁 New Files

### 1. `src/subject_wise_data_utils.py`
Enhanced data preprocessing utilities with proper subject-wise splitting:

- `SubjectWiseDataPreprocessor`: Enhanced preprocessor class
- `create_subject_wise_splits()`: Creates subject-wise train/val/test splits
- `prepare_subject_wise_fold_data()`: Prepares data for training with subject separation
- `print_split_summary()`: Displays detailed split information
- Backward compatibility functions for existing code

### 2. `src/train_subject_wise.py`
Updated training script that uses subject-wise splitting:

- Full leave-one-out cross-validation with subject-wise validation splits
- Single fold training with specified test subject
- Configurable validation split ratio
- Comprehensive logging and result tracking
- Command-line interface with extensive options

### 3. `src/demo_subject_wise_splitting.py`
Demonstration script that shows how the splitting works:

- Visual demonstration of subject-wise vs random splitting
- Data leakage validation
- Cross-validation examples
- Comparison between splitting methods

## 🚀 Usage

### Basic Usage

```bash
# Full cross-validation with subject-wise splitting
python src/train_subject_wise.py --config configs/config.yaml

# Single fold with specific test subject
python src/train_subject_wise.py --config configs/config.yaml --fold subject_01

# Use different dataset
python src/train_subject_wise.py --config configs/config.yaml --dataset bidmc

# Custom validation split (30% of training subjects for validation)
python src/train_subject_wise.py --config configs/config.yaml --val-split 0.3
```

### Advanced Usage

```bash
# List available subjects
python src/train_subject_wise.py --list-subjects --dataset bidmc

# Override configuration parameters
python src/train_subject_wise.py --override training.learning_rate=0.001 --override model.name=RWKV

# Custom random seed for reproducibility
python src/train_subject_wise.py --seed 123

# Print configuration without training
python src/train_subject_wise.py --print-config
```

### Demonstration

```bash
# Run the demonstration script
python src/demo_subject_wise_splitting.py
```

## 📊 How It Works

### 1. Data Loading and Preprocessing
```python
# Load and preprocess all subjects
preprocessor = SubjectWiseDataPreprocessor(config)
processed_data = preprocessor.prepare_dataset('src/bidmc')
# Result: {'subject_00': (ppg_segments, resp_segments), 'subject_01': ...}
```

### 2. Subject-wise Split Creation
```python
# Create splits ensuring subject separation
splits = create_subject_wise_splits(
    processed_data, 
    test_subject='subject_01',  # or None for full CV
    val_split=0.2,              # 20% of training subjects for validation
    random_seed=42              # for reproducibility
)
```

### 3. Fold Data Preparation
```python
# Prepare data for training with proper subject separation
fold_data = prepare_subject_wise_fold_data(
    processed_data,
    train_subjects=['subject_02', 'subject_03', ...],
    val_subjects=['subject_04'],
    test_subject='subject_01'
)
```

## 🔍 Validation and Verification

The implementation includes comprehensive validation to ensure no data leakage:

### Automatic Checks
- **Subject overlap detection**: Ensures no subject appears in multiple sets
- **Data shape validation**: Verifies consistent data dimensions
- **NaN value detection**: Identifies potential data quality issues
- **Split summary reporting**: Provides detailed information about each split

### Manual Verification
Run the demonstration script to see the splitting in action:
```bash
python src/demo_subject_wise_splitting.py
```

This will show:
- Subject assignments for each set
- Data leakage validation results
- Comparison with random splitting
- Visual confirmation of proper separation

## 📈 Benefits

### 1. **No Data Leakage**
- Subjects are completely separated across sets
- More realistic performance evaluation
- Better assessment of generalization capability

### 2. **Proper Cross-validation**
- True leave-one-subject-out evaluation
- Subject-independent model assessment
- More reliable performance metrics

### 3. **Reproducible Results**
- Consistent splits across runs
- Configurable random seeds
- Deterministic subject assignments

### 4. **Flexible Configuration**
- Adjustable validation split ratios
- Support for both datasets (csv/ and bidmc/)
- Compatible with existing model configurations

## 🔄 Comparison: Before vs After

### Before (Original Implementation)
```
Subjects: [A, B, C, D, E]
Test: E
Training data: Combine segments from [A, B, C, D]
Random split: 
  - Train: Random segments from [A, B, C, D]
  - Val: Random segments from [A, B, C, D]
❌ Problem: Segments from same subject in both train and val
```

### After (Subject-wise Implementation)
```
Subjects: [A, B, C, D, E]
Test: E
Subject split:
  - Train subjects: [A, B, C]
  - Val subjects: [D]
  - Train: All segments from [A, B, C]
  - Val: All segments from [D]
✅ Solution: Complete subject separation
```

## 🛠️ Configuration

The subject-wise splitting uses the same configuration file format. Key parameters:

```yaml
# Data Configuration
data:
  csv_folder: "src/bidmc"  # or "src/csv"
  input_column: "PPG"
  target_column: "NASAL CANULA"
  # ... other parameters remain the same

# Training Configuration
training:
  val_split: 0.2  # Used for subject-wise validation split
  # ... other parameters remain the same
```

## 🔧 Integration with Existing Code

The implementation maintains backward compatibility:

### Option 1: Use New Subject-wise Training Script
```bash
python src/train_subject_wise.py --config configs/config.yaml
```

### Option 2: Update Existing Code
```python
# Replace imports
from subject_wise_data_utils import (
    SubjectWiseDataPreprocessor,
    create_subject_wise_splits,
    prepare_subject_wise_fold_data
)

# Use new functions
preprocessor = SubjectWiseDataPreprocessor(config)
splits = create_subject_wise_splits(processed_data, val_split=0.2)
```

## 📋 Results and Outputs

### Training Results
Results are saved with clear identification:
- `subject_wise_single_fold_{subject}_results.pkl`: Single fold results
- `subject_wise_complete_cv_results.pkl`: Full cross-validation results

### Result Structure
```python
{
    'fold_results': [...],           # Individual fold results
    'splitting_method': 'subject_wise',
    'val_split': 0.2,
    'random_seed': 42,
    'mode': 'single_fold' or 'full_cv',
    # ... other metadata
}
```

### Enhanced Fold Results
Each fold result now includes:
```python
{
    'fold_id': 0,
    'test_subject': 'subject_01',
    'train_subjects': ['subject_02', 'subject_03', ...],
    'val_subjects': ['subject_04'],
    'test_metrics': {...},
    'predictions': [...],
    'targets': [...],
    # ... other results
}
```

## 🧪 Testing and Validation

### Run Demonstration
```bash
python src/demo_subject_wise_splitting.py
```

### Validate Specific Dataset
```bash
# List subjects in BIDMC dataset
python src/train_subject_wise.py --dataset bidmc --list-subjects

# List subjects in CSV dataset
python src/train_subject_wise.py --dataset csv --list-subjects
```

### Test Single Fold
```bash
# Quick test with single fold
python src/train_subject_wise.py --config configs/config.yaml --fold subject_00 --dataset bidmc
```

## 🚨 Important Notes

1. **Minimum Subjects**: Need at least 3 subjects for proper train/val/test split
2. **Validation Split**: With few subjects, validation set might be small
3. **Random Seeds**: Use consistent seeds for reproducible results
4. **Dataset Compatibility**: Works with both `src/csv/` and `src/bidmc/` datasets
5. **Memory Usage**: Similar to original implementation
6. **Performance**: Slightly more overhead for subject tracking, but negligible

## 🔮 Future Enhancements

Potential improvements for future versions:

1. **Stratified Splitting**: Balance subjects by metadata (age, gender, etc.)
2. **Group-wise Splitting**: Support for multiple sessions per subject
3. **Custom Split Ratios**: More flexible train/val/test ratios
4. **Cross-dataset Evaluation**: Train on one dataset, test on another
5. **Ensemble Methods**: Combine predictions across subject-wise folds

## 📞 Support

If you encounter issues or have questions:

1. Check the demonstration script output
2. Verify subject availability with `--list-subjects`
3. Ensure minimum subject requirements are met
4. Check configuration file format
5. Review error messages for specific guidance

The subject-wise splitting implementation provides a robust foundation for proper subject-independent evaluation of PPG to respiratory waveform estimation models.
