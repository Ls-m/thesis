#!/usr/bin/env python3
"""
Quick example script to test the PPG to respiratory waveform estimation pipeline.
This script runs a minimal version with reduced epochs for testing purposes.
"""

import os
import yaml
import sys

def create_test_config():
    """Create a test configuration with reduced parameters for quick testing."""
    
    test_config = {
        'data': {
            'csv_folder': 'csv',
            'input_column': 'PPG',
            'target_column': 'NASAL CANULA',
            'sampling_rate': 256,
            'segment_length': 1024,  # Reduced for faster processing
            'overlap': 0.5
        },
        'preprocessing': {
            'bandpass_filter': {
                'low_freq': 0.1,
                'high_freq': 2.0,
                'order': 4
            },
            'downsample': {
                'target_rate': 64
            },
            'normalization': 'z_score'
        },
        'model': {
            'name': 'CNN1D',
            'input_size': 256,  # After downsampling: 1024/4 = 256
            'hidden_size': 64,  # Reduced for faster training
            'num_layers': 2,    # Reduced for faster training
            'dropout': 0.2
        },
        'training': {
            'batch_size': 16,   # Reduced for faster training
            'learning_rate': 0.001,
            'max_epochs': 5,    # Very few epochs for testing
            'patience': 3,
            'val_split': 0.2,
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'weight_decay': 1e-4  # Add weight decay parameter
        },
        'cross_validation': {
            'method': 'leave_one_out'
        },
        'logging': {
            'log_dir': 'logs',
            'experiment_name': 'test_run',
            'save_top_k': 1
        },
        'evaluation': {
            'metrics': ['pearson_correlation', 'mse', 'mae'],
            'plot_samples': 2
        },
        'hardware': {
            'accelerator': 'auto',
            'devices': 1,
            'precision': 32
        }
    }
    
    return test_config

def main():
    print("=" * 60)
    print("PPG TO RESPIRATORY WAVEFORM ESTIMATION - QUICK TEST")
    print("=" * 60)
    print()
    
    # Check if CSV data exists
    if not os.path.exists('csv'):
        print("ERROR: CSV folder not found!")
        print("Please ensure your dataset is in the 'csv' folder.")
        return
    
    csv_files = [f for f in os.listdir('csv') if f.endswith('.csv')]
    if len(csv_files) == 0:
        print("ERROR: No CSV files found in the csv folder!")
        return
    
    print(f"Found {len(csv_files)} CSV files in the dataset.")
    print("Sample files:", csv_files[:5])
    print()
    
    # Create test configuration
    test_config = create_test_config()
    
    # Save test configuration
    os.makedirs('configs', exist_ok=True)
    test_config_path = 'configs/test_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    print(f"Created test configuration: {test_config_path}")
    print()
    
    # Test data preprocessing
    print("Testing data preprocessing...")
    try:
        sys.path.append('src')
        from data_utils import DataPreprocessor
        
        preprocessor = DataPreprocessor(test_config)
        
        # Test loading a single file
        import pandas as pd
        test_file = os.path.join('csv', csv_files[0])
        
        # Try reading as tab-separated first, then comma-separated
        try:
            df = pd.read_csv(test_file, sep='\t')
        except:
            df = pd.read_csv(test_file)
            
        df = df.iloc[1:].reset_index(drop=True)  # Skip header row
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Successfully loaded test file: {csv_files[0]}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        if test_config['data']['input_column'] not in df.columns:
            print(f"ERROR: Input column '{test_config['data']['input_column']}' not found!")
            return
        
        if test_config['data']['target_column'] not in df.columns:
            print(f"ERROR: Target column '{test_config['data']['target_column']}' not found!")
            return
        
        print("✓ Required columns found")
        print()
        
    except Exception as e:
        print(f"ERROR in data preprocessing test: {e}")
        return
    
    # Test model import
    print("Testing model imports...")
    try:
        sys.path.append('.')
        from models import get_model
        
        model = get_model(
            model_name=test_config['model']['name'],
            input_size=test_config['model']['input_size'],
            hidden_size=test_config['model']['hidden_size'],
            num_layers=test_config['model']['num_layers'],
            dropout=test_config['model']['dropout']
        )
        
        print(f"✓ Successfully created {test_config['model']['name']} model")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
    except Exception as e:
        print(f"ERROR in model import test: {e}")
        return
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("To run the full training pipeline:")
    print("1. For quick test (few epochs):")
    print("   cd src && python train.py --config ../configs/test_config.yaml")
    print()
    print("2. For full training:")
    print("   cd src && python train.py --config ../configs/config.yaml")
    print()
    print("3. To evaluate results:")
    print("   cd src && python test.py --results ../results/complete_cv_results.pkl")
    print()
    print("4. To monitor training:")
    print("   tensorboard --logdir logs/")
    print()

if __name__ == "__main__":
    main()
