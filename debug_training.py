#!/usr/bin/env python3
"""
Debug script to help identify and fix NaN issues during training.
"""

import torch
import numpy as np
import yaml
from src.data_utils import DataPreprocessor
from src.dataset import create_data_loaders, PPGRespiratoryDataset
from src.lightning_module import PPGRespiratoryLightningModule
import matplotlib.pyplot as plt


def load_config(config_path: str = 'configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def check_data_quality(config):
    """Check the quality of preprocessed data."""
    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Prepare dataset
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    
    print(f"\nProcessed {len(processed_data)} subjects")
    
    # Check each subject's data
    for subject_id, (ppg_segments, resp_segments) in processed_data.items():
        print(f"\nSubject {subject_id}:")
        print(f"  PPG segments: {ppg_segments.shape}")
        print(f"  RESP segments: {resp_segments.shape}")
        
        # Check for NaN/Inf values
        ppg_nan = np.isnan(ppg_segments).sum()
        resp_nan = np.isnan(resp_segments).sum()
        ppg_inf = np.isinf(ppg_segments).sum()
        resp_inf = np.isinf(resp_segments).sum()
        
        print(f"  PPG NaN: {ppg_nan}, Inf: {ppg_inf}")
        print(f"  RESP NaN: {resp_nan}, Inf: {resp_inf}")
        
        # Check value ranges
        print(f"  PPG range: [{ppg_segments.min():.4f}, {ppg_segments.max():.4f}]")
        print(f"  RESP range: [{resp_segments.min():.4f}, {resp_segments.max():.4f}]")
        
        # Check for extreme values
        ppg_extreme = np.sum(np.abs(ppg_segments) > 10)
        resp_extreme = np.sum(np.abs(resp_segments) > 10)
        print(f"  Extreme values (>10): PPG={ppg_extreme}, RESP={resp_extreme}")
    
    return processed_data


def test_dataset_creation(processed_data, config):
    """Test dataset creation for NaN issues."""
    print("\n" + "=" * 60)
    print("DATASET CREATION TEST")
    print("=" * 60)
    
    # Use first subject for testing
    subject_id = list(processed_data.keys())[0]
    ppg_segments, resp_segments = processed_data[subject_id]
    
    print(f"Testing with subject: {subject_id}")
    print(f"Original shapes: PPG={ppg_segments.shape}, RESP={resp_segments.shape}")
    
    # Create dataset
    dataset = PPGRespiratoryDataset(ppg_segments, resp_segments)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        ppg, resp = dataset[i]
        print(f"Sample {i}: PPG shape={ppg.shape}, RESP shape={resp.shape}")
        print(f"  PPG NaN: {torch.isnan(ppg).sum().item()}, Inf: {torch.isinf(ppg).sum().item()}")
        print(f"  RESP NaN: {torch.isnan(resp).sum().item()}, Inf: {torch.isinf(resp).sum().item()}")
        print(f"  PPG range: [{ppg.min():.4f}, {ppg.max():.4f}]")
        print(f"  RESP range: [{resp.min():.4f}, {resp.max():.4f}]")


def test_model_forward_pass(config):
    """Test model forward pass for NaN issues."""
    print("\n" + "=" * 60)
    print("MODEL FORWARD PASS TEST")
    print("=" * 60)
    
    # Create model
    model = PPGRespiratoryLightningModule(config)
    model.eval()
    
    # Create dummy input
    batch_size = config['training']['batch_size']
    input_size = config['model']['input_size']
    
    # Test with normal input
    print("Testing with normal input...")
    dummy_input = torch.randn(batch_size, 1, input_size)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input NaN: {torch.isnan(dummy_input).sum().item()}")
        print(f"Output NaN: {torch.isnan(output).sum().item()}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with extreme input
    print("\nTesting with extreme input...")
    extreme_input = torch.ones(batch_size, 1, input_size) * 100
    
    with torch.no_grad():
        output = model(extreme_input)
        print(f"Extreme input range: [{extreme_input.min():.4f}, {extreme_input.max():.4f}]")
        print(f"Output NaN: {torch.isnan(output).sum().item()}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with zero input
    print("\nTesting with zero input...")
    zero_input = torch.zeros(batch_size, 1, input_size)
    
    with torch.no_grad():
        output = model(zero_input)
        print(f"Zero input test - Output NaN: {torch.isnan(output).sum().item()}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")


def test_training_step(config, processed_data):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("TRAINING STEP TEST")
    print("=" * 60)
    
    # Prepare a small dataset
    from src.data_utils import prepare_fold_data, create_cross_validation_splits
    
    cv_splits = create_cross_validation_splits(processed_data)
    first_split = cv_splits[0]
    
    fold_data = prepare_fold_data(
        processed_data,
        first_split['train_subjects'][:2],  # Use only 2 subjects for testing
        first_split['test_subject'],
        val_split=0.2
    )
    
    # Create data loaders with small batch size
    test_config = config.copy()
    test_config['training']['batch_size'] = 4
    
    data_loaders = create_data_loaders(fold_data, batch_size=4, num_workers=0)
    
    # Create model
    model = PPGRespiratoryLightningModule(test_config)
    model.train()
    
    # Test a few training steps
    train_loader = data_loaders['train']
    
    print(f"Testing with {len(train_loader)} batches")
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Test only first 5 batches
            break
            
        print(f"\nBatch {batch_idx}:")
        ppg, resp = batch
        print(f"  Input shapes: PPG={ppg.shape}, RESP={resp.shape}")
        print(f"  Input NaN: PPG={torch.isnan(ppg).sum().item()}, RESP={torch.isnan(resp).sum().item()}")
        print(f"  Input ranges: PPG=[{ppg.min():.4f}, {ppg.max():.4f}], RESP=[{resp.min():.4f}, {resp.max():.4f}]")
        
        # Forward pass
        with torch.no_grad():
            pred = model(ppg)
            print(f"  Prediction shape: {pred.shape}")
            print(f"  Prediction NaN: {torch.isnan(pred).sum().item()}")
            print(f"  Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
            
            # Calculate loss
            loss = torch.nn.MSELoss()(pred, resp)
            print(f"  Loss: {loss.item():.6f}, NaN: {torch.isnan(loss).item()}")


def plot_sample_data(processed_data, save_path='sample_data_plot.png'):
    """Plot sample data to visualize potential issues."""
    print("\n" + "=" * 60)
    print("PLOTTING SAMPLE DATA")
    print("=" * 60)
    
    # Get first subject
    subject_id = list(processed_data.keys())[0]
    ppg_segments, resp_segments = processed_data[subject_id]
    
    # Plot first few segments
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sample Data from Subject {subject_id}')
    
    for i in range(min(3, len(ppg_segments))):
        # PPG signal
        axes[0, i].plot(ppg_segments[i])
        axes[0, i].set_title(f'PPG Segment {i}')
        axes[0, i].set_ylabel('Amplitude')
        axes[0, i].grid(True)
        
        # Respiratory signal
        axes[1, i].plot(resp_segments[i])
        axes[1, i].set_title(f'Respiratory Segment {i}')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].set_xlabel('Sample')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample data plot saved to: {save_path}")
    plt.close()


def main():
    """Run all debug tests."""
    print("Starting comprehensive debug analysis...")
    
    # Load configuration
    config = load_config()
    
    # Check data quality
    processed_data = check_data_quality(config)
    
    if not processed_data:
        print("No processed data available. Exiting.")
        return
    
    # Test dataset creation
    test_dataset_creation(processed_data, config)
    
    # Test model forward pass
    test_model_forward_pass(config)
    
    # Test training step
    test_training_step(config, processed_data)
    
    # Plot sample data
    plot_sample_data(processed_data)
    
    print("\n" + "=" * 60)
    print("DEBUG ANALYSIS COMPLETE")
    print("=" * 60)
    print("If you're still getting NaN errors, check:")
    print("1. Input data ranges and normalization")
    print("2. Model weight initialization")
    print("3. Learning rate (try even smaller values)")
    print("4. Gradient clipping values")
    print("5. Batch size (try smaller batches)")


if __name__ == "__main__":
    main()
