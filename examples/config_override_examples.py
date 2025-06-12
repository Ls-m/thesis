#!/usr/bin/env python3
"""
Python examples for using config overrides programmatically.
This script shows how to run training with different configurations from Python code.
"""

import subprocess
import os
import sys
from datetime import datetime

def run_training_with_overrides(overrides, experiment_name=None, capture_output=False):
    """
    Run training with specified config overrides.
    
    Args:
        overrides: List of override strings in format 'key.subkey=value'
        experiment_name: Optional experiment name
        capture_output: Whether to capture and return output
    
    Returns:
        subprocess.CompletedProcess object
    """
    cmd = [sys.executable, 'src/train.py']
    
    # Add overrides
    for override in overrides:
        cmd.extend(['--override', override])
    
    # Add experiment name if provided
    if experiment_name:
        cmd.extend(['--override', f'logging.experiment_name={experiment_name}'])
    
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd)

def quick_test_run():
    """Run a quick test with minimal epochs."""
    print("=== Quick Test Run ===")
    
    overrides = [
        'training.max_epochs=1',
        'training.batch_size=32',
        'hardware.accelerator=cpu'
    ]
    
    result = run_training_with_overrides(
        overrides, 
        experiment_name='quick_test',
        capture_output=True
    )
    
    print(f"Exit code: {result.returncode}")
    if result.returncode != 0:
        print("Error output:")
        print(result.stderr)

def learning_rate_comparison():
    """Compare different learning rates."""
    print("=== Learning Rate Comparison ===")
    
    learning_rates = [0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        overrides = [
            f'training.learning_rate={lr}',
            'training.max_epochs=5',
            'training.batch_size=64'
        ]
        
        run_training_with_overrides(
            overrides,
            experiment_name=f'lr_comparison_{lr}'
        )

def model_architecture_comparison():
    """Compare different model architectures."""
    print("=== Model Architecture Comparison ===")
    
    models = ['CNN1D', 'LSTM', 'CNN_LSTM']
    
    for model in models:
        print(f"\nTesting model: {model}")
        
        overrides = [
            f'model.name={model}',
            'training.max_epochs=5',
            'training.batch_size=128'
        ]
        
        run_training_with_overrides(
            overrides,
            experiment_name=f'model_comparison_{model}'
        )

def preprocessing_comparison():
    """Compare different preprocessing settings."""
    print("=== Preprocessing Comparison ===")
    
    # Test different normalization methods
    normalizations = ['z_score', 'min_max', 'robust']
    
    for norm in normalizations:
        print(f"\nTesting normalization: {norm}")
        
        overrides = [
            f'preprocessing.normalization={norm}',
            'training.max_epochs=3',
            'training.batch_size=64'
        ]
        
        run_training_with_overrides(
            overrides,
            experiment_name=f'preprocessing_{norm}'
        )

def print_config_example():
    """Example of printing configuration without training."""
    print("=== Print Config Example ===")
    
    cmd = [
        sys.executable, 'src/train.py',
        '--print-config',
        '--override', 'training.learning_rate=0.001',
        '--override', 'model.name=CNN1D',
        '--override', 'training.batch_size=64'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Configuration output:")
    print(result.stdout)

def advanced_override_examples():
    """Show advanced override examples."""
    print("=== Advanced Override Examples ===")
    
    # Complex nested overrides
    overrides = [
        'preprocessing.bandpass_filter.low_freq=0.1',
        'preprocessing.bandpass_filter.high_freq=1.5',
        'preprocessing.bandpass_filter.order=4',
        'preprocessing.downsample.target_rate=32',
        'model.hidden_size=256',
        'model.num_layers=4',
        'model.dropout=0.3',
        'training.learning_rate=0.0005',
        'training.batch_size=256',
        'training.max_epochs=2',
        'training.weight_decay=1e-5',
        'hardware.precision=16',
        'evaluation.plot_samples=3'
    ]
    
    print("Testing complex nested overrides...")
    run_training_with_overrides(
        overrides,
        experiment_name='advanced_config_test'
    )

def create_custom_experiment():
    """Create a custom experiment with specific settings."""
    print("=== Custom Experiment ===")
    
    # Create timestamp for unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    overrides = [
        'model.name=CNN_LSTM',
        'model.hidden_size=128',
        'model.dropout=0.25',
        'training.learning_rate=0.0008',
        'training.batch_size=128',
        'training.max_epochs=10',
        'training.optimizer=adamw',
        'training.weight_decay=1e-4',
        'preprocessing.normalization=z_score',
        'hardware.accelerator=auto',
        f'logging.log_dir=logs/custom_experiment_{timestamp}'
    ]
    
    run_training_with_overrides(
        overrides,
        experiment_name=f'custom_experiment_{timestamp}'
    )

def main():
    """Main function to run examples."""
    print("Config Override Examples")
    print("=" * 50)
    
    examples = {
        '1': ('Print Config Example', print_config_example),
        '2': ('Quick Test Run', quick_test_run),
        '3': ('Learning Rate Comparison', learning_rate_comparison),
        '4': ('Model Architecture Comparison', model_architecture_comparison),
        '5': ('Preprocessing Comparison', preprocessing_comparison),
        '6': ('Advanced Override Examples', advanced_override_examples),
        '7': ('Custom Experiment', create_custom_experiment),
        'all': ('Run All Examples', None)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter your choice (or 'all' to run everything): ").strip()
    
    if choice == 'all':
        for key, (name, func) in examples.items():
            if func is not None:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")
                try:
                    func()
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"Error running {name}: {e}")
    elif choice in examples:
        name, func = examples[choice]
        if func is not None:
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
