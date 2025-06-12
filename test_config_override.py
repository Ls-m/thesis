#!/usr/bin/env python3
"""
Test script to demonstrate config override functionality.
This script shows how the config override system works without running the full training.
"""

import sys
import os
sys.path.append('src')

from train import load_config, apply_config_overrides
import yaml

def test_config_overrides():
    """Test various config override scenarios."""
    
    print("=" * 60)
    print("CONFIG OVERRIDE FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Load base config
    config = load_config('configs/config.yaml')
    
    print("\n1. ORIGINAL CONFIG (selected values):")
    print(f"   training.learning_rate: {config['training']['learning_rate']}")
    print(f"   training.batch_size: {config['training']['batch_size']}")
    print(f"   model.name: {config['model']['name']}")
    print(f"   model.dropout: {config['model']['dropout']}")
    print(f"   hardware.accelerator: {config['hardware']['accelerator']}")
    
    # Test different override scenarios
    test_cases = [
        {
            'name': 'Basic numeric overrides',
            'overrides': [
                'training.learning_rate=0.001',
                'training.batch_size=64',
                'model.dropout=0.3'
            ]
        },
        {
            'name': 'String and boolean overrides',
            'overrides': [
                'model.name=CNN1D',
                'preprocessing.normalization=min_max',
                'hardware.accelerator=cpu'
            ]
        },
        {
            'name': 'Deep nested overrides',
            'overrides': [
                'preprocessing.bandpass_filter.low_freq=0.1',
                'preprocessing.bandpass_filter.high_freq=1.5',
                'preprocessing.downsample.target_rate=32'
            ]
        },
        {
            'name': 'Mixed type overrides',
            'overrides': [
                'training.max_epochs=100',
                'training.patience=20',
                'evaluation.plot_samples=10',
                'logging.save_top_k=5'
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 2):
        print(f"\n{i}. {test_case['name'].upper()}:")
        print(f"   Overrides: {test_case['overrides']}")
        
        # Apply overrides to a copy of the config
        test_config = load_config('configs/config.yaml')  # Fresh copy
        test_config = apply_config_overrides(test_config, test_case['overrides'])
        
        print("   Results:")
        for override in test_case['overrides']:
            key_path = override.split('=')[0]
            keys = key_path.split('.')
            
            # Navigate to the value
            current = test_config
            for key in keys:
                current = current[key]
            
            print(f"     {key_path}: {current}")
    
    print(f"\n{len(test_cases) + 2}. TESTING ERROR HANDLING:")
    
    # Test error cases
    error_cases = [
        'invalid_format_no_equals',
        'nonexistent.key=value',
        'training.learning_rate.invalid=value'  # trying to override a non-dict
    ]
    
    for error_case in error_cases:
        print(f"   Testing: {error_case}")
        test_config = load_config('configs/config.yaml')
        try:
            apply_config_overrides(test_config, [error_case])
        except Exception as e:
            print(f"     Expected error handled: {type(e).__name__}")
    
    print(f"\n{len(test_cases) + 3}. TYPE CONVERSION EXAMPLES:")
    
    type_examples = [
        ('Integer', 'training.max_epochs=50'),
        ('Float', 'training.learning_rate=0.0005'),
        ('Boolean True', 'some_flag=true'),
        ('Boolean False', 'another_flag=false'),
        ('String', 'model.name=CustomModel'),
        ('List', 'evaluation.metrics=[mse,mae,correlation]')
    ]
    
    test_config = load_config('configs/config.yaml')
    for type_name, override in type_examples:
        print(f"   {type_name}: {override}")
        try:
            apply_config_overrides(test_config, [override])
        except Exception as e:
            print(f"     Error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_config_overrides()
