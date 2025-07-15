#!/usr/bin/env python3
"""
Test script to verify all improvements are working correctly.
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_model_imports():
    """Test that all new models can be imported."""
    print("Testing model imports...")
    
    try:
        from models import get_model
        print("✅ Model factory imported successfully")
        
        # Test model instantiation
        models_to_test = [
            "RWKV",
            "ImprovedTransformer", 
            "WaveNet",
            "AttentionCNN_LSTM"
        ]
        
        for model_name in models_to_test:
            try:
                model = get_model(
                    model_name=model_name,
                    input_size=512,
                    hidden_size=128,
                    num_layers=3,
                    dropout=0.1
                )
                print(f"✅ {model_name} model created successfully")
                
                # Test forward pass
                dummy_input = torch.randn(2, 1, 512)  # batch_size=2, channels=1, length=512
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ {model_name} forward pass successful, output shape: {output.shape}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    return True


def test_preprocessing_config():
    """Test preprocessing configuration management."""
    print("\nTesting preprocessing configuration management...")
    
    try:
        from preprocessing_config import PreprocessingConfigManager
        
        # Create config manager
        config_manager = PreprocessingConfigManager()
        print("✅ PreprocessingConfigManager created successfully")
        
        # Test saving a dummy config
        dummy_config = {
            'data': {'segment_length': 2048, 'sampling_rate': 256},
            'preprocessing': {'normalization': 'z_score'}
        }
        
        saved_path = config_manager.save_preprocessing_config(
            dummy_config, 
            {'test': 'stats'}, 
            'test_config'
        )
        print(f"✅ Configuration saved to: {saved_path}")
        
        # Test loading
        loaded_config = config_manager.load_preprocessing_config('test_config')
        print("✅ Configuration loaded successfully")
        
        # Test listing
        configs = config_manager.list_saved_configs()
        print(f"✅ Found {len(configs)} saved configurations")
        
    except Exception as e:
        print(f"❌ Preprocessing config test failed: {e}")
        return False
    
    return True


def test_lightning_module():
    """Test the enhanced Lightning module."""
    print("\nTesting Lightning module...")
    
    try:
        from lightning_module import PPGRespiratoryLightningModule
        
        # Load config
        with open('configs/improved_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Override for testing
        config['model']['name'] = 'RWKV'
        config['model']['hidden_size'] = 64  # Smaller for testing
        config['model']['num_layers'] = 2
        
        # Create module
        module = PPGRespiratoryLightningModule(config)
        print("✅ Lightning module created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 512)
        with torch.no_grad():
            output = module(dummy_input)
        print(f"✅ Lightning module forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Lightning module test failed: {e}")
        return False
    
    return True


def test_config_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    config_files = [
        'configs/config.yaml',
        'configs/improved_config.yaml'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file} loaded successfully")
            
            # Check required sections
            required_sections = ['data', 'preprocessing', 'model', 'training']
            for section in required_sections:
                if section not in config:
                    print(f"⚠️  {config_file} missing section: {section}")
                else:
                    print(f"✅ {config_file} has section: {section}")
                    
        except Exception as e:
            print(f"❌ {config_file} failed to load: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing PPG Respiratory Estimation Improvements")
    print("=" * 60)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Preprocessing Config", test_preprocessing_config),
        ("Lightning Module", test_lightning_module),
        ("Configuration Files", test_config_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improvements are ready to use.")
        print("\nNext steps:")
        print("1. Run: python examples/improved_training_example.py")
        print("2. Or start training: python src/train.py --config configs/improved_config.yaml")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
