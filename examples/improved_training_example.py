#!/usr/bin/env python3
"""
Improved Training Example for PPG to Respiratory Waveform Estimation

This example demonstrates:
1. Using the new RWKV and improved model architectures
2. Enhanced logging with train/validation loss and correlation tracking
3. Preprocessing configuration saving and loading
4. TensorBoard visualization setup

Usage:
    python examples/improved_training_example.py
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing_config import (
    PreprocessingConfigManager, 
    create_preprocessing_config_from_saved,
    load_and_adjust_preprocessing_config
)


def run_training_with_different_models():
    """Run training experiments with different model architectures."""
    
    models_to_test = [
        "RWKV",
        "ImprovedTransformer", 
        "WaveNet",
        "AttentionCNN_LSTM"
    ]
    
    base_config = "configs/improved_config.yaml"
    
    print("=" * 60)
    print("IMPROVED PPG TO RESPIRATORY WAVEFORM ESTIMATION")
    print("=" * 60)
    print()
    
    for model_name in models_to_test:
        print(f"Training with {model_name} model...")
        print("-" * 40)
        
        # Run training with model override
        cmd = [
            "python", "src/train.py",
            "--config", base_config,
            "--override", f"model.name={model_name}",
            "--override", f"logging.experiment_name=improved_{model_name.lower()}",
            "--override", "training.max_epochs=10"  # Reduced for demo
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ {model_name} training completed successfully!")
                print("Last few lines of output:")
                print('\n'.join(result.stdout.split('\n')[-10:]))
            else:
                print(f"❌ {model_name} training failed!")
                print("Error output:")
                print(result.stderr[-1000:])  # Last 1000 chars of error
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} training timed out after 1 hour")
        except Exception as e:
            print(f"💥 Error running {model_name}: {e}")
        
        print()


def demonstrate_preprocessing_config_management():
    """Demonstrate preprocessing configuration saving and loading."""
    
    print("=" * 60)
    print("PREPROCESSING CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)
    print()
    
    # Initialize config manager
    config_manager = PreprocessingConfigManager()
    
    # Load base configuration
    with open("configs/improved_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Save a preprocessing configuration
    print("1. Saving preprocessing configuration...")
    saved_path = config_manager.save_preprocessing_config(
        config, 
        {"demo": "statistics"}, 
        "demo_preprocessing_config"
    )
    print(f"   Saved to: {saved_path}")
    print()
    
    # List saved configurations
    print("2. Listing saved configurations...")
    saved_configs = config_manager.list_saved_configs()
    print(f"   Found {len(saved_configs)} saved configurations:")
    for config_name in saved_configs:
        print(f"   - {config_name}")
    print()
    
    # Load and adjust configuration
    print("3. Loading and adjusting configuration...")
    if saved_configs:
        adjusted_config = load_and_adjust_preprocessing_config(
            saved_configs[0],
            segment_length=1024,  # Different segment length
            other_adjustments={
                'preprocessing.normalization': 'min_max',
                'training.batch_size': 32
            }
        )
        print("   ✅ Configuration loaded and adjusted successfully!")
        print(f"   New segment length: {adjusted_config['data']['segment_length']}")
        print(f"   New normalization: {adjusted_config['preprocessing']['normalization']}")
        print(f"   New batch size: {adjusted_config['training']['batch_size']}")
    print()


def setup_tensorboard_monitoring():
    """Set up TensorBoard monitoring for the training runs."""
    
    print("=" * 60)
    print("TENSORBOARD MONITORING SETUP")
    print("=" * 60)
    print()
    
    log_dir = "logs"
    
    if os.path.exists(log_dir):
        print(f"TensorBoard logs directory found: {log_dir}")
        print()
        print("To monitor training in real-time, run:")
        print(f"   tensorboard --logdir {log_dir}")
        print()
        print("Then open your browser to: http://localhost:6006")
        print()
        print("You will see the following metrics:")
        print("   📊 Training Loss (per step and per epoch)")
        print("   📊 Validation Loss (per epoch)")
        print("   📈 Training Pearson Correlation (per epoch)")
        print("   📈 Validation Pearson Correlation (per epoch)")
        print("   📉 Training MAE (per epoch)")
        print("   📉 Validation MAE (per epoch)")
        print()
        
        # Try to start TensorBoard automatically
        try:
            print("Attempting to start TensorBoard automatically...")
            subprocess.Popen([
                "tensorboard", "--logdir", log_dir, "--port", "6006"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ TensorBoard started! Open http://localhost:6006 in your browser")
        except FileNotFoundError:
            print("⚠️  TensorBoard not found. Install with: pip install tensorboard")
        except Exception as e:
            print(f"⚠️  Could not start TensorBoard automatically: {e}")
    else:
        print(f"⚠️  No logs directory found. Run training first to generate logs.")
    
    print()


def show_model_comparison_guide():
    """Show guide for comparing different models."""
    
    print("=" * 60)
    print("MODEL COMPARISON GUIDE")
    print("=" * 60)
    print()
    
    models_info = {
        "RWKV": {
            "description": "Receptance Weighted Key Value - Linear attention mechanism",
            "strengths": ["Linear complexity", "Good for long sequences", "Memory efficient"],
            "best_for": "Long-term temporal dependencies"
        },
        "ImprovedTransformer": {
            "description": "Multi-scale Transformer with positional encoding",
            "strengths": ["Multi-scale features", "Self-attention", "Skip connections"],
            "best_for": "Complex pattern recognition"
        },
        "WaveNet": {
            "description": "Dilated convolutions with gated activations",
            "strengths": ["Large receptive field", "Efficient", "Good for signals"],
            "best_for": "Signal-to-signal translation"
        },
        "AttentionCNN_LSTM": {
            "description": "CNN feature extraction + LSTM + Attention",
            "strengths": ["Multi-scale CNN", "Temporal modeling", "Attention mechanism"],
            "best_for": "Balanced approach with attention"
        }
    }
    
    for model_name, info in models_info.items():
        print(f"🤖 {model_name}")
        print(f"   Description: {info['description']}")
        print(f"   Strengths: {', '.join(info['strengths'])}")
        print(f"   Best for: {info['best_for']}")
        print()
    
    print("💡 Tips for model selection:")
    print("   - Start with RWKV for efficiency and long sequences")
    print("   - Use ImprovedTransformer for complex patterns")
    print("   - Try WaveNet for pure signal processing tasks")
    print("   - Use AttentionCNN_LSTM as a balanced baseline")
    print()


def main():
    """Main function to run all demonstrations."""
    
    print("🚀 PPG to Respiratory Waveform Estimation - Improved Training Demo")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/train.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("preprocessing_configs", exist_ok=True)
    
    # Show model comparison guide
    show_model_comparison_guide()
    
    # Demonstrate preprocessing config management
    demonstrate_preprocessing_config_management()
    
    # Setup TensorBoard monitoring
    setup_tensorboard_monitoring()
    
    # Ask user if they want to run training
    print("=" * 60)
    print("TRAINING EXECUTION")
    print("=" * 60)
    print()
    
    response = input("Do you want to run training with different models? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_training_with_different_models()
    else:
        print("Skipping training execution.")
        print()
        print("To run training manually, use:")
        print("   python src/train.py --config configs/improved_config.yaml")
        print()
    
    print("=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. 📊 Monitor training with TensorBoard: http://localhost:6006")
    print("2. 🔍 Check results in the 'results/' directory")
    print("3. ⚙️  Adjust hyperparameters in 'configs/improved_config.yaml'")
    print("4. 🧪 Experiment with different models using --override")
    print()
    print("Example commands:")
    print("   # Train RWKV model")
    print("   python src/train.py --config configs/improved_config.yaml --override model.name=RWKV")
    print()
    print("   # Train with different segment length")
    print("   python src/train.py --config configs/improved_config.yaml --override data.segment_length=1024")
    print()
    print("   # Train with higher learning rate")
    print("   python src/train.py --config configs/improved_config.yaml --override training.learning_rate=0.001")


if __name__ == "__main__":
    main()
