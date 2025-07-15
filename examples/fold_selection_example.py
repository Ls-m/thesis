#!/usr/bin/env python3
"""
Fold Selection Example for PPG to Respiratory Waveform Estimation

This example demonstrates the new fold selection functionality:
1. Listing available subjects in datasets
2. Running single fold training with specific test subjects
3. Switching between different datasets
4. Comparing results across different folds

Usage:
    python examples/fold_selection_example.py
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Command executed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Command failed!")
            print("\nError:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 5 minutes")
    except Exception as e:
        print(f"💥 Error running command: {e}")


def demonstrate_subject_listing():
    """Demonstrate listing subjects in different datasets."""
    
    print("=" * 80)
    print("📋 SUBJECT LISTING DEMONSTRATION")
    print("=" * 80)
    
    # List subjects in CSV dataset
    run_command([
        "python", "src/train.py",
        "--config", "configs/improved_config.yaml",
        "--dataset", "csv",
        "--list-subjects"
    ], "Listing subjects in CSV dataset (src/csv/)")
    
    # List subjects in BIDMC dataset
    run_command([
        "python", "src/train.py", 
        "--config", "configs/improved_config.yaml",
        "--dataset", "bidmc",
        "--list-subjects"
    ], "Listing subjects in BIDMC dataset (src/bidmc/)")


def demonstrate_single_fold_training():
    """Demonstrate single fold training with specific subjects."""
    
    print("=" * 80)
    print("🎯 SINGLE FOLD TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Note: These are example commands - actual subject names depend on your datasets
    example_commands = [
        {
            "description": "Single fold training with CSV dataset",
            "cmd": [
                "python", "src/train.py",
                "--config", "configs/improved_config.yaml",
                "--dataset", "csv",
                "--fold", "AL25",  # Example subject name
                "--override", "training.max_epochs=5",  # Reduced for demo
                "--override", "model.name=RWKV"
            ]
        },
        {
            "description": "Single fold training with BIDMC dataset",
            "cmd": [
                "python", "src/train.py",
                "--config", "configs/improved_config.yaml", 
                "--dataset", "bidmc",
                "--fold", "subject_01",  # Example subject name
                "--override", "training.max_epochs=5",  # Reduced for demo
                "--override", "model.name=ImprovedTransformer"
            ]
        }
    ]
    
    print("📝 Example commands for single fold training:")
    print("(Note: Actual subject names depend on your datasets)")
    print()
    
    for i, example in enumerate(example_commands, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {' '.join(example['cmd'])}")
        print()
    
    print("💡 To run these commands:")
    print("1. First list subjects to see available names")
    print("2. Replace example subject names with actual ones from your dataset")
    print("3. Run the commands above")


def demonstrate_comparison_workflow():
    """Demonstrate workflow for comparing different folds."""
    
    print("=" * 80)
    print("📊 FOLD COMPARISON WORKFLOW")
    print("=" * 80)
    
    workflow_steps = [
        "1. List available subjects",
        "2. Select multiple subjects for comparison",
        "3. Train models with each subject as test set",
        "4. Compare results across folds",
        "5. Analyze performance variations"
    ]
    
    print("🔄 Recommended workflow for fold comparison:")
    print()
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print()
    print("📋 Example comparison script:")
    print()
    
    comparison_script = '''
# Step 1: List subjects
python src/train.py --config configs/improved_config.yaml --dataset csv --list-subjects

# Step 2: Train with different test subjects
python src/train.py --config configs/improved_config.yaml --dataset csv --fold AL25 --override model.name=RWKV
python src/train.py --config configs/improved_config.yaml --dataset csv --fold MM21 --override model.name=RWKV
python src/train.py --config configs/improved_config.yaml --dataset csv --fold XX30 --override model.name=RWKV

# Step 3: Compare results
# Results will be saved as:
# - results/single_fold_AL25_results.pkl
# - results/single_fold_MM21_results.pkl  
# - results/single_fold_XX30_results.pkl
'''
    
    print(comparison_script)


def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    
    print("=" * 80)
    print("🚀 ADVANCED USAGE PATTERNS")
    print("=" * 80)
    
    advanced_examples = [
        {
            "title": "Model Architecture Comparison",
            "description": "Compare different models on the same test subject",
            "commands": [
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override model.name=RWKV",
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override model.name=ImprovedTransformer",
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override model.name=WaveNet"
            ]
        },
        {
            "title": "Hyperparameter Sweep on Single Fold",
            "description": "Test different hyperparameters on a specific subject",
            "commands": [
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override training.learning_rate=0.0001",
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override training.learning_rate=0.0005",
                "python src/train.py --config configs/improved_config.yaml --fold subject_01 --override training.learning_rate=0.001"
            ]
        },
        {
            "title": "Cross-Dataset Validation",
            "description": "Train on one dataset, test on specific subject from another",
            "commands": [
                "# Train on CSV dataset with AL25 as test",
                "python src/train.py --config configs/improved_config.yaml --dataset csv --fold AL25",
                "# Train on BIDMC dataset with subject_01 as test", 
                "python src/train.py --config configs/improved_config.yaml --dataset bidmc --fold subject_01"
            ]
        }
    ]
    
    for example in advanced_examples:
        print(f"🎯 {example['title']}")
        print(f"   {example['description']}")
        print()
        for cmd in example['commands']:
            if cmd.startswith('#'):
                print(f"   {cmd}")
            else:
                print(f"   {cmd}")
        print()


def show_cli_reference():
    """Show complete CLI reference."""
    
    print("=" * 80)
    print("📖 COMPLETE CLI REFERENCE")
    print("=" * 80)
    
    cli_options = [
        ("--config", "Path to configuration file", "configs/improved_config.yaml"),
        ("--fold", "Specific subject for test set", "subject_01, AL25, etc."),
        ("--dataset", "Dataset to use", "bidmc, csv"),
        ("--list-subjects", "List available subjects and exit", ""),
        ("--seed", "Random seed for reproducibility", "42"),
        ("--override", "Override config values", "model.name=RWKV"),
        ("--print-config", "Print configuration and exit", "")
    ]
    
    print("🔧 Available CLI options:")
    print()
    
    for option, description, example in cli_options:
        print(f"   {option:15} {description}")
        if example:
            print(f"                   Example: {example}")
        print()
    
    print("📝 Usage patterns:")
    print()
    
    usage_patterns = [
        ("Full CV", "python src/train.py --config configs/improved_config.yaml"),
        ("Single fold", "python src/train.py --config configs/improved_config.yaml --fold subject_01"),
        ("Dataset switch", "python src/train.py --config configs/improved_config.yaml --dataset bidmc"),
        ("List subjects", "python src/train.py --config configs/improved_config.yaml --dataset csv --list-subjects"),
        ("Model override", "python src/train.py --config configs/improved_config.yaml --fold AL25 --override model.name=RWKV")
    ]
    
    for pattern_name, command in usage_patterns:
        print(f"   {pattern_name:12} {command}")
    print()


def main():
    """Main demonstration function."""
    
    print("🚀 PPG to Respiratory Waveform Estimation - Fold Selection Demo")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/train.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Show CLI reference
    show_cli_reference()
    
    # Demonstrate subject listing
    demonstrate_subject_listing()
    
    # Demonstrate single fold training
    demonstrate_single_fold_training()
    
    # Demonstrate comparison workflow
    demonstrate_comparison_workflow()
    
    # Demonstrate advanced usage
    demonstrate_advanced_usage()
    
    print("=" * 80)
    print("✅ FOLD SELECTION DEMO COMPLETED!")
    print("=" * 80)
    print()
    print("🎯 Key Features Added:")
    print("   • Single fold training with --fold argument")
    print("   • Dataset switching with --dataset argument")
    print("   • Subject listing with --list-subjects")
    print("   • Automatic result file naming for single folds")
    print("   • Full backward compatibility with existing CV workflow")
    print()
    print("📋 Next Steps:")
    print("   1. List subjects in your datasets")
    print("   2. Try single fold training with specific subjects")
    print("   3. Compare results across different folds")
    print("   4. Use TensorBoard to visualize training progress")
    print()
    print("🔗 Quick Commands:")
    print("   # List CSV dataset subjects")
    print("   python src/train.py --config configs/improved_config.yaml --dataset csv --list-subjects")
    print()
    print("   # List BIDMC dataset subjects")
    print("   python src/train.py --config configs/improved_config.yaml --dataset bidmc --list-subjects")
    print()
    print("   # Train with specific subject (replace 'subject_name' with actual name)")
    print("   python src/train.py --config configs/improved_config.yaml --fold subject_name")


if __name__ == "__main__":
    main()
