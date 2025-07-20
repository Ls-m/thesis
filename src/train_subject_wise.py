import os
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from typing import Dict, List, Any
import pickle
from time import sleep
from subject_wise_data_utils import (
    SubjectWiseDataPreprocessor, 
    create_subject_wise_splits, 
    prepare_subject_wise_fold_data,
    print_split_summary
)
from dataset import create_data_loaders
from lightning_module import PPGRespiratoryLightningModule
from preprocessing_config import PreprocessingConfigManager, EnhancedDataPreprocessor


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_config_override(override_str: str) -> tuple:
    """
    Parse a config override string in the format 'key.subkey=value'.
    
    Args:
        override_str: String in format 'key.subkey=value' or 'key=value'
        
    Returns:
        tuple: (key_path_list, value)
        
    Examples:
        'training.learning_rate=0.001' -> (['training', 'learning_rate'], 0.001)
        'model.name=CNN1D' -> (['model', 'name'], 'CNN1D')
        'training.batch_size=64' -> (['training', 'batch_size'], 64)
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value' or 'key.subkey=value'")
    
    key_path, value_str = override_str.split('=', 1)
    key_list = key_path.split('.')
    
    # Try to convert value to appropriate type
    value = convert_string_to_type(value_str)
    
    return key_list, value


def convert_string_to_type(value_str: str) -> Any:
    """
    Convert string value to appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Converted value with appropriate type
    """
    # Handle boolean values
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # Handle None/null values
    if value_str.lower() in ['none', 'null']:
        return None
    
    # Try to convert to int
    try:
        if '.' not in value_str:
            return int(value_str)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle lists (comma-separated values in brackets)
    if value_str.startswith('[') and value_str.endswith(']'):
        list_str = value_str[1:-1].strip()
        if not list_str:
            return []
        items = [convert_string_to_type(item.strip()) for item in list_str.split(',')]
        return items
    
    # Return as string if no other type matches
    return value_str


def update_nested_dict(config: Dict, key_path: List[str], value: Any) -> None:
    """
    Update a nested dictionary with a value at the specified key path.
    
    Args:
        config: The configuration dictionary to update
        key_path: List of keys representing the path to the value
        value: The new value to set
        
    Examples:
        update_nested_dict(config, ['training', 'learning_rate'], 0.001)
        update_nested_dict(config, ['model', 'name'], 'CNN1D')
    """
    current = config
    
    # Navigate to the parent of the target key
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(f"Cannot override {'.'.join(key_path)}: {key} is not a dictionary")
        current = current[key]
    
    # Set the final value
    final_key = key_path[-1]
    old_value = current.get(final_key, "NOT_SET")
    current[final_key] = value
    
    print(f"Config override: {'.'.join(key_path)} = {value} (was: {old_value})")


def apply_config_overrides(config: Dict, overrides: List[str]) -> Dict:
    """
    Apply command line config overrides to the configuration dictionary.
    
    Args:
        config: Base configuration dictionary
        overrides: List of override strings in format 'key.subkey=value'
        
    Returns:
        Updated configuration dictionary
    """
    for override in overrides:
        try:
            key_path, value = parse_config_override(override)
            update_nested_dict(config, key_path, value)
        except Exception as e:
            print(f"Warning: Failed to apply override '{override}': {e}")
            continue
    
    return config


def setup_callbacks(config: Dict) -> List:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['logging']['log_dir'], 'checkpoints'),
        filename='{epoch}-{val_loss:.4f}-{val_correlation:.4f}',
        monitor='val_correlation',
        mode='max',
        save_top_k=config['logging']['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_correlation',
        mode='max',
        patience=config['training']['patience'],
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stopping)
    
    return callbacks


def train_single_fold(config: Dict, fold_data: Dict, fold_id: int, 
                     test_subject: str, train_subjects: List[str], 
                     val_subjects: List[str]) -> Dict:
    """Train a single fold of cross-validation with subject-wise splitting."""
    
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_id} - Test Subject: {test_subject}")
    print(f"Train Subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val Subjects ({len(val_subjects)}): {val_subjects}")
    print(f"{'='*50}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        fold_data, 
        batch_size=config['training']['batch_size'],
        num_workers=4
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Val samples: {len(data_loaders['val'].dataset)}")
    print(f"Test samples: {len(data_loaders['test'].dataset)}")
    
    # Initialize model
    model = PPGRespiratoryLightningModule(config)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name'],
        version=f'fold_{fold_id}_{test_subject}_subject_wise'
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Initialize trainer
    gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=1,
        deterministic=True,
        benchmark=False
    )
    
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders['val']
    )
    
    # Test the model
    test_results = trainer.test(
        model,
        dataloaders=data_loaders['test'],
        ckpt_path='best'
    )
    
    sleep(10)
    # Get predictions for ensemble
    model = PPGRespiratoryLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        config=config
    )
    model.eval()
    
    predictions = trainer.predict(
        model,
        dataloaders=data_loaders['test']
    )
    
    # Combine predictions
    all_predictions = torch.cat([p['predictions'] for p in predictions], dim=0)
    all_targets = torch.cat([p['targets'] for p in predictions], dim=0)
    
    fold_results = {
        'fold_id': fold_id,
        'test_subject': test_subject,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_metrics': test_results[0] if test_results else {},
        'predictions': all_predictions.cpu().numpy(),
        'targets': all_targets.cpu().numpy(),
        'best_model_path': trainer.checkpoint_callback.best_model_path
    }
    
    print(f"Fold {fold_id} completed!")
    if test_results:
        print(f"Test Correlation: {test_results[0].get('test_correlation', 'N/A'):.4f}")
        print(f"Test Loss: {test_results[0].get('test_loss', 'N/A'):.4f}")
        print(f"Test MAE: {test_results[0].get('test_mae', 'N/A'):.4f}")
    
    return fold_results


def run_subject_wise_cross_validation(config: Dict, target_subject: str = None, 
                                     val_split: float = 0.2, random_seed: int = 42) -> Dict:
    """Run subject-wise cross-validation."""
    
    print("Starting data preprocessing with subject-wise splitting...")
    
    # Initialize subject-wise data preprocessor
    preprocessor = SubjectWiseDataPreprocessor(config)
    
    # Prepare dataset
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    
    print(f"Processed data for {len(processed_data)} subjects")
    
    # List available subjects
    available_subjects = list(processed_data.keys())
    print(f"Available subjects: {available_subjects}")
    
    # Save preprocessing configuration
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['name']
    config_name = f"{model_name}_subject_wise_preprocessing_{timestamp}"
    
    # Create some basic stats for saving
    total_segments = sum(len(ppg_segments) for ppg_segments, _ in processed_data.values())
    preprocessing_stats = {
        'subjects_processed': len(processed_data),
        'total_segments': total_segments,
        'model_used': model_name,
        'timestamp': timestamp,
        'splitting_method': 'subject_wise'
    }
    
    # Initialize config manager for saving
    config_manager = PreprocessingConfigManager()
    saved_config_path = config_manager.save_preprocessing_config(
        config, preprocessing_stats, config_name
    )
    print(f"Preprocessing configuration saved as: {config_name}")
    
    # Create subject-wise splits
    splits = create_subject_wise_splits(
        processed_data, 
        test_subject=target_subject, 
        val_split=val_split, 
        random_seed=random_seed
    )
    
    # Print split summary
    print_split_summary(splits, processed_data)
    
    # Store all fold results
    all_fold_results = []
    
    # Run each fold
    for split in splits:
        fold_id = split['fold_id']
        test_subject = split['test_subject']
        train_subjects = split['train_subjects']
        val_subjects = split['val_subjects']
        
        print(f"\nFold {fold_id}: Test subject = {test_subject}")
        print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
        
        # Prepare fold data with subject-wise splitting
        fold_data = prepare_subject_wise_fold_data(
            processed_data,
            train_subjects,
            val_subjects,
            test_subject
        )
        
        # Train fold
        fold_results = train_single_fold(
            config, fold_data, fold_id, test_subject, train_subjects, val_subjects
        )
        
        all_fold_results.append(fold_results)
        
        # Save intermediate results
        results_path = os.path.join('results', f'subject_wise_fold_{fold_id}_results.pkl')
        os.makedirs('results', exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(fold_results, f)
        
        # If running single fold, break after first iteration
        if target_subject:
            break
    
    return {
        'fold_results': all_fold_results,
        'processed_data': processed_data,
        'preprocessing_config_name': config_name,
        'preprocessing_config_path': saved_config_path,
        'target_subject': target_subject,
        'mode': 'single_fold' if target_subject else 'full_cv',
        'splitting_method': 'subject_wise',
        'val_split': val_split,
        'random_seed': random_seed
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train PPG to Respiratory Waveform Estimation with Subject-wise Splitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full leave-one-out cross-validation with subject-wise splitting (default)
  python src/train_subject_wise.py --config configs/improved_config.yaml
  
  # Single fold with specific test subject and subject-wise validation
  python src/train_subject_wise.py --config configs/improved_config.yaml --fold subject_01
  
  # Use benchmark dataset (BIDMC) with subject-wise splitting
  python src/train_subject_wise.py --config configs/improved_config.yaml --dataset bidmc
  
  # Use alternate dataset with custom validation split
  python src/train_subject_wise.py --config configs/improved_config.yaml --dataset csv --val-split 0.3
  
  # Single fold with benchmark dataset and custom seed
  python src/train_subject_wise.py --config configs/improved_config.yaml --dataset bidmc --fold subject_01 --seed 123
  
  # Config overrides
  python src/train_subject_wise.py --override training.learning_rate=0.001
  python src/train_subject_wise.py --override model.name=RWKV --override training.batch_size=32
        """
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--fold', type=str, default=None,
                       help='Specific subject to use as test set (e.g., subject_01, AL25). If not specified, runs full leave-one-out CV')
    parser.add_argument('--dataset', type=str, choices=['bidmc', 'csv'], default=None,
                       help='Dataset to use: "bidmc" for benchmark dataset (src/bidmc/), "csv" for alternate dataset (src/csv/)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of training subjects to use for validation (default: 0.2)')
    parser.add_argument('--list-subjects', action='store_true',
                       help='List available subjects in the dataset and exit')
    parser.add_argument('--override', action='append', default=[],
                       help='Override config values. Format: key.subkey=value (can be used multiple times)')
    parser.add_argument('--print-config', action='store_true',
                       help='Print the final configuration and exit without training')
    
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle dataset selection
    if args.dataset:
        if args.dataset == 'bidmc':
            config['data']['csv_folder'] = 'src/bidmc'
            print(f"Using benchmark dataset: {config['data']['csv_folder']}")
        elif args.dataset == 'csv':
            config['data']['csv_folder'] = 'src/csv'
            print(f"Using alternate dataset: {config['data']['csv_folder']}")
    
    # Apply config overrides
    if args.override:
        print(f"\nApplying {len(args.override)} config override(s):")
        config = apply_config_overrides(config, args.override)
    
    # If listing subjects, do that and exit
    if args.list_subjects:
        print("Loading dataset to list available subjects...")
        preprocessor = SubjectWiseDataPreprocessor(config)
        processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
        available_subjects = list(processed_data.keys())
        
        print(f"\nDataset: {config['data']['csv_folder']}")
        print(f"Available subjects ({len(available_subjects)}):")
        for i, subject in enumerate(sorted(available_subjects), 1):
            print(f"  {i:2d}. {subject}")
        
        print(f"\nTo train with a specific subject as test set, use:")
        print(f"  python src/train_subject_wise.py --config {args.config} --fold <subject_name>")
        if args.dataset:
            print(f"  python src/train_subject_wise.py --config {args.config} --dataset {args.dataset} --fold <subject_name>")
        return
    
    print("\nFinal Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # If only printing config, exit here
    if args.print_config:
        print("Configuration printed. Exiting without training.")
        return
    
    # Print training mode
    if args.fold:
        print(f"\n🎯 SINGLE FOLD TRAINING MODE (SUBJECT-WISE SPLITTING)")
        print(f"Test Subject: {args.fold}")
        print(f"Dataset: {config['data']['csv_folder']}")
        print(f"Validation Split: {args.val_split:.1%} of training subjects")
    else:
        print(f"\n🔄 FULL LEAVE-ONE-OUT CROSS-VALIDATION MODE (SUBJECT-WISE SPLITTING)")
        print(f"Dataset: {config['data']['csv_folder']}")
        print(f"Validation Split: {args.val_split:.1%} of training subjects per fold")
    
    print(f"Random Seed: {args.seed}")
    
    # Create necessary directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run subject-wise cross-validation
    cv_results = run_subject_wise_cross_validation(
        config, 
        target_subject=args.fold, 
        val_split=args.val_split, 
        random_seed=args.seed
    )
    
    # Save complete results
    if args.fold:
        results_filename = f'subject_wise_single_fold_{args.fold}_results.pkl'
    else:
        results_filename = 'subject_wise_complete_cv_results.pkl'
    
    results_path = os.path.join('results', results_filename)
    with open(results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\nTraining completed!")
    print(f"Mode: {cv_results['mode']} with {cv_results['splitting_method']} splitting")
    if cv_results['target_subject']:
        print(f"Test Subject: {cv_results['target_subject']}")
    print(f"Validation Split: {cv_results['val_split']:.1%}")
    print(f"Random Seed: {cv_results['random_seed']}")
    print(f"Results saved to: {results_path}")
    
    # Print summary statistics
    fold_results = cv_results['fold_results']
    correlations = []
    losses = []
    maes = []
    
    for fold_result in fold_results:
        metrics = fold_result['test_metrics']
        if 'test_correlation' in metrics:
            correlations.append(metrics['test_correlation'])
        if 'test_loss' in metrics:
            losses.append(metrics['test_loss'])
        if 'test_mae' in metrics:
            maes.append(metrics['test_mae'])
    
    if correlations:
        print(f"\nSummary Statistics:")
        if len(correlations) == 1:
            print(f"Test Correlation: {correlations[0]:.4f}")
            print(f"Test Loss: {losses[0]:.4f}")
            print(f"Test MAE: {maes[0]:.4f}")
        else:
            print(f"Average Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
            print(f"Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
            print(f"Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    
    print(f"\n✅ Subject-wise splitting ensures no data leakage between train/val/test sets!")


if __name__ == "__main__":
    main()
