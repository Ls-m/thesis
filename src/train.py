import os
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from typing import Dict, List
import pickle

from data_utils import DataPreprocessor, create_cross_validation_splits, prepare_fold_data
from dataset import create_data_loaders
from lightning_module import PPGRespiratoryLightningModule


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
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
                     test_subject: str) -> Dict:
    """Train a single fold of cross-validation."""
    
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_id} - Test Subject: {test_subject}")
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
        version=f'fold_{fold_id}_{test_subject}'
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
        gradient_clip_val=gradient_clip_val,  # Use gradient clipping from config
        # detect_anomaly=True,     # Detect NaN/Inf in gradients
        accumulate_grad_batches=1,  # No gradient accumulation for stability
        deterministic=True,      # For reproducibility
        benchmark=False          # Disable cudnn benchmark for stability
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


def run_cross_validation(config: Dict) -> Dict:
    """Run leave-one-out cross-validation."""
    
    print("Starting data preprocessing...")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Prepare dataset
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    
    print(f"Processed data for {len(processed_data)} subjects")
    
    # Create cross-validation splits
    cv_splits = create_cross_validation_splits(processed_data)
    
    print(f"Created {len(cv_splits)} cross-validation folds")
    
    # Store all fold results
    all_fold_results = []
    
    # Run each fold
    for cv_split in cv_splits:
        fold_id = cv_split['fold_id']
        test_subject = cv_split['test_subject']
        train_subjects = cv_split['train_subjects']
        
        # Prepare fold data
        fold_data = prepare_fold_data(
            processed_data,
            train_subjects,
            test_subject,
            val_split=config['training']['val_split']
        )
        
        # Train fold
        fold_results = train_single_fold(
            config, fold_data, fold_id, test_subject
        )
        
        all_fold_results.append(fold_results)
        
        # Save intermediate results
        results_path = os.path.join('results', f'fold_{fold_id}_results.pkl')
        os.makedirs('results', exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(fold_results, f)
    
    return {
        'fold_results': all_fold_results,
        'processed_data': processed_data
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPG to Respiratory Waveform Estimation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create necessary directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run cross-validation
    cv_results = run_cross_validation(config)
    
    # Save complete results
    results_path = os.path.join('results', 'complete_cv_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\nCross-validation completed!")
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
        print(f"Average Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f"Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")


if __name__ == "__main__":
    main()
