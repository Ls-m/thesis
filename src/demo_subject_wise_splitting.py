#!/usr/bin/env python3
"""
Demonstration script for subject-wise data splitting.

This script shows how the subject-wise splitting works and validates that
there's no data leakage between train/validation/test sets.
"""

import os
import yaml
import numpy as np
from typing import Dict, List
from subject_wise_data_utils import (
    SubjectWiseDataPreprocessor,
    create_subject_wise_splits,
    prepare_subject_wise_fold_data,
    print_split_summary
)


def load_config(config_path: str = 'configs/config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def validate_no_data_leakage(fold_data: Dict, train_subjects: List[str], 
                           val_subjects: List[str], test_subject: str):
    """
    Validate that there's no data leakage between train/val/test sets.
    
    This function checks that:
    1. All subjects are properly separated
    2. No subject appears in multiple sets
    3. Data shapes are consistent
    """
    print(f"\n🔍 VALIDATING DATA SPLIT (No Data Leakage Check)")
    print(f"{'='*50}")
    
    # Check subject separation
    all_subjects = set(train_subjects + val_subjects + [test_subject])
    print(f"Total unique subjects: {len(all_subjects)}")
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subject: {test_subject}")
    
    # Check for overlaps
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = {test_subject}
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap:
        print(f"❌ ERROR: Train-Val overlap detected: {train_val_overlap}")
    else:
        print(f"✅ No Train-Val overlap")
    
    if train_test_overlap:
        print(f"❌ ERROR: Train-Test overlap detected: {train_test_overlap}")
    else:
        print(f"✅ No Train-Test overlap")
    
    if val_test_overlap:
        print(f"❌ ERROR: Val-Test overlap detected: {val_test_overlap}")
    else:
        print(f"✅ No Val-Test overlap")
    
    # Check data shapes
    print(f"\nData shapes:")
    print(f"  Train: PPG={fold_data['train_ppg'].shape}, RESP={fold_data['train_resp'].shape}")
    print(f"  Val:   PPG={fold_data['val_ppg'].shape}, RESP={fold_data['val_resp'].shape}")
    print(f"  Test:  PPG={fold_data['test_ppg'].shape}, RESP={fold_data['test_resp'].shape}")
    
    # Check for NaN values
    train_ppg_nan = np.isnan(fold_data['train_ppg']).sum()
    train_resp_nan = np.isnan(fold_data['train_resp']).sum()
    val_ppg_nan = np.isnan(fold_data['val_ppg']).sum()
    val_resp_nan = np.isnan(fold_data['val_resp']).sum()
    test_ppg_nan = np.isnan(fold_data['test_ppg']).sum()
    test_resp_nan = np.isnan(fold_data['test_resp']).sum()
    
    print(f"\nNaN values:")
    print(f"  Train: PPG={train_ppg_nan}, RESP={train_resp_nan}")
    print(f"  Val:   PPG={val_ppg_nan}, RESP={val_resp_nan}")
    print(f"  Test:  PPG={test_ppg_nan}, RESP={test_resp_nan}")
    
    total_nan = train_ppg_nan + train_resp_nan + val_ppg_nan + val_resp_nan + test_ppg_nan + test_resp_nan
    if total_nan == 0:
        print(f"✅ No NaN values detected")
    else:
        print(f"⚠️  Warning: {total_nan} NaN values detected")
    
    print(f"{'='*50}")


def demo_single_fold_splitting(config: Dict, dataset_name: str = 'bidmc'):
    """Demonstrate single fold splitting with a specific test subject."""
    
    print(f"\n🎯 SINGLE FOLD SPLITTING DEMO")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Set dataset path
    if dataset_name == 'bidmc':
        config['data']['csv_folder'] = 'src/bidmc'
    else:
        config['data']['csv_folder'] = 'src/csv'
    
    # Initialize preprocessor
    preprocessor = SubjectWiseDataPreprocessor(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    
    available_subjects = list(processed_data.keys())
    print(f"Available subjects ({len(available_subjects)}): {available_subjects}")
    
    # Choose a test subject (first one for demo)
    test_subject = available_subjects[0]
    print(f"Selected test subject: {test_subject}")
    
    # Create subject-wise splits
    splits = create_subject_wise_splits(
        processed_data, 
        test_subject=test_subject, 
        val_split=0.2, 
        random_seed=42
    )
    
    # Print split summary
    print_split_summary(splits, processed_data)
    
    # Get the split details
    split = splits[0]  # Only one split for single fold
    train_subjects = split['train_subjects']
    val_subjects = split['val_subjects']
    
    # Prepare fold data
    fold_data = prepare_subject_wise_fold_data(
        processed_data,
        train_subjects,
        val_subjects,
        test_subject
    )
    
    # Validate no data leakage
    validate_no_data_leakage(fold_data, train_subjects, val_subjects, test_subject)
    
    return processed_data, splits, fold_data


def demo_cross_validation_splitting(config: Dict, dataset_name: str = 'bidmc', max_folds: int = 3):
    """Demonstrate cross-validation splitting (limited to first few folds for demo)."""
    
    print(f"\n🔄 CROSS-VALIDATION SPLITTING DEMO")
    print(f"Dataset: {dataset_name}")
    print(f"Max folds to show: {max_folds}")
    print(f"{'='*60}")
    
    # Set dataset path
    if dataset_name == 'bidmc':
        config['data']['csv_folder'] = 'src/bidmc'
    else:
        config['data']['csv_folder'] = 'src/csv'
    
    # Initialize preprocessor
    preprocessor = SubjectWiseDataPreprocessor(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    
    available_subjects = list(processed_data.keys())
    print(f"Available subjects ({len(available_subjects)}): {available_subjects}")
    
    # Create subject-wise splits for cross-validation
    splits = create_subject_wise_splits(
        processed_data, 
        test_subject=None,  # None means full cross-validation
        val_split=0.2, 
        random_seed=42
    )
    
    print(f"Total folds: {len(splits)}")
    print(f"Showing first {min(max_folds, len(splits))} folds:")
    
    # Show details for first few folds
    for i, split in enumerate(splits[:max_folds]):
        fold_id = split['fold_id']
        test_subject = split['test_subject']
        train_subjects = split['train_subjects']
        val_subjects = split['val_subjects']
        
        print(f"\n--- Fold {fold_id} ---")
        print(f"Test subject: {test_subject}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
        
        # Prepare fold data
        fold_data = prepare_subject_wise_fold_data(
            processed_data,
            train_subjects,
            val_subjects,
            test_subject
        )
        
        # Quick validation
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        test_set = {test_subject}
        
        overlaps = []
        if train_set.intersection(val_set):
            overlaps.append("Train-Val")
        if train_set.intersection(test_set):
            overlaps.append("Train-Test")
        if val_set.intersection(test_set):
            overlaps.append("Val-Test")
        
        if overlaps:
            print(f"❌ Overlaps detected: {', '.join(overlaps)}")
        else:
            print(f"✅ No overlaps detected")
        
        print(f"Data shapes: Train={fold_data['train_ppg'].shape}, Val={fold_data['val_ppg'].shape}, Test={fold_data['test_ppg'].shape}")
    
    return processed_data, splits


def compare_splitting_methods(config: Dict, dataset_name: str = 'bidmc'):
    """Compare subject-wise splitting with random splitting to show the difference."""
    
    print(f"\n⚖️  COMPARISON: Subject-wise vs Random Splitting")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Set dataset path
    if dataset_name == 'bidmc':
        config['data']['csv_folder'] = 'src/bidmc'
    else:
        config['data']['csv_folder'] = 'src/csv'
    
    # Initialize preprocessor
    preprocessor = SubjectWiseDataPreprocessor(config)
    
    # Load and preprocess data
    processed_data = preprocessor.prepare_dataset(config['data']['csv_folder'])
    available_subjects = list(processed_data.keys())
    
    # Choose a test subject
    test_subject = available_subjects[0]
    remaining_subjects = [s for s in available_subjects if s != test_subject]
    
    print(f"Test subject: {test_subject}")
    print(f"Remaining subjects: {remaining_subjects}")
    
    # Method 1: Subject-wise splitting
    print(f"\n1️⃣ SUBJECT-WISE SPLITTING:")
    splits = create_subject_wise_splits(
        processed_data, 
        test_subject=test_subject, 
        val_split=0.2, 
        random_seed=42
    )
    
    split = splits[0]
    sw_train_subjects = split['train_subjects']
    sw_val_subjects = split['val_subjects']
    
    print(f"   Train subjects: {sw_train_subjects}")
    print(f"   Val subjects: {sw_val_subjects}")
    
    sw_fold_data = prepare_subject_wise_fold_data(
        processed_data, sw_train_subjects, sw_val_subjects, test_subject
    )
    
    print(f"   Train samples: {len(sw_fold_data['train_ppg'])}")
    print(f"   Val samples: {len(sw_fold_data['val_ppg'])}")
    print(f"   Test samples: {len(sw_fold_data['test_ppg'])}")
    
    # Method 2: Simulate random splitting (like the old method)
    print(f"\n2️⃣ SIMULATED RANDOM SPLITTING (for comparison):")
    
    # Combine all training data first
    all_train_ppg = []
    all_train_resp = []
    for subject in remaining_subjects:
        ppg_segments, resp_segments = processed_data[subject]
        all_train_ppg.append(ppg_segments)
        all_train_resp.append(resp_segments)
    
    combined_train_ppg = np.concatenate(all_train_ppg, axis=0)
    combined_train_resp = np.concatenate(all_train_resp, axis=0)
    
    # Random split
    n_samples = len(combined_train_ppg)
    n_val = int(n_samples * 0.2)
    
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    random_train_ppg = combined_train_ppg[train_indices]
    random_val_ppg = combined_train_ppg[val_indices]
    
    print(f"   All training subjects used: {remaining_subjects}")
    print(f"   Train samples: {len(random_train_ppg)}")
    print(f"   Val samples: {len(random_val_ppg)}")
    print(f"   Test samples: {len(sw_fold_data['test_ppg'])}")
    
    print(f"\n🔍 KEY DIFFERENCES:")
    print(f"   Subject-wise: Validation uses separate subjects ({sw_val_subjects})")
    print(f"   Random: Validation mixes segments from all training subjects")
    print(f"   Subject-wise: Prevents data leakage from same subject")
    print(f"   Random: May have segments from same subject in both train and val")
    
    return sw_fold_data


def main():
    """Main demonstration function."""
    
    print("🧪 SUBJECT-WISE DATA SPLITTING DEMONSTRATION")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Demo 1: Single fold splitting
    try:
        print("\n" + "="*80)
        demo_single_fold_splitting(config, dataset_name='bidmc')
    except Exception as e:
        print(f"Error in single fold demo: {e}")
    
    # Demo 2: Cross-validation splitting (first 3 folds)
    try:
        print("\n" + "="*80)
        demo_cross_validation_splitting(config, dataset_name='bidmc', max_folds=3)
    except Exception as e:
        print(f"Error in cross-validation demo: {e}")
    
    # Demo 3: Comparison with random splitting
    try:
        print("\n" + "="*80)
        compare_splitting_methods(config, dataset_name='bidmc')
    except Exception as e:
        print(f"Error in comparison demo: {e}")
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETED!")
    print("\nKey Benefits of Subject-wise Splitting:")
    print("1. 🚫 No data leakage between train/validation/test sets")
    print("2. 🎯 More realistic evaluation (subjects are truly independent)")
    print("3. 🔄 Better generalization assessment")
    print("4. 📊 Proper cross-validation for subject-independent models")
    print("\nTo use subject-wise splitting in your training:")
    print("  python src/train_subject_wise.py --config configs/config.yaml")
    print("  python src/train_subject_wise.py --config configs/config.yaml --fold subject_01")


if __name__ == "__main__":
    main()
