#!/usr/bin/env python3
"""
Standalone script for generating data distribution reports for PPG respiratory datasets.
"""

import os
import yaml
import argparse
from data_distribution_analyzer import DataDistributionAnalyzer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate data distribution analysis report for PPG respiratory datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for capno dataset
  python src/generate_data_report.py --dataset capno --config configs/improved_config.yaml
  
  # Generate report for bidmc dataset
  python src/generate_data_report.py --dataset bidmc --config configs/improved_config.yaml
  
  # Generate report for custom dataset path
  python src/generate_data_report.py --dataset-path src/my_data --config configs/improved_config.yaml
  
  # Generate report with custom output directory
  python src/generate_data_report.py --dataset capno --output-dir my_analysis_report
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/improved_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['bidmc', 'csv', 'capno'], 
                       help='Dataset to analyze: "bidmc", "csv", or "capno"')
    parser.add_argument('--dataset-path', type=str,
                       help='Custom path to dataset folder (overrides --dataset)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for the report (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
        print(f"Using custom dataset path: {dataset_path}")
    elif args.dataset:
        if args.dataset == 'bidmc':
            dataset_path = 'src/bidmc'
        elif args.dataset == 'csv':
            dataset_path = 'src/csv'
        elif args.dataset == 'capno':
            dataset_path = 'src/capno'
        print(f"Using {args.dataset} dataset: {dataset_path}")
    else:
        # Use default from config
        dataset_path = config['data']['csv_folder']
        print(f"Using dataset from config: {dataset_path}")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        return 1
    
    # Update config with the dataset path
    config['data']['csv_folder'] = dataset_path
    
    # Initialize analyzer
    analyzer = DataDistributionAnalyzer(config)
    
    # Generate report
    print("Generating comprehensive data distribution report...")
    try:
        results = analyzer.generate_report(dataset_path, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA DISTRIBUTION ANALYSIS SUMMARY")
        print("="*60)
        
        raw_summary = results['raw_data_summary']
        processed_summary = results['processed_data_summary']
        
        print(f"Dataset: {dataset_path}")
        print(f"Total Subjects: {raw_summary['total_subjects']}")
        print(f"Successfully Processed: {processed_summary['subjects_successfully_processed']}")
        print(f"Total Duration: {raw_summary['total_duration_hours']:.1f} hours")
        print(f"Average Duration per Subject: {raw_summary['average_duration_minutes']:.1f} minutes")
        print(f"Total Segments Generated: {processed_summary['total_segments']}")
        print(f"Average Segments per Subject: {processed_summary['average_segments_per_subject']:.1f}")
        
        # Data quality
        good_quality_subjects = raw_summary['subjects_with_good_quality']
        total_subjects = raw_summary['total_subjects']
        quality_percentage = (good_quality_subjects / total_subjects) * 100 if total_subjects > 0 else 0
        print(f"Data Quality: {good_quality_subjects}/{total_subjects} subjects ({quality_percentage:.1f}%) have good quality")
        
        # Missing data
        avg_ppg_missing = raw_summary['average_ppg_nan_percentage']
        avg_resp_missing = raw_summary['average_resp_nan_percentage']
        print(f"Average Missing Data: PPG {avg_ppg_missing:.2f}%, Respiratory {avg_resp_missing:.2f}%")
        
        # Class imbalance
        imbalanced_subjects = processed_summary['subjects_with_potential_imbalance']
        print(f"Potential Class Imbalance: {imbalanced_subjects} subjects")
        
        print("\nGenerated Files:")
        for plot_file in results['generated_plots']:
            print(f"  - {os.path.basename(plot_file)}")
        
        output_dir = os.path.dirname(results['generated_plots'][0]) if results['generated_plots'] else 'unknown'
        print(f"\nReport saved to: {output_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
