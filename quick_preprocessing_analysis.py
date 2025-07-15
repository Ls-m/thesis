#!/usr/bin/env python3
"""
Quick Preprocessing Analysis Report Generator (Subset of Subjects)

This script generates a preprocessing analysis report using a subset of subjects
to demonstrate the methodology and provide faster results.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, entropy
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import data utilities
sys.path.append('src')
from data_utils import DataPreprocessor

class QuickPreprocessingAnalysisReport:
    """Generate quick preprocessing analysis report with subset of subjects."""
    
    def __init__(self, config_path='configs/config.yaml', csv_folder='csv', max_subjects=5):
        """Initialize the analysis report generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.csv_folder = csv_folder
        self.max_subjects = max_subjects
        self.preprocessor = DataPreprocessor(self.config)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_raw_data_subset(self):
        """Load raw data for a subset of subjects."""
        print(f"Loading raw data for up to {self.max_subjects} subjects...")
        raw_data = {}
        
        files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        files = files[:self.max_subjects]  # Limit to first N subjects
        
        for filename in files:
            subject_id = filename.replace('.csv', '')
            
            try:
                # Read the CSV file
                df = pd.read_csv(os.path.join(self.csv_folder, filename))
                
                # Handle concatenated header format
                if len(df.columns) == 1:
                    df = pd.read_csv(os.path.join(self.csv_folder, filename), header=None)
                    df.columns = ['Time', 'AIRFLOW', 'NASAL CANULA', 'CHEST', 'ABDOMEN', 'PPG'][:len(df.columns)]
                
                # Skip the first row (sampling rates) and convert to numeric
                df = df.iloc[1:].reset_index(drop=True)
                
                # Handle single column data
                if len(df.columns) == 1:
                    data_rows = []
                    for idx, row in df.iterrows():
                        row_str = str(row.iloc[0])
                        parts = row_str.split('\t') if '\t' in row_str else row_str.split()
                        if len(parts) >= 6:
                            data_rows.append(parts[:6])
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=['Time', 'AIRFLOW', 'NASAL CANULA', 'CHEST', 'ABDOMEN', 'PPG'])
                
                # Convert to numeric
                for col in df.columns:
                    if col != 'Time':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Extract signals
                if 'PPG' in df.columns and 'NASAL CANULA' in df.columns:
                    ppg_signal = df['PPG'].values
                    resp_signal = df['NASAL CANULA'].values
                    
                    # Sample data to reduce processing time (take every 10th point)
                    sample_rate = 10
                    ppg_signal = ppg_signal[::sample_rate]
                    resp_signal = resp_signal[::sample_rate]
                    
                    # Remove NaN values
                    valid_indices = ~(np.isnan(ppg_signal) | np.isnan(resp_signal))
                    ppg_signal = ppg_signal[valid_indices]
                    resp_signal = resp_signal[valid_indices]
                    
                    if len(ppg_signal) > 0:
                        raw_data[subject_id] = {
                            'ppg': ppg_signal,
                            'resp': resp_signal
                        }
                        print(f"Loaded {subject_id}: PPG={len(ppg_signal)} samples, RESP={len(resp_signal)} samples")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                    
        print(f"Loaded raw data for {len(raw_data)} subjects")
        return raw_data
    
    def load_preprocessed_data_subset(self, raw_subjects):
        """Load preprocessed data for the same subset of subjects."""
        print("Loading preprocessed data...")
        
        # Temporarily modify config to process only our subset
        temp_csv_folder = 'temp_csv_subset'
        os.makedirs(temp_csv_folder, exist_ok=True)
        
        # Copy subset files
        for subject_id in raw_subjects:
            src_file = os.path.join(self.csv_folder, f"{subject_id}.csv")
            dst_file = os.path.join(temp_csv_folder, f"{subject_id}.csv")
            if os.path.exists(src_file):
                import shutil
                shutil.copy2(src_file, dst_file)
        
        try:
            processed_data = self.preprocessor.prepare_dataset(temp_csv_folder)
            
            # Convert segmented data back to continuous signals
            preprocessed_data = {}
            for subject_id, (ppg_segments, resp_segments) in processed_data.items():
                ppg_flat = ppg_segments.flatten()
                resp_flat = resp_segments.flatten()
                
                preprocessed_data[subject_id] = {
                    'ppg': ppg_flat,
                    'resp': resp_flat,
                    'segments': (ppg_segments, resp_segments)
                }
                
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_csv_folder, ignore_errors=True)
            
        print(f"Loaded preprocessed data for {len(preprocessed_data)} subjects")
        return preprocessed_data
    
    def calculate_statistics(self, data, signal_type='ppg'):
        """Calculate statistical summaries for each subject."""
        stats_dict = {}
        
        for subject_id, signals in data.items():
            signal = signals[signal_type]
            if len(signal) > 0:
                stats_dict[subject_id] = {
                    'mean': np.mean(signal),
                    'std': np.std(signal),
                    'min': np.min(signal),
                    'max': np.max(signal),
                    'median': np.median(signal),
                    'q25': np.percentile(signal, 25),
                    'q75': np.percentile(signal, 75),
                    'skewness': stats.skew(signal),
                    'kurtosis': stats.kurtosis(signal),
                    'length': len(signal)
                }
        
        return stats_dict
    
    def calculate_inter_subject_variability(self, stats_dict):
        """Calculate inter-subject variability metrics."""
        metrics = ['mean', 'std', 'min', 'max', 'median']
        variability = {}
        
        for metric in metrics:
            values = [stats_dict[subj][metric] for subj in stats_dict.keys()]
            variability[f'{metric}_std'] = np.std(values)
            variability[f'{metric}_cv'] = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else np.inf
            variability[f'{metric}_range'] = np.max(values) - np.min(values)
        
        return variability
    
    def calculate_distribution_distances(self, raw_data, preprocessed_data, signal_type='ppg'):
        """Calculate distance metrics between subject distributions."""
        subjects = list(set(raw_data.keys()) & set(preprocessed_data.keys()))
        
        wasserstein_distances = {}
        
        for i, subj1 in enumerate(subjects):
            for j, subj2 in enumerate(subjects[i+1:], i+1):
                try:
                    # Sample data for distance calculation if too large
                    raw_sig1 = raw_data[subj1][signal_type]
                    raw_sig2 = raw_data[subj2][signal_type]
                    prep_sig1 = preprocessed_data[subj1][signal_type]
                    prep_sig2 = preprocessed_data[subj2][signal_type]
                    
                    # Sample if too large
                    max_samples = 10000
                    if len(raw_sig1) > max_samples:
                        raw_sig1 = np.random.choice(raw_sig1, max_samples, replace=False)
                    if len(raw_sig2) > max_samples:
                        raw_sig2 = np.random.choice(raw_sig2, max_samples, replace=False)
                    if len(prep_sig1) > max_samples:
                        prep_sig1 = np.random.choice(prep_sig1, max_samples, replace=False)
                    if len(prep_sig2) > max_samples:
                        prep_sig2 = np.random.choice(prep_sig2, max_samples, replace=False)
                    
                    raw_wasserstein = wasserstein_distance(raw_sig1, raw_sig2)
                    prep_wasserstein = wasserstein_distance(prep_sig1, prep_sig2)
                    
                    pair_key = f"{subj1}_vs_{subj2}"
                    wasserstein_distances[pair_key] = {
                        'raw': raw_wasserstein,
                        'preprocessed': prep_wasserstein,
                        'improvement': (raw_wasserstein - prep_wasserstein) / raw_wasserstein if raw_wasserstein > 0 else 0
                    }
                except Exception as e:
                    print(f"Error calculating distance for {subj1} vs {subj2}: {e}")
        
        return wasserstein_distances
    
    def create_summary_plots(self, raw_data, preprocessed_data, output_dir):
        """Create summary visualization plots."""
        subjects = list(raw_data.keys())
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Preprocessing Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. PPG Raw vs Preprocessed Distributions
        ax1 = axes[0, 0]
        for subject in subjects:
            raw_signal = raw_data[subject]['ppg']
            if len(raw_signal) > 5000:
                raw_signal = np.random.choice(raw_signal, 5000, replace=False)
            ax1.hist(raw_signal, bins=30, alpha=0.3, label=f'{subject} (raw)', density=True)
        ax1.set_title('PPG Raw Data Distributions')
        ax1.set_xlabel('Signal Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        ax2 = axes[0, 1]
        for subject in subjects:
            if subject in preprocessed_data:
                prep_signal = preprocessed_data[subject]['ppg']
                if len(prep_signal) > 5000:
                    prep_signal = np.random.choice(prep_signal, 5000, replace=False)
                ax2.hist(prep_signal, bins=30, alpha=0.3, label=f'{subject} (prep)', density=True)
        ax2.set_title('PPG Preprocessed Data Distributions')
        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 2. Respiratory Raw vs Preprocessed Distributions
        ax3 = axes[0, 2]
        for subject in subjects:
            raw_signal = raw_data[subject]['resp']
            if len(raw_signal) > 5000:
                raw_signal = np.random.choice(raw_signal, 5000, replace=False)
            ax3.hist(raw_signal, bins=30, alpha=0.3, label=f'{subject} (raw)', density=True)
        ax3.set_title('Respiratory Raw Data Distributions')
        ax3.set_xlabel('Signal Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        ax4 = axes[1, 0]
        for subject in subjects:
            if subject in preprocessed_data:
                prep_signal = preprocessed_data[subject]['resp']
                if len(prep_signal) > 5000:
                    prep_signal = np.random.choice(prep_signal, 5000, replace=False)
                ax4.hist(prep_signal, bins=30, alpha=0.3, label=f'{subject} (prep)', density=True)
        ax4.set_title('Respiratory Preprocessed Data Distributions')
        ax4.set_xlabel('Signal Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        # 3. Standard Deviation Comparison
        ax5 = axes[1, 1]
        raw_ppg_stats = self.calculate_statistics(raw_data, 'ppg')
        prep_ppg_stats = self.calculate_statistics(preprocessed_data, 'ppg')
        
        raw_stds = [raw_ppg_stats[subj]['std'] for subj in subjects if subj in raw_ppg_stats]
        prep_stds = [prep_ppg_stats[subj]['std'] for subj in subjects if subj in prep_ppg_stats]
        
        x = np.arange(len(subjects))
        width = 0.35
        ax5.bar(x - width/2, raw_stds, width, label='Raw', alpha=0.7)
        ax5.bar(x + width/2, prep_stds, width, label='Preprocessed', alpha=0.7)
        ax5.set_title('PPG Standard Deviation Comparison')
        ax5.set_xlabel('Subjects')
        ax5.set_ylabel('Standard Deviation')
        ax5.set_xticks(x)
        ax5.set_xticklabels(subjects, rotation=45)
        ax5.legend()
        
        # 4. Distance Analysis
        ax6 = axes[1, 2]
        ppg_distances = self.calculate_distribution_distances(raw_data, preprocessed_data, 'ppg')
        
        if ppg_distances:
            improvements = [v['improvement'] for v in ppg_distances.values()]
            ax6.hist(improvements, bins=10, alpha=0.7, color='green')
            ax6.axvline(np.mean(improvements), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(improvements):.3f}')
            ax6.set_title('PPG Distance Improvement Distribution')
            ax6.set_xlabel('Improvement Ratio')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'quick_preprocessing_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return ppg_distances
    
    def generate_quick_report(self, output_dir='quick_preprocessing_report'):
        """Generate the quick preprocessing analysis report."""
        print("Generating Quick Preprocessing Analysis Report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        raw_data = self.load_raw_data_subset()
        preprocessed_data = self.load_preprocessed_data_subset(raw_data.keys())
        
        # Generate visualizations and analysis
        print("Creating visualizations...")
        ppg_distances = self.create_summary_plots(raw_data, preprocessed_data, output_dir)
        
        # Calculate statistics
        raw_ppg_stats = self.calculate_statistics(raw_data, 'ppg')
        prep_ppg_stats = self.calculate_statistics(preprocessed_data, 'ppg')
        raw_resp_stats = self.calculate_statistics(raw_data, 'resp')
        prep_resp_stats = self.calculate_statistics(preprocessed_data, 'resp')
        
        raw_ppg_var = self.calculate_inter_subject_variability(raw_ppg_stats)
        prep_ppg_var = self.calculate_inter_subject_variability(prep_ppg_stats)
        
        # Generate text report
        report_path = os.path.join(output_dir, 'quick_preprocessing_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Quick Preprocessing Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"This is a quick analysis using {len(raw_data)} subjects to demonstrate the preprocessing ")
            f.write("effectiveness methodology. This provides faster results while the full analysis runs.\n\n")
            
            f.write("## Subjects Analyzed\n\n")
            for subject_id in raw_data.keys():
                ppg_len = len(raw_data[subject_id]['ppg'])
                resp_len = len(raw_data[subject_id]['resp'])
                f.write(f"- **{subject_id}**: PPG={ppg_len:,} samples, RESP={resp_len:,} samples\n")
            f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # PPG Analysis
            f.write("### PPG Signal Analysis\n\n")
            f.write("#### Statistical Summary\n")
            f.write("| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std | Std Reduction |\n")
            f.write("|---------|----------|---------|-----------|----------|---------------|\n")
            
            for subject in raw_data.keys():
                if subject in raw_ppg_stats and subject in prep_ppg_stats:
                    raw_std = raw_ppg_stats[subject]['std']
                    prep_std = prep_ppg_stats[subject]['std']
                    std_reduction = (raw_std - prep_std) / raw_std * 100 if raw_std > 0 else 0
                    
                    f.write(f"| {subject} | {raw_ppg_stats[subject]['mean']:.2f} | "
                           f"{raw_std:.2f} | {prep_ppg_stats[subject]['mean']:.2f} | "
                           f"{prep_std:.2f} | {std_reduction:.1f}% |\n")
            
            f.write("\n#### Inter-Subject Variability\n")
            f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            
            for metric in ['mean_std', 'std_std', 'median_std']:
                if metric in raw_ppg_var and metric in prep_ppg_var:
                    raw_val = raw_ppg_var[metric]
                    prep_val = prep_ppg_var[metric]
                    improvement = (raw_val - prep_val) / raw_val * 100 if raw_val != 0 else 0
                    f.write(f"| {metric} | {raw_val:.4f} | {prep_val:.4f} | {improvement:.2f}% |\n")
            
            # Distance Analysis
            if ppg_distances:
                improvements = [v['improvement'] for v in ppg_distances.values()]
                mean_improvement = np.mean(improvements)
                positive_improvements = sum(1 for imp in improvements if imp > 0)
                total_pairs = len(improvements)
                
                f.write(f"\n#### Distance Analysis\n")
                f.write(f"- **Mean Wasserstein Distance Improvement**: {mean_improvement:.4f} ({mean_improvement*100:.2f}%)\n")
                f.write(f"- **Subject pairs with improvement**: {positive_improvements}/{total_pairs} ({positive_improvements/total_pairs*100:.1f}%)\n")
                
                if mean_improvement > 0:
                    f.write(f"\n✅ **Conclusion**: Preprocessing shows **POSITIVE EFFECT** in reducing inter-subject variability.\n")
                else:
                    f.write(f"\n❌ **Conclusion**: Preprocessing shows **LIMITED EFFECT** in reducing inter-subject variability.\n")
            
            f.write(f"\n## Methodology Note\n\n")
            f.write(f"This quick analysis uses:\n")
            f.write(f"- Subset of {len(raw_data)} subjects (out of 28 total)\n")
            f.write(f"- Downsampled data (every 10th point) for faster processing\n")
            f.write(f"- Core statistical and distance metrics\n")
            f.write(f"- The full analysis will provide comprehensive results for all subjects\n\n")
            
            f.write("---\n")
            f.write(f"*Quick report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"Quick report generated successfully in '{output_dir}' directory!")
        print(f"Files created:")
        print(f"- {report_path}")
        print(f"- {os.path.join(output_dir, 'quick_preprocessing_analysis.png')}")
        
        return {
            'raw_data': raw_data,
            'preprocessed_data': preprocessed_data,
            'ppg_distances': ppg_distances
        }


def main():
    """Main function to generate the quick report."""
    try:
        # Initialize the report generator
        report_generator = QuickPreprocessingAnalysisReport(max_subjects=5)
        
        # Generate the quick report
        results = report_generator.generate_quick_report()
        
        print("\n" + "="*60)
        print("QUICK PREPROCESSING ANALYSIS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating quick report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
