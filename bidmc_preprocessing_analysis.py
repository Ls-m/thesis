#!/usr/bin/env python3
"""
BIDMC Preprocessing Analysis Report Generator

This script generates a comprehensive report comparing data distributions 
before and after preprocessing for the BIDMC dataset to assess inter-subject 
variability reduction.
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

class BIDMCPreprocessingAnalysisReport:
    """Generate comprehensive preprocessing analysis report for BIDMC dataset."""
    
    def __init__(self, config_path='configs/config.yaml', bidmc_folder='src/bidmc', max_subjects=10):
        """Initialize the analysis report generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bidmc_folder = bidmc_folder
        self.max_subjects = max_subjects
        
        # Create a modified preprocessor for BIDMC data
        # Update config for BIDMC format
        self.config['data']['input_column'] = 'PG'  # Note: PG not PPG in BIDMC
        self.config['data']['target_column'] = 'NASAL CANULA'
        self.config['data']['sampling_rate'] = 125  # BIDMC sampling rate
        
        # Adjust downsampling for BIDMC (125 Hz -> 62.5 Hz to avoid decimation issues)
        self.config['preprocessing']['downsample']['target_rate'] = 62  # Close to 64 but compatible with 125
        
        # Adjust segment length for new sampling rate
        # Original: 2048 samples at 256 Hz = 8 seconds
        # For 125 Hz: 8 seconds = 1000 samples, for 62 Hz: 8 seconds = 496 samples
        self.config['data']['segment_length'] = 1000  # 8 seconds at 125 Hz
        
        self.preprocessor = DataPreprocessor(self.config)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_bidmc_raw_data(self):
        """Load raw BIDMC data."""
        print(f"Loading BIDMC raw data for up to {self.max_subjects} subjects...")
        raw_data = {}
        
        # Get list of BIDMC files
        files = [f for f in os.listdir(self.bidmc_folder) if f.endswith('.csv')]
        files = sorted(files)[:self.max_subjects]  # Limit and sort for consistency
        
        for filename in files:
            subject_id = filename.replace('.csv', '')
            
            try:
                # Read CSV file (comma-separated)
                df = pd.read_csv(os.path.join(self.bidmc_folder, filename))
                
                print(f"Loaded {filename}: shape={df.shape}, columns={df.columns.tolist()}")
                
                # Skip the first row (sampling rates) and convert to numeric
                df = df.iloc[1:].reset_index(drop=True)
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Extract signals using BIDMC column names
                # Handle both 'PG' and 'PPG' column names
                ppg_col = None
                resp_col = None
                
                for col in df.columns:
                    if col in ['PG', 'PPG']:
                        ppg_col = col
                    elif col == 'NASAL CANULA':
                        resp_col = col
                
                if ppg_col and resp_col:
                    ppg_signal = df[ppg_col].values
                    resp_signal = df[resp_col].values
                    
                    # Remove NaN values
                    valid_indices = ~(np.isnan(ppg_signal) | np.isnan(resp_signal))
                    ppg_signal = ppg_signal[valid_indices]
                    resp_signal = resp_signal[valid_indices]
                    
                    if len(ppg_signal) > 1000:  # Need minimum data
                        raw_data[subject_id] = {
                            'ppg': ppg_signal,
                            'resp': resp_signal
                        }
                        print(f"Processed {subject_id}: PPG={len(ppg_signal)} samples, RESP={len(resp_signal)} samples")
                else:
                    print(f"Warning: Expected columns not found in {filename}. Available: {df.columns.tolist()}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                    
        print(f"Loaded BIDMC raw data for {len(raw_data)} subjects")
        return raw_data
    
    def load_bidmc_preprocessed_data(self, raw_subjects):
        """Load preprocessed BIDMC data."""
        print("Loading BIDMC preprocessed data...")
        
        # Create temporary CSV folder with BIDMC format
        temp_csv_folder = 'temp_bidmc_subset'
        os.makedirs(temp_csv_folder, exist_ok=True)
        
        # Copy and convert subset files to expected format
        for subject_id in raw_subjects:
            src_file = os.path.join(self.bidmc_folder, f"{subject_id}.csv")
            dst_file = os.path.join(temp_csv_folder, f"{subject_id}.csv")
            
            if os.path.exists(src_file):
                # Read original BIDMC file
                df = pd.read_csv(src_file)
                
                # Rename column to match config expectation
                if 'PG' in df.columns:
                    df = df.rename(columns={'PG': 'PPG'})  # Rename for preprocessor
                
                # Save in temp folder
                df.to_csv(dst_file, index=False)
        
        try:
            # Temporarily update config for preprocessing
            original_input_col = self.config['data']['input_column']
            self.config['data']['input_column'] = 'PPG'  # Use PPG for preprocessing
            
            processed_data = self.preprocessor.prepare_dataset(temp_csv_folder)
            
            # Restore original config
            self.config['data']['input_column'] = original_input_col
            
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
            
        print(f"Loaded BIDMC preprocessed data for {len(preprocessed_data)} subjects")
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
        """Calculate KL divergence and Wasserstein distance between subject distributions."""
        subjects = list(set(raw_data.keys()) & set(preprocessed_data.keys()))
        
        kl_divergences = {}
        wasserstein_distances = {}
        
        for i, subj1 in enumerate(subjects):
            for j, subj2 in enumerate(subjects[i+1:], i+1):
                # Sample data for distance calculation if too large
                raw_sig1 = raw_data[subj1][signal_type]
                raw_sig2 = raw_data[subj2][signal_type]
                prep_sig1 = preprocessed_data[subj1][signal_type]
                prep_sig2 = preprocessed_data[subj2][signal_type]
                
                # Sample if too large for performance
                max_samples = 10000
                if len(raw_sig1) > max_samples:
                    raw_sig1 = np.random.choice(raw_sig1, max_samples, replace=False)
                if len(raw_sig2) > max_samples:
                    raw_sig2 = np.random.choice(raw_sig2, max_samples, replace=False)
                if len(prep_sig1) > max_samples:
                    prep_sig1 = np.random.choice(prep_sig1, max_samples, replace=False)
                if len(prep_sig2) > max_samples:
                    prep_sig2 = np.random.choice(prep_sig2, max_samples, replace=False)
                
                # Calculate Wasserstein distance
                try:
                    raw_wasserstein = wasserstein_distance(raw_sig1, raw_sig2)
                    prep_wasserstein = wasserstein_distance(prep_sig1, prep_sig2)
                    
                    pair_key = f"{subj1}_vs_{subj2}"
                    wasserstein_distances[pair_key] = {
                        'raw': raw_wasserstein,
                        'preprocessed': prep_wasserstein,
                        'improvement': (raw_wasserstein - prep_wasserstein) / raw_wasserstein if raw_wasserstein > 0 else 0
                    }
                except Exception as e:
                    print(f"Error calculating Wasserstein distance for {subj1} vs {subj2}: {e}")
                
                # Calculate KL divergence
                try:
                    bins = 50
                    raw_hist1, _ = np.histogram(raw_sig1, bins=bins, density=True)
                    raw_hist2, _ = np.histogram(raw_sig2, bins=bins, density=True)
                    prep_hist1, _ = np.histogram(prep_sig1, bins=bins, density=True)
                    prep_hist2, _ = np.histogram(prep_sig2, bins=bins, density=True)
                    
                    # Add small epsilon to avoid log(0)
                    eps = 1e-10
                    raw_hist1 += eps
                    raw_hist2 += eps
                    prep_hist1 += eps
                    prep_hist2 += eps
                    
                    # Normalize
                    raw_hist1 /= raw_hist1.sum()
                    raw_hist2 /= raw_hist2.sum()
                    prep_hist1 /= prep_hist1.sum()
                    prep_hist2 /= prep_hist2.sum()
                    
                    raw_kl = entropy(raw_hist1, raw_hist2)
                    prep_kl = entropy(prep_hist1, prep_hist2)
                    
                    pair_key = f"{subj1}_vs_{subj2}"
                    kl_divergences[pair_key] = {
                        'raw': raw_kl,
                        'preprocessed': prep_kl,
                        'improvement': (raw_kl - prep_kl) / raw_kl if raw_kl > 0 else 0
                    }
                except Exception as e:
                    print(f"Error calculating KL divergence for {subj1} vs {subj2}: {e}")
        
        return kl_divergences, wasserstein_distances
    
    def create_bidmc_visualization(self, raw_data, preprocessed_data, output_dir):
        """Create BIDMC-specific visualization."""
        subjects = list(raw_data.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BIDMC Dataset: Preprocessing Analysis', fontsize=16, fontweight='bold')
        
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
        kl_divs, wasserstein_dists = self.calculate_distribution_distances(raw_data, preprocessed_data, 'ppg')
        
        if wasserstein_dists:
            improvements = [v['improvement'] for v in wasserstein_dists.values()]
            ax6.hist(improvements, bins=10, alpha=0.7, color='green')
            ax6.axvline(np.mean(improvements), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(improvements):.3f}')
            ax6.set_title('PPG Distance Improvement Distribution')
            ax6.set_xlabel('Improvement Ratio')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'bidmc_preprocessing_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return kl_divs, wasserstein_dists
    
    def generate_bidmc_report(self, output_dir='bidmc_preprocessing_report'):
        """Generate the BIDMC preprocessing analysis report."""
        print("="*60)
        print("BIDMC PREPROCESSING ANALYSIS REPORT")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        raw_data = self.load_bidmc_raw_data()
        
        if len(raw_data) < 2:
            print("Error: Need at least 2 subjects for comparison")
            return
        
        preprocessed_data = self.load_bidmc_preprocessed_data(raw_data.keys())
        
        # Generate visualizations and analysis
        print("Creating visualizations...")
        kl_divs, wasserstein_dists = self.create_bidmc_visualization(raw_data, preprocessed_data, output_dir)
        
        # Calculate statistics
        raw_ppg_stats = self.calculate_statistics(raw_data, 'ppg')
        prep_ppg_stats = self.calculate_statistics(preprocessed_data, 'ppg')
        raw_resp_stats = self.calculate_statistics(raw_data, 'resp')
        prep_resp_stats = self.calculate_statistics(preprocessed_data, 'resp')
        
        raw_ppg_var = self.calculate_inter_subject_variability(raw_ppg_stats)
        prep_ppg_var = self.calculate_inter_subject_variability(prep_ppg_stats)
        raw_resp_var = self.calculate_inter_subject_variability(raw_resp_stats)
        prep_resp_var = self.calculate_inter_subject_variability(prep_resp_stats)
        
        # Generate text report
        report_path = os.path.join(output_dir, 'bidmc_preprocessing_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# BIDMC Dataset Preprocessing Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"This report analyzes the effectiveness of the preprocessing pipeline in reducing inter-subject variability ")
            f.write(f"for the BIDMC dataset. The analysis uses {len(raw_data)} subjects and compares data distributions ")
            f.write(f"before and after preprocessing.\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write("- **Dataset**: BIDMC (Beth Israel Deaconess Medical Center)\n")
            f.write(f"- **Subjects analyzed**: {len(raw_data)} out of 53 total\n")
            f.write("- **Sampling rate**: 125 Hz\n")
            f.write("- **Signals**: PPG (PG) and Nasal Cannula respiratory signal\n")
            f.write("- **Format**: Comma-separated CSV files\n\n")
            
            f.write("## Subjects Analyzed\n\n")
            for subject_id in sorted(raw_data.keys()):
                ppg_len = len(raw_data[subject_id]['ppg'])
                resp_len = len(raw_data[subject_id]['resp'])
                f.write(f"- **{subject_id}**: PPG={ppg_len:,} samples, RESP={resp_len:,} samples\n")
            f.write("\n")
            
            f.write("## Preprocessing Pipeline\n\n")
            f.write("The preprocessing pipeline includes:\n")
            f.write("1. **Bandpass Filtering**: 0.05-2.0 Hz with 2nd order Butterworth filter\n")
            f.write("2. **Downsampling**: From 125 Hz to 64 Hz\n")
            f.write("3. **Normalization**: Z-score normalization\n")
            f.write("4. **Segmentation**: 8-second segments with 50% overlap\n\n")
            
            f.write("## Key Findings\n\n")
            
            # PPG Analysis
            f.write("### PPG Signal Analysis\n\n")
            f.write("#### Statistical Summary\n")
            f.write("| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std | Std Reduction |\n")
            f.write("|---------|----------|---------|-----------|----------|---------------|\n")
            
            for subject in sorted(raw_data.keys()):
                if subject in raw_ppg_stats and subject in prep_ppg_stats:
                    raw_std = raw_ppg_stats[subject]['std']
                    prep_std = prep_ppg_stats[subject]['std']
                    std_reduction = (raw_std - prep_std) / raw_std * 100 if raw_std > 0 else 0
                    
                    f.write(f"| {subject} | {raw_ppg_stats[subject]['mean']:.4f} | "
                           f"{raw_std:.4f} | {prep_ppg_stats[subject]['mean']:.4f} | "
                           f"{prep_std:.4f} | {std_reduction:.1f}% |\n")
            
            f.write("\n#### Inter-Subject Variability (PPG)\n")
            f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            
            for metric in ['mean_std', 'std_std', 'median_std']:
                if metric in raw_ppg_var and metric in prep_ppg_var:
                    raw_val = raw_ppg_var[metric]
                    prep_val = prep_ppg_var[metric]
                    improvement = (raw_val - prep_val) / raw_val * 100 if raw_val != 0 else 0
                    f.write(f"| {metric} | {raw_val:.4f} | {prep_val:.4f} | {improvement:.2f}% |\n")
            
            # Respiratory Analysis
            f.write("\n### Respiratory Signal Analysis\n\n")
            f.write("#### Inter-Subject Variability (Respiratory)\n")
            f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            
            for metric in ['mean_std', 'std_std', 'median_std']:
                if metric in raw_resp_var and metric in prep_resp_var:
                    raw_val = raw_resp_var[metric]
                    prep_val = prep_resp_var[metric]
                    improvement = (raw_val - prep_val) / raw_val * 100 if raw_val != 0 else 0
                    f.write(f"| {metric} | {raw_val:.4f} | {prep_val:.4f} | {improvement:.2f}% |\n")
            
            # Distance Analysis
            if wasserstein_dists:
                improvements = [v['improvement'] for v in wasserstein_dists.values()]
                mean_improvement = np.mean(improvements)
                positive_improvements = sum(1 for imp in improvements if imp > 0)
                total_pairs = len(improvements)
                
                f.write(f"\n### Distance Analysis\n")
                f.write(f"- **Mean Wasserstein Distance Improvement**: {mean_improvement:.4f} ({mean_improvement*100:.2f}%)\n")
                f.write(f"- **Subject pairs with improvement**: {positive_improvements}/{total_pairs} ({positive_improvements/total_pairs*100:.1f}%)\n")
                
                if mean_improvement > 0:
                    f.write(f"\n✅ **Conclusion**: Preprocessing shows **POSITIVE EFFECT** in reducing inter-subject variability for BIDMC dataset.\n")
                else:
                    f.write(f"\n⚠️ **Conclusion**: Preprocessing shows **MIXED RESULTS** for BIDMC dataset.\n")
            
            f.write(f"\n## Comparison with Other Datasets\n\n")
            f.write(f"This BIDMC analysis can be compared with the main dataset analysis to understand:\n")
            f.write(f"- Dataset-specific preprocessing effectiveness\n")
            f.write(f"- Generalizability of preprocessing approaches\n")
            f.write(f"- Optimal preprocessing parameters for different data sources\n\n")
            
            f.write("---\n")
            f.write(f"*BIDMC report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"\nBIDMC analysis completed!")
        print(f"Subjects analyzed: {sorted(raw_data.keys())}")
        print(f"Files generated:")
        print(f"- {report_path}")
        print(f"- {os.path.join(output_dir, 'bidmc_preprocessing_analysis.png')}")
        
        if wasserstein_dists:
            improvements = [v['improvement'] for v in wasserstein_dists.values()]
            mean_improvement = np.mean(improvements)
            print(f"\nKey Result: {mean_improvement*100:.2f}% average improvement in inter-subject distance")
            
            if mean_improvement > 0:
                print("✅ Preprocessing shows positive effect for BIDMC dataset")
            else:
                print("⚠️ Preprocessing shows mixed results for BIDMC dataset")
        
        print("="*60)
        
        return {
            'raw_data': raw_data,
            'preprocessed_data': preprocessed_data,
            'wasserstein_distances': wasserstein_dists,
            'kl_divergences': kl_divs
        }


def main():
    """Main function to generate the BIDMC report."""
    try:
        # Initialize the report generator
        report_generator = BIDMCPreprocessingAnalysisReport(max_subjects=53)
        
        # Generate the BIDMC report
        results = report_generator.generate_bidmc_report()
        
    except Exception as e:
        print(f"Error generating BIDMC report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
