#!/usr/bin/env python3
"""
Preprocessing Analysis Report Generator

This script generates a comprehensive report comparing data distributions 
before and after preprocessing to assess inter-subject variability reduction.
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
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import data utilities
sys.path.append('src')
from data_utils import DataPreprocessor

class PreprocessingAnalysisReport:
    """Generate comprehensive preprocessing analysis report."""
    
    def __init__(self, config_path='configs/config.yaml', csv_folder='csv'):
        """Initialize the analysis report generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.csv_folder = csv_folder
        self.preprocessor = DataPreprocessor(self.config)
        self.results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_raw_data(self):
        """Load raw data before preprocessing."""
        print("Loading raw data...")
        raw_data = {}
        
        for filename in os.listdir(self.csv_folder):
            if filename.endswith('.csv'):
                subject_id = filename.replace('.csv', '')
                
                try:
                    # Read the CSV file with proper handling
                    df = pd.read_csv(os.path.join(self.csv_folder, filename))
                    
                    # Check if columns are properly separated
                    if len(df.columns) == 1:
                        # If all data is in one column, we need to parse it differently
                        # Read without header and manually set column names
                        df = pd.read_csv(os.path.join(self.csv_folder, filename), header=None)
                        # Based on the head output, the columns should be:
                        # Time, AIRFLOW, NASAL CANULA, CHEST, ABDOMEN, PPG
                        expected_cols = ['Time', 'AIRFLOW', 'NASAL CANULA', 'CHEST', 'ABDOMEN', 'PPG']
                        
                        # Split the first row to get proper column names
                        first_row = str(df.iloc[0, 0])
                        if 'PPG' in first_row and 'NASAL CANULA' in first_row:
                            # Parse the concatenated header
                            df.columns = expected_cols[:len(df.columns)]
                    
                    # Skip the first row (sampling rates) and convert to numeric
                    df = df.iloc[1:].reset_index(drop=True)
                    
                    # Handle the case where data might be in a single column
                    if len(df.columns) == 1:
                        # Try to split the data
                        data_rows = []
                        for idx, row in df.iterrows():
                            row_str = str(row.iloc[0])
                            # Split by tabs or multiple spaces
                            parts = row_str.split('\t') if '\t' in row_str else row_str.split()
                            if len(parts) >= 6:  # We need at least 6 columns
                                data_rows.append(parts[:6])
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=['Time', 'AIRFLOW', 'NASAL CANULA', 'CHEST', 'ABDOMEN', 'PPG'])
                    
                    # Convert to numeric
                    for col in df.columns:
                        if col != 'Time':  # Skip time column
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Extract signals using the correct column names
                    if 'PPG' in df.columns and 'NASAL CANULA' in df.columns:
                        ppg_signal = df['PPG'].values
                        resp_signal = df['NASAL CANULA'].values
                    else:
                        print(f"Warning: Expected columns not found in {filename}. Available columns: {df.columns.tolist()}")
                        continue
                    
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
    
    def load_preprocessed_data(self):
        """Load preprocessed data."""
        print("Loading preprocessed data...")
        processed_data = self.preprocessor.prepare_dataset(self.csv_folder)
        
        # Convert segmented data back to continuous signals for comparison
        preprocessed_data = {}
        for subject_id, (ppg_segments, resp_segments) in processed_data.items():
            # Flatten segments to get continuous signal approximation
            ppg_flat = ppg_segments.flatten()
            resp_flat = resp_segments.flatten()
            
            preprocessed_data[subject_id] = {
                'ppg': ppg_flat,
                'resp': resp_flat,
                'segments': (ppg_segments, resp_segments)
            }
            
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
        """Calculate KL divergence and Wasserstein distance between subject distributions."""
        subjects = list(set(raw_data.keys()) & set(preprocessed_data.keys()))
        
        kl_divergences = {}
        wasserstein_distances = {}
        
        for i, subj1 in enumerate(subjects):
            for j, subj2 in enumerate(subjects[i+1:], i+1):
                # Raw data comparison
                raw_sig1 = raw_data[subj1][signal_type]
                raw_sig2 = raw_data[subj2][signal_type]
                
                # Preprocessed data comparison
                prep_sig1 = preprocessed_data[subj1][signal_type]
                prep_sig2 = preprocessed_data[subj2][signal_type]
                
                # Calculate Wasserstein distance (more robust than KL divergence)
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
                
                # Calculate KL divergence (using histograms)
                try:
                    # Create histograms
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
    
    def create_distribution_plots(self, raw_data, preprocessed_data, signal_type='ppg', max_subjects=10):
        """Create distribution comparison plots."""
        subjects = list(raw_data.keys())[:max_subjects]  # Limit for readability
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{signal_type.upper()} Signal Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histograms comparison
        ax1 = axes[0, 0]
        for subject in subjects:
            raw_signal = raw_data[subject][signal_type]
            # Sample data for plotting if too large
            if len(raw_signal) > 10000:
                raw_signal = np.random.choice(raw_signal, 10000, replace=False)
            ax1.hist(raw_signal, bins=50, alpha=0.3, label=f'{subject} (raw)', density=True)
        
        ax1.set_title('Raw Data Distributions')
        ax1.set_xlabel('Signal Value')
        ax1.set_ylabel('Density')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Preprocessed histograms
        ax2 = axes[0, 1]
        for subject in subjects:
            if subject in preprocessed_data:
                prep_signal = preprocessed_data[subject][signal_type]
                if len(prep_signal) > 10000:
                    prep_signal = np.random.choice(prep_signal, 10000, replace=False)
                ax2.hist(prep_signal, bins=50, alpha=0.3, label=f'{subject} (prep)', density=True)
        
        ax2.set_title('Preprocessed Data Distributions')
        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('Density')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Box plots comparison
        ax3 = axes[1, 0]
        raw_data_list = []
        prep_data_list = []
        labels = []
        
        for subject in subjects:
            raw_signal = raw_data[subject][signal_type]
            if len(raw_signal) > 5000:
                raw_signal = np.random.choice(raw_signal, 5000, replace=False)
            raw_data_list.append(raw_signal)
            
            if subject in preprocessed_data:
                prep_signal = preprocessed_data[subject][signal_type]
                if len(prep_signal) > 5000:
                    prep_signal = np.random.choice(prep_signal, 5000, replace=False)
                prep_data_list.append(prep_signal)
            else:
                prep_data_list.append([])
            
            labels.append(subject)
        
        # Create box plots
        positions_raw = np.arange(1, len(subjects) * 2, 2)
        positions_prep = np.arange(2, len(subjects) * 2 + 1, 2)
        
        bp1 = ax3.boxplot(raw_data_list, positions=positions_raw, widths=0.6, 
                         patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
        bp2 = ax3.boxplot(prep_data_list, positions=positions_prep, widths=0.6,
                         patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
        
        ax3.set_title('Distribution Comparison: Raw vs Preprocessed')
        ax3.set_xlabel('Subjects')
        ax3.set_ylabel('Signal Value')
        ax3.set_xticks(np.arange(1.5, len(subjects) * 2, 2))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw', 'Preprocessed'])
        
        # 4. Variability comparison
        ax4 = axes[1, 1]
        raw_stats = self.calculate_statistics(raw_data, signal_type)
        prep_stats = self.calculate_statistics(preprocessed_data, signal_type)
        
        raw_stds = [raw_stats[subj]['std'] for subj in subjects if subj in raw_stats]
        prep_stds = [prep_stats[subj]['std'] for subj in subjects if subj in prep_stats]
        
        x = np.arange(len(subjects))
        width = 0.35
        
        ax4.bar(x - width/2, raw_stds, width, label='Raw', alpha=0.7, color='lightblue')
        ax4.bar(x + width/2, prep_stds, width, label='Preprocessed', alpha=0.7, color='lightcoral')
        
        ax4.set_title('Standard Deviation Comparison')
        ax4.set_xlabel('Subjects')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def create_distance_analysis_plots(self, kl_divergences, wasserstein_distances):
        """Create plots for distance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Inter-Subject Distance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Wasserstein distance comparison
        ax1 = axes[0, 0]
        if wasserstein_distances:
            raw_wasserstein = [v['raw'] for v in wasserstein_distances.values()]
            prep_wasserstein = [v['preprocessed'] for v in wasserstein_distances.values()]
            
            ax1.scatter(raw_wasserstein, prep_wasserstein, alpha=0.6)
            ax1.plot([min(raw_wasserstein), max(raw_wasserstein)], 
                    [min(raw_wasserstein), max(raw_wasserstein)], 'r--', label='y=x')
            ax1.set_xlabel('Raw Data Wasserstein Distance')
            ax1.set_ylabel('Preprocessed Data Wasserstein Distance')
            ax1.set_title('Wasserstein Distance: Raw vs Preprocessed')
            ax1.legend()
        
        # 2. KL divergence comparison
        ax2 = axes[0, 1]
        if kl_divergences:
            raw_kl = [v['raw'] for v in kl_divergences.values() if not np.isinf(v['raw'])]
            prep_kl = [v['preprocessed'] for v in kl_divergences.values() if not np.isinf(v['preprocessed'])]
            
            if raw_kl and prep_kl:
                ax2.scatter(raw_kl, prep_kl, alpha=0.6)
                ax2.plot([min(raw_kl), max(raw_kl)], 
                        [min(raw_kl), max(raw_kl)], 'r--', label='y=x')
                ax2.set_xlabel('Raw Data KL Divergence')
                ax2.set_ylabel('Preprocessed Data KL Divergence')
                ax2.set_title('KL Divergence: Raw vs Preprocessed')
                ax2.legend()
        
        # 3. Improvement distribution (Wasserstein)
        ax3 = axes[1, 0]
        if wasserstein_distances:
            improvements = [v['improvement'] for v in wasserstein_distances.values()]
            ax3.hist(improvements, bins=20, alpha=0.7, color='green')
            ax3.axvline(np.mean(improvements), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(improvements):.3f}')
            ax3.set_xlabel('Improvement Ratio')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Wasserstein Distance Improvement Distribution')
            ax3.legend()
        
        # 4. Improvement distribution (KL)
        ax4 = axes[1, 1]
        if kl_divergences:
            improvements = [v['improvement'] for v in kl_divergences.values() 
                          if not np.isinf(v['improvement'])]
            if improvements:
                ax4.hist(improvements, bins=20, alpha=0.7, color='orange')
                ax4.axvline(np.mean(improvements), color='red', linestyle='--',
                           label=f'Mean: {np.mean(improvements):.3f}')
                ax4.set_xlabel('Improvement Ratio')
                ax4.set_ylabel('Frequency')
                ax4.set_title('KL Divergence Improvement Distribution')
                ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_summary_statistics(self, raw_data, preprocessed_data):
        """Generate summary statistics for the report."""
        summary = {}
        
        for signal_type in ['ppg', 'resp']:
            raw_stats = self.calculate_statistics(raw_data, signal_type)
            prep_stats = self.calculate_statistics(preprocessed_data, signal_type)
            
            raw_variability = self.calculate_inter_subject_variability(raw_stats)
            prep_variability = self.calculate_inter_subject_variability(prep_stats)
            
            kl_div, wasserstein_dist = self.calculate_distribution_distances(
                raw_data, preprocessed_data, signal_type)
            
            summary[signal_type] = {
                'raw_stats': raw_stats,
                'preprocessed_stats': prep_stats,
                'raw_variability': raw_variability,
                'preprocessed_variability': prep_variability,
                'kl_divergences': kl_div,
                'wasserstein_distances': wasserstein_dist
            }
        
        return summary
    
    def create_report(self, output_dir='preprocessing_analysis_report'):
        """Generate the complete preprocessing analysis report."""
        print("Generating Preprocessing Analysis Report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        raw_data = self.load_raw_data()
        preprocessed_data = self.load_preprocessed_data()
        
        # Generate summary statistics
        summary = self.generate_summary_statistics(raw_data, preprocessed_data)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # PPG signal analysis
        ppg_fig = self.create_distribution_plots(raw_data, preprocessed_data, 'ppg')
        ppg_fig.savefig(os.path.join(output_dir, 'ppg_distribution_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close(ppg_fig)
        
        # Respiratory signal analysis
        resp_fig = self.create_distribution_plots(raw_data, preprocessed_data, 'resp')
        resp_fig.savefig(os.path.join(output_dir, 'resp_distribution_analysis.png'), 
                        dpi=300, bbox_inches='tight')
        plt.close(resp_fig)
        
        # Distance analysis for PPG
        ppg_distance_fig = self.create_distance_analysis_plots(
            summary['ppg']['kl_divergences'], 
            summary['ppg']['wasserstein_distances']
        )
        ppg_distance_fig.savefig(os.path.join(output_dir, 'ppg_distance_analysis.png'), 
                                dpi=300, bbox_inches='tight')
        plt.close(ppg_distance_fig)
        
        # Distance analysis for Respiratory
        resp_distance_fig = self.create_distance_analysis_plots(
            summary['resp']['kl_divergences'], 
            summary['resp']['wasserstein_distances']
        )
        resp_distance_fig.savefig(os.path.join(output_dir, 'resp_distance_analysis.png'), 
                                 dpi=300, bbox_inches='tight')
        plt.close(resp_distance_fig)
        
        # Generate text report
        self.generate_text_report(summary, output_dir)
        
        print(f"Report generated successfully in '{output_dir}' directory!")
        return summary
    
    def generate_text_report(self, summary, output_dir):
        """Generate detailed text report."""
        report_path = os.path.join(output_dir, 'preprocessing_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Preprocessing Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the effectiveness of the preprocessing pipeline in reducing inter-subject variability ")
            f.write("for PPG to respiratory waveform estimation. The analysis compares data distributions before and after ")
            f.write("preprocessing using statistical summaries, visualizations, and distance metrics.\n\n")
            
            f.write("## Preprocessing Pipeline\n\n")
            f.write("The preprocessing pipeline includes:\n")
            f.write("1. **Bandpass Filtering**: 0.05-2.0 Hz with 2nd order Butterworth filter\n")
            f.write("2. **Downsampling**: From 256 Hz to 64 Hz\n")
            f.write("3. **Normalization**: Z-score normalization\n")
            f.write("4. **Segmentation**: 8-second segments with 50% overlap\n\n")
            
            # Analysis for each signal type
            for signal_type in ['ppg', 'resp']:
                signal_name = 'PPG' if signal_type == 'ppg' else 'Respiratory'
                f.write(f"## {signal_name} Signal Analysis\n\n")
                
                # Statistical summaries
                f.write("### Statistical Summaries\n\n")
                f.write("#### Raw Data Statistics\n")
                f.write("| Subject | Mean | Std | Min | Max | Median | Skewness | Kurtosis |\n")
                f.write("|---------|------|-----|-----|-----|--------|----------|----------|\n")
                
                raw_stats = summary[signal_type]['raw_stats']
                for subject, stats in raw_stats.items():
                    f.write(f"| {subject} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} | "
                           f"{stats['skewness']:.4f} | {stats['kurtosis']:.4f} |\n")
                
                f.write("\n#### Preprocessed Data Statistics\n")
                f.write("| Subject | Mean | Std | Min | Max | Median | Skewness | Kurtosis |\n")
                f.write("|---------|------|-----|-----|-----|--------|----------|----------|\n")
                
                prep_stats = summary[signal_type]['preprocessed_stats']
                for subject, stats in prep_stats.items():
                    f.write(f"| {subject} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} | "
                           f"{stats['skewness']:.4f} | {stats['kurtosis']:.4f} |\n")
                
                # Inter-subject variability
                f.write("\n### Inter-Subject Variability Analysis\n\n")
                raw_var = summary[signal_type]['raw_variability']
                prep_var = summary[signal_type]['preprocessed_variability']
                
                f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
                f.write("|--------|----------|--------------|-------------|\n")
                
                for metric in ['mean_std', 'std_std', 'median_std']:
                    raw_val = raw_var[metric]
                    prep_val = prep_var[metric]
                    improvement = (raw_val - prep_val) / raw_val * 100 if raw_val != 0 else 0
                    f.write(f"| {metric} | {raw_val:.4f} | {prep_val:.4f} | {improvement:.2f}% |\n")
                
                # Distance metrics
                f.write("\n### Distance Metrics Analysis\n\n")
                wasserstein_dists = summary[signal_type]['wasserstein_distances']
                
                if wasserstein_dists:
                    improvements = [v['improvement'] for v in wasserstein_dists.values()]
                    mean_improvement = np.mean(improvements)
                    median_improvement = np.median(improvements)
                    positive_improvements = sum(1 for imp in improvements if imp > 0)
                    total_pairs = len(improvements)
                    
                    f.write(f"**Wasserstein Distance Analysis:**\n")
                    f.write(f"- Mean improvement: {mean_improvement:.4f} ({mean_improvement*100:.2f}%)\n")
                    f.write(f"- Median improvement: {median_improvement:.4f} ({median_improvement*100:.2f}%)\n")
                    f.write(f"- Pairs with improvement: {positive_improvements}/{total_pairs} ({positive_improvements/total_pairs*100:.1f}%)\n\n")
                
                kl_divs = summary[signal_type]['kl_divergences']
                if kl_divs:
                    kl_improvements = [v['improvement'] for v in kl_divs.values() 
                                     if not np.isinf(v['improvement'])]
                    if kl_improvements:
                        mean_kl_improvement = np.mean(kl_improvements)
                        median_kl_improvement = np.median(kl_improvements)
                        positive_kl_improvements = sum(1 for imp in kl_improvements if imp > 0)
                        total_kl_pairs = len(kl_improvements)
                        
                        f.write(f"**KL Divergence Analysis:**\n")
                        f.write(f"- Mean improvement: {mean_kl_improvement:.4f} ({mean_kl_improvement*100:.2f}%)\n")
                        f.write(f"- Median improvement: {median_kl_improvement:.4f} ({median_kl_improvement*100:.2f}%)\n")
                        f.write(f"- Pairs with improvement: {positive_kl_improvements}/{total_kl_pairs} ({positive_kl_improvements/total_kl_pairs*100:.1f}%)\n\n")
            
            # Overall conclusions
            f.write("## Key Findings and Conclusions\n\n")
            
            # Calculate overall effectiveness
            ppg_wasserstein = summary['ppg']['wasserstein_distances']
            resp_wasserstein = summary['resp']['wasserstein_distances']
            
            if ppg_wasserstein and resp_wasserstein:
                ppg_improvements = [v['improvement'] for v in ppg_wasserstein.values()]
                resp_improvements = [v['improvement'] for v in resp_wasserstein.values()]
                
                overall_ppg_improvement = np.mean(ppg_improvements)
                overall_resp_improvement = np.mean(resp_improvements)
                
                f.write(f"### Preprocessing Effectiveness\n\n")
                f.write(f"- **PPG Signal**: Average {overall_ppg_improvement*100:.2f}% reduction in inter-subject distance\n")
                f.write(f"- **Respiratory Signal**: Average {overall_resp_improvement*100:.2f}% reduction in inter-subject distance\n\n")
                
                if overall_ppg_improvement > 0 and overall_resp_improvement > 0:
                    f.write("✅ **Conclusion**: The preprocessing pipeline is **EFFECTIVE** in reducing inter-subject variability.\n\n")
                elif overall_ppg_improvement > 0 or overall_resp_improvement > 0:
                    f.write("⚠️ **Conclusion**: The preprocessing pipeline shows **MIXED RESULTS** - effective for some signals but not others.\n\n")
                else:
                    f.write("❌ **Conclusion**: The preprocessing pipeline is **NOT EFFECTIVE** in reducing inter-subject variability.\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("1. **Monitor preprocessing parameters**: Regularly validate that filtering and normalization parameters are optimal\n")
            f.write("2. **Subject-specific analysis**: Consider individual subject characteristics that may require tailored preprocessing\n")
            f.write("3. **Alternative normalization**: If z-score normalization is not effective, consider robust or min-max normalization\n")
            f.write("4. **Segmentation strategy**: Evaluate if different segment lengths or overlap ratios improve consistency\n\n")
            
            f.write("### Visualizations\n\n")
            f.write("The following visualizations are included in this report:\n")
            f.write("- `ppg_distribution_analysis.png`: PPG signal distribution comparisons\n")
            f.write("- `resp_distribution_analysis.png`: Respiratory signal distribution comparisons\n")
            f.write("- `ppg_distance_analysis.png`: PPG inter-subject distance analysis\n")
            f.write("- `resp_distance_analysis.png`: Respiratory inter-subject distance analysis\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    """Main function to generate the report."""
    try:
        # Initialize the report generator
        report_generator = PreprocessingAnalysisReport()
        
        # Generate the complete report
        summary = report_generator.create_report()
        
        print("\n" + "="*60)
        print("PREPROCESSING ANALYSIS REPORT COMPLETED")
        print("="*60)
        print("Files generated:")
        print("- preprocessing_analysis_report/preprocessing_analysis_report.md")
        print("- preprocessing_analysis_report/ppg_distribution_analysis.png")
        print("- preprocessing_analysis_report/resp_distribution_analysis.png")
        print("- preprocessing_analysis_report/ppg_distance_analysis.png")
        print("- preprocessing_analysis_report/resp_distance_analysis.png")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
