#!/usr/bin/env python3
"""
Capno Dataset Preprocessing Analysis Script

This script generates a comprehensive preprocessing analysis report for the Capno dataset,
similar to the BIDMC preprocessing analysis. It analyzes the effectiveness of the 
preprocessing pipeline in reducing inter-subject variability.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
import yaml
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, 'src')

from data_utils import DataPreprocessor
from subject_wise_data_utils import SubjectWiseDataPreprocessor


class CapnoPreprocessingAnalyzer:
    """Analyze preprocessing effectiveness for Capno dataset."""
    
    def __init__(self, config_path: str = 'configs/improved_config.yaml'):
        """Initialize analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set capno dataset path
        self.config['data']['csv_folder'] = 'src/capno'
        
        self.preprocessor = SubjectWiseDataPreprocessor(self.config)
        self.raw_data = {}
        self.processed_data = {}
        
    def load_and_analyze_raw_data(self) -> Dict:
        """Load and analyze raw data statistics."""
        print("Loading and analyzing raw data...")
        
        # Load raw CSV files
        csv_files = self.preprocessor.load_csv_files('src/capno')
        
        raw_stats = {}
        
        for subject_id, df in csv_files.items():
            # Extract signals
            ppg_signal = df[self.config['data']['input_column']].values
            resp_signal = df[self.config['data']['target_column']].values
            
            # Remove NaN values for analysis
            valid_indices = ~(np.isnan(ppg_signal) | np.isnan(resp_signal))
            ppg_clean = ppg_signal[valid_indices]
            resp_clean = resp_signal[valid_indices]
            
            if len(ppg_clean) > 0:
                raw_stats[subject_id] = {
                    'ppg': {
                        'mean': np.mean(ppg_clean),
                        'std': np.std(ppg_clean),
                        'min': np.min(ppg_clean),
                        'max': np.max(ppg_clean),
                        'samples': len(ppg_clean),
                        'nan_count': np.isnan(ppg_signal).sum(),
                        'nan_percentage': (np.isnan(ppg_signal).sum() / len(ppg_signal)) * 100
                    },
                    'resp': {
                        'mean': np.mean(resp_clean),
                        'std': np.std(resp_clean),
                        'min': np.min(resp_clean),
                        'max': np.max(resp_clean),
                        'samples': len(resp_clean),
                        'nan_count': np.isnan(resp_signal).sum(),
                        'nan_percentage': (np.isnan(resp_signal).sum() / len(resp_signal)) * 100
                    },
                    'total_samples': len(ppg_signal),
                    'valid_samples': len(ppg_clean),
                    'duration_minutes': len(ppg_signal) / (self.config['data']['sampling_rate'] * 60)
                }
        
        self.raw_data = raw_stats
        return raw_stats
    
    def load_and_analyze_processed_data(self) -> Dict:
        """Load and analyze processed data statistics."""
        print("Processing data and analyzing...")
        
        # Process data using the preprocessor
        processed_data = self.preprocessor.prepare_dataset('src/capno')
        
        processed_stats = {}
        
        for subject_id, (ppg_segments, resp_segments) in processed_data.items():
            if len(ppg_segments) > 0:
                # Flatten segments for analysis
                ppg_flat = ppg_segments.flatten()
                resp_flat = resp_segments.flatten()
                
                processed_stats[subject_id] = {
                    'ppg': {
                        'mean': np.mean(ppg_flat),
                        'std': np.std(ppg_flat),
                        'min': np.min(ppg_flat),
                        'max': np.max(ppg_flat),
                        'samples': len(ppg_flat)
                    },
                    'resp': {
                        'mean': np.mean(resp_flat),
                        'std': np.std(resp_flat),
                        'min': np.min(resp_flat),
                        'max': np.max(resp_flat),
                        'samples': len(resp_flat)
                    },
                    'num_segments': len(ppg_segments),
                    'segment_length': ppg_segments.shape[1] if len(ppg_segments.shape) > 1 else 0
                }
        
        self.processed_data = processed_stats
        return processed_stats
    
    def calculate_inter_subject_variability(self, data_stats: Dict, signal_type: str) -> Dict:
        """Calculate inter-subject variability metrics."""
        means = [stats[signal_type]['mean'] for stats in data_stats.values()]
        stds = [stats[signal_type]['std'] for stats in data_stats.values()]
        
        return {
            'mean_std': np.std(means),
            'std_std': np.std(stds),
            'mean_mean': np.mean(means),
            'std_mean': np.mean(stds),
            'median_mean': np.median(means),
            'median_std': np.median(stds)
        }
    
    def calculate_wasserstein_distances(self) -> Dict:
        """Calculate Wasserstein distances between subject distributions."""
        print("Calculating Wasserstein distances...")
        
        # This is a simplified version - in practice you'd need the actual signal data
        # For now, we'll use the statistical summaries as a proxy
        subjects = list(self.raw_data.keys())
        n_subjects = len(subjects)
        
        raw_distances = []
        processed_distances = []
        
        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                subj1, subj2 = subjects[i], subjects[j]
                
                # Use means and stds as proxy for distribution comparison
                raw_dist = abs(self.raw_data[subj1]['ppg']['mean'] - self.raw_data[subj2]['ppg']['mean'])
                proc_dist = abs(self.processed_data[subj1]['ppg']['mean'] - self.processed_data[subj2]['ppg']['mean'])
                
                raw_distances.append(raw_dist)
                processed_distances.append(proc_dist)
        
        mean_raw_dist = np.mean(raw_distances)
        mean_proc_dist = np.mean(processed_distances)
        improvement = mean_proc_dist - mean_raw_dist
        improvement_pct = (improvement / mean_raw_dist) * 100 if mean_raw_dist != 0 else 0
        
        return {
            'mean_raw_distance': mean_raw_dist,
            'mean_processed_distance': mean_proc_dist,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'pairs_improved': sum(1 for r, p in zip(raw_distances, processed_distances) if p < r),
            'total_pairs': len(raw_distances)
        }
    
    def generate_visualization(self, output_dir: str) -> str:
        """Generate preprocessing analysis visualization."""
        print("Generating visualization...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        subjects = list(self.raw_data.keys())
        
        # 1. PPG Mean comparison
        raw_ppg_means = [self.raw_data[s]['ppg']['mean'] for s in subjects]
        proc_ppg_means = [self.processed_data[s]['ppg']['mean'] for s in subjects]
        
        ax1.scatter(raw_ppg_means, proc_ppg_means, alpha=0.7, s=60)
        ax1.set_xlabel('Raw PPG Mean')
        ax1.set_ylabel('Processed PPG Mean')
        ax1.set_title('PPG Signal: Raw vs Processed Means')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        min_val = min(min(raw_ppg_means), min(proc_ppg_means))
        max_val = max(max(raw_ppg_means), max(proc_ppg_means))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        ax1.legend()
        
        # 2. PPG Std comparison
        raw_ppg_stds = [self.raw_data[s]['ppg']['std'] for s in subjects]
        proc_ppg_stds = [self.processed_data[s]['ppg']['std'] for s in subjects]
        
        ax2.scatter(raw_ppg_stds, proc_ppg_stds, alpha=0.7, s=60)
        ax2.set_xlabel('Raw PPG Std')
        ax2.set_ylabel('Processed PPG Std')
        ax2.set_title('PPG Signal: Raw vs Processed Standard Deviations')
        ax2.grid(True, alpha=0.3)
        
        # 3. Respiratory Mean comparison
        raw_resp_means = [self.raw_data[s]['resp']['mean'] for s in subjects]
        proc_resp_means = [self.processed_data[s]['resp']['mean'] for s in subjects]
        
        ax3.scatter(raw_resp_means, proc_resp_means, alpha=0.7, s=60)
        ax3.set_xlabel('Raw Respiratory Mean')
        ax3.set_ylabel('Processed Respiratory Mean')
        ax3.set_title('Respiratory Signal: Raw vs Processed Means')
        ax3.grid(True, alpha=0.3)
        
        # 4. Data duration and segments
        durations = [self.raw_data[s]['duration_minutes'] for s in subjects]
        segments = [self.processed_data[s]['num_segments'] for s in subjects]
        
        ax4.scatter(durations, segments, alpha=0.7, s=60)
        ax4.set_xlabel('Recording Duration (minutes)')
        ax4.set_ylabel('Number of Segments Generated')
        ax4.set_title('Duration vs Segments Generated')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'capno_preprocessing_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_report(self, output_dir: str = 'capno_preprocessing_report') -> str:
        """Generate comprehensive preprocessing analysis report."""
        print(f"Generating Capno preprocessing analysis report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and analyze data
        raw_stats = self.load_and_analyze_raw_data()
        processed_stats = self.load_and_analyze_processed_data()
        
        # Calculate metrics
        ppg_raw_variability = self.calculate_inter_subject_variability(raw_stats, 'ppg')
        ppg_proc_variability = self.calculate_inter_subject_variability(processed_stats, 'ppg')
        resp_raw_variability = self.calculate_inter_subject_variability(raw_stats, 'resp')
        resp_proc_variability = self.calculate_inter_subject_variability(processed_stats, 'resp')
        
        distance_analysis = self.calculate_wasserstein_distances()
        
        # Generate visualization
        plot_path = self.generate_visualization(output_dir)
        
        # Generate markdown report
        report_path = os.path.join(output_dir, 'capno_preprocessing_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Capno Dataset Preprocessing Analysis Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the effectiveness of the preprocessing pipeline in reducing ")
            f.write(f"inter-subject variability for the Capno dataset. The analysis uses {len(raw_stats)} ")
            f.write("subjects and compares data distributions before and after preprocessing.\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write("- **Dataset**: Capno (Capnography-based respiratory signals)\n")
            f.write(f"- **Subjects analyzed**: {len(raw_stats)} out of {len(raw_stats)} total\n")
            f.write(f"- **Sampling rate**: {self.config['data']['sampling_rate']} Hz\n")
            f.write("- **Signals**: PPG and Nasal Cannula respiratory signal\n")
            f.write("- **Format**: Comma-separated CSV files\n\n")
            
            f.write("## Subjects Analyzed\n\n")
            for subject_id in sorted(raw_stats.keys()):
                stats = raw_stats[subject_id]
                f.write(f"- **{subject_id}**: PPG={stats['ppg']['samples']:,} samples, ")
                f.write(f"RESP={stats['resp']['samples']:,} samples, ")
                f.write(f"Duration={stats['duration_minutes']:.1f} min\n")
            
            f.write("\n## Preprocessing Pipeline\n\n")
            f.write("The preprocessing pipeline includes:\n")
            f.write(f"1. **Bandpass Filtering**: {self.config['preprocessing']['bandpass_filter']['low_freq']}-")
            f.write(f"{self.config['preprocessing']['bandpass_filter']['high_freq']} Hz with ")
            f.write(f"{self.config['preprocessing']['bandpass_filter']['order']}nd order Butterworth filter\n")
            f.write(f"2. **Downsampling**: From {self.config['data']['sampling_rate']} Hz to ")
            f.write(f"{self.config['preprocessing']['downsample']['target_rate']} Hz\n")
            f.write(f"3. **Normalization**: {self.config['preprocessing']['normalization']} normalization\n")
            f.write(f"4. **Segmentation**: {self.config['data']['segment_length'] // (self.config['data']['sampling_rate'] // self.config['preprocessing']['downsample']['target_rate'])}-sample segments ")
            f.write(f"with {int(self.config['data']['overlap'] * 100)}% overlap\n\n")
            
            f.write("## Key Findings\n\n")
            
            f.write("### PPG Signal Analysis\n\n")
            f.write("#### Statistical Summary\n")
            f.write("| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std | Std Reduction |\n")
            f.write("|---------|----------|---------|-----------|----------|---------------|\n")
            
            for subject_id in sorted(raw_stats.keys()):
                if subject_id in processed_stats:
                    raw = raw_stats[subject_id]['ppg']
                    proc = processed_stats[subject_id]['ppg']
                    std_reduction = ((proc['std'] - raw['std']) / raw['std']) * 100 if raw['std'] != 0 else 0
                    
                    f.write(f"| {subject_id} | {raw['mean']:.4f} | {raw['std']:.4f} | ")
                    f.write(f"{proc['mean']:.4f} | {proc['std']:.4f} | {std_reduction:.1f}% |\n")
            
            f.write("\n#### Inter-Subject Variability (PPG)\n")
            f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            
            mean_std_improvement = ((ppg_proc_variability['mean_std'] - ppg_raw_variability['mean_std']) / ppg_raw_variability['mean_std']) * 100
            std_std_improvement = ((ppg_proc_variability['std_std'] - ppg_raw_variability['std_std']) / ppg_raw_variability['std_std']) * 100
            median_std_improvement = ((ppg_proc_variability['median_std'] - ppg_raw_variability['median_std']) / ppg_raw_variability['median_std']) * 100
            
            f.write(f"| mean_std | {ppg_raw_variability['mean_std']:.4f} | {ppg_proc_variability['mean_std']:.4f} | {-mean_std_improvement:.2f}% |\n")
            f.write(f"| std_std | {ppg_raw_variability['std_std']:.4f} | {ppg_proc_variability['std_std']:.4f} | {-std_std_improvement:.2f}% |\n")
            f.write(f"| median_std | {ppg_raw_variability['median_std']:.4f} | {ppg_proc_variability['median_std']:.4f} | {-median_std_improvement:.2f}% |\n")
            
            f.write("\n### Respiratory Signal Analysis\n\n")
            f.write("#### Inter-Subject Variability (Respiratory)\n")
            f.write("| Metric | Raw Data | Preprocessed | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            
            resp_mean_std_improvement = ((resp_proc_variability['mean_std'] - resp_raw_variability['mean_std']) / resp_raw_variability['mean_std']) * 100
            resp_std_std_improvement = ((resp_proc_variability['std_std'] - resp_raw_variability['std_std']) / resp_raw_variability['std_std']) * 100
            resp_median_std_improvement = ((resp_proc_variability['median_std'] - resp_raw_variability['median_std']) / resp_raw_variability['median_std']) * 100
            
            f.write(f"| mean_std | {resp_raw_variability['mean_std']:.4f} | {resp_proc_variability['mean_std']:.4f} | {-resp_mean_std_improvement:.2f}% |\n")
            f.write(f"| std_std | {resp_raw_variability['std_std']:.4f} | {resp_proc_variability['std_std']:.4f} | {-resp_std_std_improvement:.2f}% |\n")
            f.write(f"| median_std | {resp_raw_variability['median_std']:.4f} | {resp_proc_variability['median_std']:.4f} | {-resp_median_std_improvement:.2f}% |\n")
            
            f.write("\n### Distance Analysis\n")
            f.write(f"- **Mean Distance Improvement**: {distance_analysis['improvement']:.4f} ({distance_analysis['improvement_percentage']:.2f}%)\n")
            f.write(f"- **Subject pairs with improvement**: {distance_analysis['pairs_improved']}/{distance_analysis['total_pairs']} ")
            f.write(f"({(distance_analysis['pairs_improved']/distance_analysis['total_pairs']*100):.1f}%)\n\n")
            
            # Determine conclusion
            overall_improvement = (mean_std_improvement + std_std_improvement) / 2
            if overall_improvement < -10:
                conclusion = "**EXCELLENT RESULTS**"
            elif overall_improvement < -5:
                conclusion = "**GOOD RESULTS**"
            elif overall_improvement < 0:
                conclusion = "**MODERATE IMPROVEMENT**"
            else:
                conclusion = "**MIXED RESULTS**"
            
            f.write(f"⚠️ **Conclusion**: Preprocessing shows {conclusion} for Capno dataset.\n\n")
            
            f.write("## Comparison with Other Datasets\n\n")
            f.write("This Capno analysis can be compared with the BIDMC dataset analysis to understand:\n")
            f.write("- Dataset-specific preprocessing effectiveness\n")
            f.write("- Generalizability of preprocessing approaches\n")
            f.write("- Optimal preprocessing parameters for different data sources\n\n")
            
            f.write("---\n")
            f.write(f"*Capno report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"Report generated successfully!")
        print(f"Report saved to: {report_path}")
        print(f"Visualization saved to: {plot_path}")
        
        return report_path


def main():
    """Main function to run the analysis."""
    print("=" * 60)
    print("CAPNO DATASET PREPROCESSING ANALYSIS")
    print("=" * 60)
    
    # Check if capno dataset exists
    if not os.path.exists('src/capno'):
        print("Error: Capno dataset not found at 'src/capno'")
        print("Please ensure the capno dataset is available in the src/capno/ directory")
        return 1
    
    # Check if there are CSV files
    csv_files = [f for f in os.listdir('src/capno') if f.endswith('.csv')]
    if not csv_files:
        print("Error: No CSV files found in 'src/capno'")
        print("Please ensure the capno dataset contains CSV files")
        return 1
    
    print(f"Found {len(csv_files)} CSV files in capno dataset")
    
    try:
        # Initialize analyzer
        analyzer = CapnoPreprocessingAnalyzer()
        
        # Generate report
        report_path = analyzer.generate_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Report location: {os.path.abspath(report_path)}")
        print(f"Subjects analyzed: {len(analyzer.raw_data)}")
        
        # Print quick summary
        if analyzer.raw_data and analyzer.processed_data:
            total_duration = sum(stats['duration_minutes'] for stats in analyzer.raw_data.values())
            total_segments = sum(stats['num_segments'] for stats in analyzer.processed_data.values())
            
            print(f"Total recording duration: {total_duration:.1f} minutes")
            print(f"Total segments generated: {total_segments}")
            print(f"Average segments per subject: {total_segments / len(analyzer.processed_data):.1f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
