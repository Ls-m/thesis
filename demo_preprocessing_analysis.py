#!/usr/bin/env python3
"""
Demo Preprocessing Analysis - Immediate Results

This script provides an immediate demonstration of the preprocessing analysis methodology
using minimal data processing to show the concept and generate quick results.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

def load_sample_data(csv_folder='csv', max_subjects=3, sample_size=1000):
    """Load a small sample of data for demonstration."""
    print(f"Loading sample data from {max_subjects} subjects...")
    
    raw_data = {}
    files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')][:max_subjects]
    
    for filename in files:
        subject_id = filename.replace('.csv', '')
        
        try:
            # Read just the first few thousand lines for speed
            df = pd.read_csv(os.path.join(csv_folder, filename), nrows=sample_size*2)
            
            # Handle the data format
            if len(df.columns) == 1:
                # Parse concatenated data
                data_rows = []
                for idx, row in df.iterrows():
                    if idx == 0:  # Skip header row
                        continue
                    row_str = str(row.iloc[0])
                    parts = row_str.split('\t') if '\t' in row_str else row_str.split()
                    if len(parts) >= 6:
                        data_rows.append(parts[:6])
                    if len(data_rows) >= sample_size:
                        break
                
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=['Time', 'AIRFLOW', 'NASAL CANULA', 'CHEST', 'ABDOMEN', 'PPG'])
            else:
                df = df.iloc[1:sample_size+1]  # Skip header, take sample
            
            # Convert to numeric
            for col in df.columns:
                if col != 'Time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Extract signals
            if 'PPG' in df.columns and 'NASAL CANULA' in df.columns:
                ppg_signal = df['PPG'].values
                resp_signal = df['NASAL CANULA'].values
                
                # Remove NaN values
                valid_indices = ~(np.isnan(ppg_signal) | np.isnan(resp_signal))
                ppg_signal = ppg_signal[valid_indices]
                resp_signal = resp_signal[valid_indices]
                
                if len(ppg_signal) > 100:  # Need minimum data
                    raw_data[subject_id] = {
                        'ppg': ppg_signal,
                        'resp': resp_signal
                    }
                    print(f"Loaded {subject_id}: PPG={len(ppg_signal)} samples, RESP={len(resp_signal)} samples")
        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return raw_data

def simulate_preprocessing(raw_data):
    """Simulate preprocessing effects for demonstration."""
    print("Simulating preprocessing effects...")
    
    preprocessed_data = {}
    
    for subject_id, signals in raw_data.items():
        ppg_signal = signals['ppg'].copy()
        resp_signal = signals['resp'].copy()
        
        # Simulate bandpass filtering (simple smoothing)
        window_size = min(5, len(ppg_signal) // 10)
        if window_size > 1:
            ppg_filtered = np.convolve(ppg_signal, np.ones(window_size)/window_size, mode='same')
            resp_filtered = np.convolve(resp_signal, np.ones(window_size)/window_size, mode='same')
        else:
            ppg_filtered = ppg_signal
            resp_filtered = resp_signal
        
        # Simulate normalization (z-score)
        ppg_normalized = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)
        resp_normalized = (resp_filtered - np.mean(resp_filtered)) / np.std(resp_filtered)
        
        # Clip extreme values
        ppg_normalized = np.clip(ppg_normalized, -3, 3)
        resp_normalized = np.clip(resp_normalized, -3, 3)
        
        preprocessed_data[subject_id] = {
            'ppg': ppg_normalized,
            'resp': resp_normalized
        }
    
    return preprocessed_data

def calculate_basic_stats(data, signal_type='ppg'):
    """Calculate basic statistics."""
    stats_dict = {}
    
    for subject_id, signals in data.items():
        signal = signals[signal_type]
        stats_dict[subject_id] = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'median': np.median(signal)
        }
    
    return stats_dict

def calculate_inter_subject_variability(stats_dict):
    """Calculate variability between subjects."""
    means = [stats['mean'] for stats in stats_dict.values()]
    stds = [stats['std'] for stats in stats_dict.values()]
    
    return {
        'mean_variability': np.std(means),
        'std_variability': np.std(stds),
        'mean_range': np.max(means) - np.min(means),
        'std_range': np.max(stds) - np.min(stds)
    }

def calculate_pairwise_distances(raw_data, preprocessed_data, signal_type='ppg'):
    """Calculate Wasserstein distances between subject pairs."""
    subjects = list(raw_data.keys())
    distances = {'raw': [], 'preprocessed': [], 'improvements': []}
    
    for i in range(len(subjects)):
        for j in range(i+1, len(subjects)):
            subj1, subj2 = subjects[i], subjects[j]
            
            # Calculate distances
            raw_dist = wasserstein_distance(
                raw_data[subj1][signal_type], 
                raw_data[subj2][signal_type]
            )
            prep_dist = wasserstein_distance(
                preprocessed_data[subj1][signal_type], 
                preprocessed_data[subj2][signal_type]
            )
            
            improvement = (raw_dist - prep_dist) / raw_dist if raw_dist > 0 else 0
            
            distances['raw'].append(raw_dist)
            distances['preprocessed'].append(prep_dist)
            distances['improvements'].append(improvement)
    
    return distances

def create_demo_visualization(raw_data, preprocessed_data, output_dir):
    """Create demonstration visualization."""
    subjects = list(raw_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Preprocessing Analysis Demonstration', fontsize=16, fontweight='bold')
    
    # 1. Raw PPG distributions
    ax1 = axes[0, 0]
    for subject in subjects:
        ax1.hist(raw_data[subject]['ppg'], bins=20, alpha=0.5, label=f'{subject} (raw)', density=True)
    ax1.set_title('Raw PPG Signal Distributions')
    ax1.set_xlabel('Signal Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # 2. Preprocessed PPG distributions
    ax2 = axes[0, 1]
    for subject in subjects:
        ax2.hist(preprocessed_data[subject]['ppg'], bins=20, alpha=0.5, label=f'{subject} (prep)', density=True)
    ax2.set_title('Preprocessed PPG Signal Distributions')
    ax2.set_xlabel('Signal Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # 3. Standard deviation comparison
    ax3 = axes[1, 0]
    raw_stats = calculate_basic_stats(raw_data, 'ppg')
    prep_stats = calculate_basic_stats(preprocessed_data, 'ppg')
    
    raw_stds = [raw_stats[subj]['std'] for subj in subjects]
    prep_stds = [prep_stats[subj]['std'] for subj in subjects]
    
    x = np.arange(len(subjects))
    width = 0.35
    ax3.bar(x - width/2, raw_stds, width, label='Raw', alpha=0.7)
    ax3.bar(x + width/2, prep_stds, width, label='Preprocessed', alpha=0.7)
    ax3.set_title('Standard Deviation Comparison')
    ax3.set_xlabel('Subjects')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_xticks(x)
    ax3.set_xticklabels(subjects)
    ax3.legend()
    
    # 4. Distance improvements
    ax4 = axes[1, 1]
    distances = calculate_pairwise_distances(raw_data, preprocessed_data, 'ppg')
    
    ax4.scatter(distances['raw'], distances['preprocessed'], alpha=0.7)
    ax4.plot([min(distances['raw']), max(distances['raw'])], 
             [min(distances['raw']), max(distances['raw'])], 'r--', label='y=x')
    ax4.set_xlabel('Raw Data Distance')
    ax4.set_ylabel('Preprocessed Data Distance')
    ax4.set_title('Wasserstein Distance: Raw vs Preprocessed')
    ax4.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'demo_preprocessing_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return distances

def generate_demo_report():
    """Generate the demonstration report."""
    print("="*60)
    print("PREPROCESSING ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create output directory
    output_dir = 'demo_preprocessing_report'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample data
    raw_data = load_sample_data(max_subjects=3, sample_size=1000)
    
    if len(raw_data) < 2:
        print("Error: Need at least 2 subjects for comparison")
        return
    
    # Simulate preprocessing
    preprocessed_data = simulate_preprocessing(raw_data)
    
    # Calculate statistics
    raw_ppg_stats = calculate_basic_stats(raw_data, 'ppg')
    prep_ppg_stats = calculate_basic_stats(preprocessed_data, 'ppg')
    
    raw_variability = calculate_inter_subject_variability(raw_ppg_stats)
    prep_variability = calculate_inter_subject_variability(prep_ppg_stats)
    
    # Create visualization
    print("Creating visualization...")
    distances = create_demo_visualization(raw_data, preprocessed_data, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, 'demo_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Preprocessing Analysis Demonstration\n\n")
        f.write("## Overview\n\n")
        f.write("This demonstration shows the methodology for analyzing preprocessing effectiveness ")
        f.write("in reducing inter-subject variability using a small sample of data.\n\n")
        
        f.write("## Subjects Analyzed\n\n")
        for subject_id in raw_data.keys():
            ppg_len = len(raw_data[subject_id]['ppg'])
            f.write(f"- **{subject_id}**: {ppg_len:,} samples\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        
        # Statistics comparison
        f.write("### Statistical Summary\n")
        f.write("| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std |\n")
        f.write("|---------|----------|---------|-----------|----------|\n")
        
        for subject in raw_data.keys():
            raw_mean = raw_ppg_stats[subject]['mean']
            raw_std = raw_ppg_stats[subject]['std']
            prep_mean = prep_ppg_stats[subject]['mean']
            prep_std = prep_ppg_stats[subject]['std']
            
            f.write(f"| {subject} | {raw_mean:.2f} | {raw_std:.2f} | {prep_mean:.2f} | {prep_std:.2f} |\n")
        
        # Variability analysis
        f.write(f"\n### Inter-Subject Variability\n")
        f.write(f"- **Raw data mean variability**: {raw_variability['mean_variability']:.4f}\n")
        f.write(f"- **Preprocessed data mean variability**: {prep_variability['mean_variability']:.4f}\n")
        
        mean_var_improvement = (raw_variability['mean_variability'] - prep_variability['mean_variability']) / raw_variability['mean_variability'] * 100
        f.write(f"- **Mean variability improvement**: {mean_var_improvement:.2f}%\n\n")
        
        f.write(f"- **Raw data std variability**: {raw_variability['std_variability']:.4f}\n")
        f.write(f"- **Preprocessed data std variability**: {prep_variability['std_variability']:.4f}\n")
        
        std_var_improvement = (raw_variability['std_variability'] - prep_variability['std_variability']) / raw_variability['std_variability'] * 100
        f.write(f"- **Std variability improvement**: {std_var_improvement:.2f}%\n\n")
        
        # Distance analysis
        if distances['improvements']:
            mean_distance_improvement = np.mean(distances['improvements'])
            f.write(f"### Distance Analysis\n")
            f.write(f"- **Mean Wasserstein distance improvement**: {mean_distance_improvement:.4f} ({mean_distance_improvement*100:.2f}%)\n")
            f.write(f"- **Number of subject pairs**: {len(distances['improvements'])}\n")
            
            positive_improvements = sum(1 for imp in distances['improvements'] if imp > 0)
            f.write(f"- **Pairs with improvement**: {positive_improvements}/{len(distances['improvements'])}\n\n")
            
            if mean_distance_improvement > 0:
                f.write("✅ **Conclusion**: Preprocessing demonstrates **POSITIVE EFFECT** in reducing inter-subject variability.\n\n")
            else:
                f.write("⚠️ **Conclusion**: Preprocessing shows **MIXED RESULTS** in this sample.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("This demonstration uses:\n")
        f.write("1. **Sample data**: Small subset for fast processing\n")
        f.write("2. **Simulated preprocessing**: Smoothing + Z-score normalization\n")
        f.write("3. **Statistical analysis**: Mean, std, variability metrics\n")
        f.write("4. **Distance metrics**: Wasserstein distance between distributions\n")
        f.write("5. **Visualization**: Distribution comparisons and improvement analysis\n\n")
        
        f.write("The full analysis (running in background) will provide:\n")
        f.write("- All 28 subjects with complete data\n")
        f.write("- Full preprocessing pipeline (filtering, downsampling, normalization, segmentation)\n")
        f.write("- Comprehensive statistical analysis\n")
        f.write("- Multiple distance metrics (Wasserstein + KL divergence)\n")
        f.write("- Detailed visualizations\n\n")
        
        f.write("---\n")
        f.write(f"*Demo report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Print summary
    print(f"\nDemo analysis completed!")
    print(f"Subjects analyzed: {list(raw_data.keys())}")
    print(f"Files generated:")
    print(f"- {report_path}")
    print(f"- {os.path.join(output_dir, 'demo_preprocessing_analysis.png')}")
    
    if distances['improvements']:
        mean_improvement = np.mean(distances['improvements'])
        print(f"\nKey Result: {mean_improvement*100:.2f}% average improvement in inter-subject distance")
        
        if mean_improvement > 0:
            print("✅ Preprocessing shows positive effect in reducing variability")
        else:
            print("⚠️ Preprocessing shows mixed results in this sample")
    
    print("="*60)

if __name__ == "__main__":
    generate_demo_report()
