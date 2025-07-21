import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from data_utils import DataPreprocessor
from preprocessing_config import PreprocessingConfigManager, EnhancedDataPreprocessor


class DataDistributionAnalyzer:
    """Analyze data distribution across subjects before and after preprocessing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        
    def load_raw_data_info(self, csv_folder: str) -> Dict:
        """Load raw data information without full preprocessing."""
        raw_data_info = {}
        csv_path = csv_folder
        
        print(f"Analyzing raw data in: {csv_path}")
        
        for filename in os.listdir(csv_path):
            if filename.endswith('.csv'):
                subject_id = filename.replace('.csv', '')
                try:
                    df = pd.read_csv(os.path.join(csv_path, filename))
                    
                    # Skip the first row which contains sampling rates
                    df = df.iloc[1:].reset_index(drop=True)
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Extract signals
                    ppg_signal = df[self.data_config['input_column']].values
                    resp_signal = df[self.data_config['target_column']].values
                    
                    # Calculate statistics
                    raw_data_info[subject_id] = {
                        'total_samples': len(ppg_signal),
                        'duration_minutes': len(ppg_signal) / (self.data_config['sampling_rate'] * 60),
                        'ppg_stats': {
                            'mean': np.nanmean(ppg_signal),
                            'std': np.nanstd(ppg_signal),
                            'min': np.nanmin(ppg_signal),
                            'max': np.nanmax(ppg_signal),
                            'nan_count': np.isnan(ppg_signal).sum(),
                            'nan_percentage': (np.isnan(ppg_signal).sum() / len(ppg_signal)) * 100
                        },
                        'resp_stats': {
                            'mean': np.nanmean(resp_signal),
                            'std': np.nanstd(resp_signal),
                            'min': np.nanmin(resp_signal),
                            'max': np.nanmax(resp_signal),
                            'nan_count': np.isnan(resp_signal).sum(),
                            'nan_percentage': (np.isnan(resp_signal).sum() / len(resp_signal)) * 100
                        },
                        'valid_samples': len(ppg_signal) - max(np.isnan(ppg_signal).sum(), np.isnan(resp_signal).sum()),
                        'data_quality': 'good' if (np.isnan(ppg_signal).sum() + np.isnan(resp_signal).sum()) / (2 * len(ppg_signal)) < 0.1 else 'poor'
                    }
                    
                    print(f"Analyzed {subject_id}: {len(ppg_signal)} samples, {raw_data_info[subject_id]['duration_minutes']:.1f} minutes")
                    
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
                    
        return raw_data_info
    
    def analyze_processed_data(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Analyze processed data statistics."""
        processed_info = {}
        
        for subject_id, (ppg_segments, resp_segments) in processed_data.items():
            if len(ppg_segments) > 0:
                # Flatten segments for overall statistics
                ppg_flat = ppg_segments.flatten()
                resp_flat = resp_segments.flatten()
                
                processed_info[subject_id] = {
                    'num_segments': len(ppg_segments),
                    'segment_length': ppg_segments.shape[1] if len(ppg_segments.shape) > 1 else 0,
                    'total_processed_samples': len(ppg_flat),
                    'ppg_processed_stats': {
                        'mean': np.mean(ppg_flat),
                        'std': np.std(ppg_flat),
                        'min': np.min(ppg_flat),
                        'max': np.max(ppg_flat),
                        'nan_count': np.isnan(ppg_flat).sum(),
                        'inf_count': np.isinf(ppg_flat).sum()
                    },
                    'resp_processed_stats': {
                        'mean': np.mean(resp_flat),
                        'std': np.std(resp_flat),
                        'min': np.min(resp_flat),
                        'max': np.max(resp_flat),
                        'nan_count': np.isnan(resp_flat).sum(),
                        'inf_count': np.isinf(resp_flat).sum()
                    }
                }
        
        return processed_info
    
    def detect_class_imbalance(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                              threshold: float = 0.1) -> Dict:
        """
        Detect potential class imbalance by analyzing respiratory signal distributions.
        
        Args:
            processed_data: Dictionary of processed subject data
            threshold: Threshold for detecting different respiratory patterns
            
        Returns:
            Dictionary containing class balance analysis
        """
        class_analysis = {}
        
        for subject_id, (ppg_segments, resp_segments) in processed_data.items():
            if len(resp_segments) > 0:
                # Analyze respiratory patterns by looking at signal characteristics
                resp_flat = resp_segments.flatten()
                
                # Simple pattern detection based on signal variance and mean
                high_variance_mask = np.var(resp_segments, axis=1) > np.median(np.var(resp_segments, axis=1))
                low_variance_mask = ~high_variance_mask
                
                positive_mean_mask = np.mean(resp_segments, axis=1) > 0
                negative_mean_mask = ~positive_mean_mask
                
                class_analysis[subject_id] = {
                    'total_segments': len(resp_segments),
                    'high_variance_segments': np.sum(high_variance_mask),
                    'low_variance_segments': np.sum(low_variance_mask),
                    'positive_mean_segments': np.sum(positive_mean_mask),
                    'negative_mean_segments': np.sum(negative_mean_mask),
                    'variance_ratio': np.sum(high_variance_mask) / len(resp_segments),
                    'mean_ratio': np.sum(positive_mean_mask) / len(resp_segments),
                    'potential_imbalance': abs(0.5 - np.sum(positive_mean_mask) / len(resp_segments)) > threshold
                }
        
        return class_analysis
    
    def generate_distribution_plots(self, raw_data_info: Dict, processed_info: Dict, 
                                  output_dir: str) -> List[str]:
        """Generate distribution analysis plots."""
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Subject count and data duration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        subjects = list(raw_data_info.keys())
        durations = [raw_data_info[s]['duration_minutes'] for s in subjects]
        
        ax1.bar(range(len(subjects)), durations)
        ax1.set_xlabel('Subject Index')
        ax1.set_ylabel('Duration (minutes)')
        ax1.set_title('Data Duration per Subject')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Duration (minutes)')
        ax2.set_ylabel('Number of Subjects')
        ax2.set_title('Distribution of Data Durations')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'subject_durations.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # 2. Data quality analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ppg_nan_percentages = [raw_data_info[s]['ppg_stats']['nan_percentage'] for s in subjects]
        resp_nan_percentages = [raw_data_info[s]['resp_stats']['nan_percentage'] for s in subjects]
        
        ax1.scatter(ppg_nan_percentages, resp_nan_percentages, alpha=0.7)
        ax1.set_xlabel('PPG NaN Percentage')
        ax1.set_ylabel('Respiratory NaN Percentage')
        ax1.set_title('Data Quality: Missing Data Analysis')
        ax1.grid(True, alpha=0.3)
        
        quality_counts = {}
        for s in subjects:
            quality = raw_data_info[s]['data_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        ax2.pie(quality_counts.values(), labels=quality_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Overall Data Quality Distribution')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'data_quality.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # 3. Signal statistics comparison (before vs after preprocessing)
        if processed_info:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # PPG statistics
            raw_ppg_means = [raw_data_info[s]['ppg_stats']['mean'] for s in subjects if s in processed_info]
            processed_ppg_means = [processed_info[s]['ppg_processed_stats']['mean'] for s in subjects if s in processed_info]
            
            ax1.scatter(raw_ppg_means, processed_ppg_means, alpha=0.7)
            ax1.set_xlabel('Raw PPG Mean')
            ax1.set_ylabel('Processed PPG Mean')
            ax1.set_title('PPG Signal: Raw vs Processed Means')
            ax1.grid(True, alpha=0.3)
            
            raw_ppg_stds = [raw_data_info[s]['ppg_stats']['std'] for s in subjects if s in processed_info]
            processed_ppg_stds = [processed_info[s]['ppg_processed_stats']['std'] for s in subjects if s in processed_info]
            
            ax2.scatter(raw_ppg_stds, processed_ppg_stds, alpha=0.7)
            ax2.set_xlabel('Raw PPG Std')
            ax2.set_ylabel('Processed PPG Std')
            ax2.set_title('PPG Signal: Raw vs Processed Standard Deviations')
            ax2.grid(True, alpha=0.3)
            
            # Respiratory statistics
            raw_resp_means = [raw_data_info[s]['resp_stats']['mean'] for s in subjects if s in processed_info]
            processed_resp_means = [processed_info[s]['resp_processed_stats']['mean'] for s in subjects if s in processed_info]
            
            ax3.scatter(raw_resp_means, processed_resp_means, alpha=0.7)
            ax3.set_xlabel('Raw Respiratory Mean')
            ax3.set_ylabel('Processed Respiratory Mean')
            ax3.set_title('Respiratory Signal: Raw vs Processed Means')
            ax3.grid(True, alpha=0.3)
            
            raw_resp_stds = [raw_data_info[s]['resp_stats']['std'] for s in subjects if s in processed_info]
            processed_resp_stds = [processed_info[s]['resp_processed_stats']['std'] for s in subjects if s in processed_info]
            
            ax4.scatter(raw_resp_stds, processed_resp_stds, alpha=0.7)
            ax4.set_xlabel('Raw Respiratory Std')
            ax4.set_ylabel('Processed Respiratory Std')
            ax4.set_title('Respiratory Signal: Raw vs Processed Standard Deviations')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, 'preprocessing_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            # 4. Segment count distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            segment_counts = [processed_info[s]['num_segments'] for s in subjects if s in processed_info]
            
            ax1.bar(range(len(segment_counts)), segment_counts)
            ax1.set_xlabel('Subject Index')
            ax1.set_ylabel('Number of Segments')
            ax1.set_title('Segments per Subject After Preprocessing')
            
            ax2.hist(segment_counts, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Number of Segments')
            ax2.set_ylabel('Number of Subjects')
            ax2.set_title('Distribution of Segment Counts')
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, 'segment_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        return plot_files
    
    def generate_report(self, csv_folder: str, output_dir: str = None) -> Dict:
        """
        Generate comprehensive data distribution report.
        
        Args:
            csv_folder: Path to CSV data folder
            output_dir: Output directory for report (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        if output_dir is None:
            dataset_name = os.path.basename(csv_folder)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{dataset_name}_distribution_report_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating data distribution report for: {csv_folder}")
        print(f"Output directory: {output_dir}")
        
        # Analyze raw data
        print("Analyzing raw data...")
        raw_data_info = self.load_raw_data_info(csv_folder)
        
        # Process data for comparison
        print("Processing data for comparison...")
        preprocessor = DataPreprocessor(self.config)
        processed_data = preprocessor.prepare_dataset(csv_folder)
        
        # Analyze processed data
        print("Analyzing processed data...")
        processed_info = self.analyze_processed_data(processed_data)
        
        # Analyze class balance
        print("Analyzing class balance...")
        class_analysis = self.detect_class_imbalance(processed_data)
        
        # Generate plots
        print("Generating plots...")
        plot_files = self.generate_distribution_plots(raw_data_info, processed_info, output_dir)
        
        # Compile summary statistics
        summary_stats = {
            'dataset_info': {
                'dataset_path': csv_folder,
                'total_subjects': len(raw_data_info),
                'analysis_timestamp': datetime.now().isoformat(),
                'config_used': self.config
            },
            'raw_data_summary': {
                'total_subjects': len(raw_data_info),
                'total_duration_hours': sum(info['duration_minutes'] for info in raw_data_info.values()) / 60,
                'average_duration_minutes': np.mean([info['duration_minutes'] for info in raw_data_info.values()]),
                'subjects_with_good_quality': sum(1 for info in raw_data_info.values() if info['data_quality'] == 'good'),
                'average_ppg_nan_percentage': np.mean([info['ppg_stats']['nan_percentage'] for info in raw_data_info.values()]),
                'average_resp_nan_percentage': np.mean([info['resp_stats']['nan_percentage'] for info in raw_data_info.values()])
            },
            'processed_data_summary': {
                'subjects_successfully_processed': len(processed_info),
                'total_segments': sum(info['num_segments'] for info in processed_info.values()),
                'average_segments_per_subject': np.mean([info['num_segments'] for info in processed_info.values()]) if processed_info else 0,
                'subjects_with_potential_imbalance': sum(1 for info in class_analysis.values() if info['potential_imbalance'])
            },
            'detailed_analysis': {
                'raw_data_info': raw_data_info,
                'processed_info': processed_info,
                'class_analysis': class_analysis
            },
            'generated_plots': plot_files
        }
        
        # Save detailed report as JSON
        report_file = os.path.join(output_dir, 'distribution_analysis_report.json')
        with open(report_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def clean_for_json(data):
                if isinstance(data, dict):
                    return {k: clean_for_json(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_for_json(item) for item in data]
                else:
                    return convert_numpy(data)
            
            clean_summary = clean_for_json(summary_stats)
            json.dump(clean_summary, f, indent=2, default=str)
        
        # Generate markdown report
        self.generate_markdown_report(summary_stats, output_dir)
        
        print(f"Data distribution analysis completed!")
        print(f"Report saved to: {output_dir}")
        print(f"Summary: {len(raw_data_info)} subjects, {len(processed_info)} successfully processed")
        
        return summary_stats
    
    def generate_markdown_report(self, summary_stats: Dict, output_dir: str):
        """Generate a markdown report from the analysis results."""
        report_path = os.path.join(output_dir, 'distribution_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Data Distribution Analysis Report\n\n")
            
            # Dataset info
            dataset_info = summary_stats['dataset_info']
            f.write(f"**Dataset:** {dataset_info['dataset_path']}\n")
            f.write(f"**Analysis Date:** {dataset_info['analysis_timestamp']}\n")
            f.write(f"**Total Subjects:** {dataset_info['total_subjects']}\n\n")
            
            # Raw data summary
            raw_summary = summary_stats['raw_data_summary']
            f.write("## Raw Data Summary\n\n")
            f.write(f"- **Total Subjects:** {raw_summary['total_subjects']}\n")
            f.write(f"- **Total Duration:** {raw_summary['total_duration_hours']:.1f} hours\n")
            f.write(f"- **Average Duration per Subject:** {raw_summary['average_duration_minutes']:.1f} minutes\n")
            f.write(f"- **Subjects with Good Quality:** {raw_summary['subjects_with_good_quality']}/{raw_summary['total_subjects']}\n")
            f.write(f"- **Average PPG Missing Data:** {raw_summary['average_ppg_nan_percentage']:.2f}%\n")
            f.write(f"- **Average Respiratory Missing Data:** {raw_summary['average_resp_nan_percentage']:.2f}%\n\n")
            
            # Processed data summary
            processed_summary = summary_stats['processed_data_summary']
            f.write("## Processed Data Summary\n\n")
            f.write(f"- **Successfully Processed Subjects:** {processed_summary['subjects_successfully_processed']}\n")
            f.write(f"- **Total Segments Generated:** {processed_summary['total_segments']}\n")
            f.write(f"- **Average Segments per Subject:** {processed_summary['average_segments_per_subject']:.1f}\n")
            f.write(f"- **Subjects with Potential Class Imbalance:** {processed_summary['subjects_with_potential_imbalance']}\n\n")
            
            # Generated plots
            f.write("## Generated Visualizations\n\n")
            for plot_file in summary_stats['generated_plots']:
                plot_name = os.path.basename(plot_file)
                f.write(f"- {plot_name}\n")
            
            f.write("\n## Configuration Used\n\n")
            f.write("```yaml\n")
            import yaml
            f.write(yaml.dump(dataset_info['config_used'], default_flow_style=False))
            f.write("```\n")
