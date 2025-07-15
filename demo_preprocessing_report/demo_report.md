# Preprocessing Analysis Demonstration

## Overview

This demonstration shows the methodology for analyzing preprocessing effectiveness in reducing inter-subject variability using a small sample of data.

## Subjects Analyzed

- **EJ14**: 1,000 samples
- **WM05**: 1,000 samples
- **TS34**: 1,000 samples

## Key Findings

### Statistical Summary
| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std |
|---------|----------|---------|-----------|----------|
| EJ14 | 132.71 | 44.93 | 0.00 | 1.00 |
| WM05 | 122.14 | 33.91 | 0.00 | 1.00 |
| TS34 | 125.86 | 46.27 | -0.00 | 1.00 |

### Inter-Subject Variability
- **Raw data mean variability**: 4.3798
- **Preprocessed data mean variability**: 0.0000
- **Mean variability improvement**: 100.00%

- **Raw data std variability**: 5.5369
- **Preprocessed data std variability**: 0.0000
- **Std variability improvement**: 100.00%

### Distance Analysis
- **Mean Wasserstein distance improvement**: 0.9869 (98.69%)
- **Number of subject pairs**: 3
- **Pairs with improvement**: 3/3

✅ **Conclusion**: Preprocessing demonstrates **POSITIVE EFFECT** in reducing inter-subject variability.

## Methodology

This demonstration uses:
1. **Sample data**: Small subset for fast processing
2. **Simulated preprocessing**: Smoothing + Z-score normalization
3. **Statistical analysis**: Mean, std, variability metrics
4. **Distance metrics**: Wasserstein distance between distributions
5. **Visualization**: Distribution comparisons and improvement analysis

The full analysis (running in background) will provide:
- All 28 subjects with complete data
- Full preprocessing pipeline (filtering, downsampling, normalization, segmentation)
- Comprehensive statistical analysis
- Multiple distance metrics (Wasserstein + KL divergence)
- Detailed visualizations

---
*Demo report generated on 2025-07-13 20:38:47*
