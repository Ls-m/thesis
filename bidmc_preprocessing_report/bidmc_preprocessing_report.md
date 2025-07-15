# BIDMC Dataset Preprocessing Analysis Report

## Executive Summary

This report analyzes the effectiveness of the preprocessing pipeline in reducing inter-subject variability for the BIDMC dataset. The analysis uses 53 subjects and compares data distributions before and after preprocessing.

## Dataset Information

- **Dataset**: BIDMC (Beth Israel Deaconess Medical Center)
- **Subjects analyzed**: 53 out of 53 total
- **Sampling rate**: 125 Hz
- **Signals**: PPG (PG) and Nasal Cannula respiratory signal
- **Format**: Comma-separated CSV files

## Subjects Analyzed

- **subject_00**: PPG=60,001 samples, RESP=60,001 samples
- **subject_01**: PPG=60,001 samples, RESP=60,001 samples
- **subject_02**: PPG=60,001 samples, RESP=60,001 samples
- **subject_03**: PPG=60,001 samples, RESP=60,001 samples
- **subject_04**: PPG=60,001 samples, RESP=60,001 samples
- **subject_05**: PPG=60,001 samples, RESP=60,001 samples
- **subject_06**: PPG=60,001 samples, RESP=60,001 samples
- **subject_07**: PPG=60,001 samples, RESP=60,001 samples
- **subject_08**: PPG=60,001 samples, RESP=60,001 samples
- **subject_09**: PPG=60,001 samples, RESP=60,001 samples
- **subject_10**: PPG=60,001 samples, RESP=60,001 samples
- **subject_11**: PPG=60,001 samples, RESP=60,001 samples
- **subject_12**: PPG=60,001 samples, RESP=60,001 samples
- **subject_13**: PPG=60,001 samples, RESP=60,001 samples
- **subject_14**: PPG=60,001 samples, RESP=60,001 samples
- **subject_15**: PPG=60,001 samples, RESP=60,001 samples
- **subject_16**: PPG=60,001 samples, RESP=60,001 samples
- **subject_17**: PPG=60,001 samples, RESP=60,001 samples
- **subject_18**: PPG=60,001 samples, RESP=60,001 samples
- **subject_19**: PPG=60,001 samples, RESP=60,001 samples
- **subject_20**: PPG=60,001 samples, RESP=60,001 samples
- **subject_21**: PPG=60,001 samples, RESP=60,001 samples
- **subject_22**: PPG=60,001 samples, RESP=60,001 samples
- **subject_23**: PPG=60,001 samples, RESP=60,001 samples
- **subject_24**: PPG=60,001 samples, RESP=60,001 samples
- **subject_25**: PPG=60,001 samples, RESP=60,001 samples
- **subject_26**: PPG=60,001 samples, RESP=60,001 samples
- **subject_27**: PPG=60,001 samples, RESP=60,001 samples
- **subject_28**: PPG=60,001 samples, RESP=60,001 samples
- **subject_29**: PPG=60,001 samples, RESP=60,001 samples
- **subject_30**: PPG=60,001 samples, RESP=60,001 samples
- **subject_31**: PPG=60,001 samples, RESP=60,001 samples
- **subject_32**: PPG=60,001 samples, RESP=60,001 samples
- **subject_33**: PPG=60,001 samples, RESP=60,001 samples
- **subject_34**: PPG=60,001 samples, RESP=60,001 samples
- **subject_35**: PPG=60,001 samples, RESP=60,001 samples
- **subject_36**: PPG=60,001 samples, RESP=60,001 samples
- **subject_37**: PPG=60,001 samples, RESP=60,001 samples
- **subject_38**: PPG=60,001 samples, RESP=60,001 samples
- **subject_39**: PPG=60,001 samples, RESP=60,001 samples
- **subject_40**: PPG=60,001 samples, RESP=60,001 samples
- **subject_41**: PPG=60,001 samples, RESP=60,001 samples
- **subject_42**: PPG=60,001 samples, RESP=60,001 samples
- **subject_43**: PPG=60,001 samples, RESP=60,001 samples
- **subject_44**: PPG=60,001 samples, RESP=60,001 samples
- **subject_45**: PPG=60,001 samples, RESP=60,001 samples
- **subject_46**: PPG=60,001 samples, RESP=60,001 samples
- **subject_47**: PPG=60,001 samples, RESP=60,001 samples
- **subject_48**: PPG=60,001 samples, RESP=60,001 samples
- **subject_49**: PPG=60,001 samples, RESP=60,001 samples
- **subject_50**: PPG=60,001 samples, RESP=60,001 samples
- **subject_51**: PPG=60,001 samples, RESP=60,001 samples
- **subject_52**: PPG=60,001 samples, RESP=60,001 samples

## Preprocessing Pipeline

The preprocessing pipeline includes:
1. **Bandpass Filtering**: 0.05-2.0 Hz with 2nd order Butterworth filter
2. **Downsampling**: From 125 Hz to 64 Hz
3. **Normalization**: Z-score normalization
4. **Segmentation**: 8-second segments with 50% overlap

## Key Findings

### PPG Signal Analysis

#### Statistical Summary
| Subject | Raw Mean | Raw Std | Prep Mean | Prep Std | Std Reduction |
|---------|----------|---------|-----------|----------|---------------|
| subject_00 | 0.4658 | 0.0655 | -0.0023 | 0.9994 | -1425.5% |
| subject_01 | 0.4684 | 0.1501 | 0.0035 | 0.9987 | -565.3% |
| subject_02 | 0.4276 | 0.1498 | -0.0068 | 0.9905 | -561.1% |
| subject_03 | 0.4282 | 0.1534 | 0.0039 | 0.9940 | -548.1% |
| subject_04 | 0.4952 | 0.0968 | -0.0045 | 0.9969 | -929.6% |
| subject_05 | 0.4181 | 0.1561 | -0.0036 | 0.9988 | -539.9% |
| subject_06 | 0.4705 | 0.1558 | 0.0009 | 0.9973 | -540.1% |
| subject_07 | 0.4653 | 0.1608 | 0.0033 | 0.9956 | -519.3% |
| subject_08 | 0.4678 | 0.1288 | -0.0024 | 0.9992 | -676.1% |
| subject_09 | 0.4441 | 0.1608 | -0.0007 | 1.0006 | -522.3% |
| subject_10 | 0.4590 | 0.1056 | 0.0070 | 0.9984 | -845.0% |
| subject_11 | 0.4781 | 0.0857 | 0.0013 | 0.9976 | -1063.6% |
| subject_12 | 0.4311 | 0.1177 | 0.0056 | 0.9969 | -747.2% |
| subject_13 | 0.4736 | 0.1884 | 0.0044 | 1.0008 | -431.1% |
| subject_14 | 0.4888 | 0.1509 | -0.0024 | 0.9934 | -558.3% |
| subject_15 | 0.4334 | 0.1448 | -0.0002 | 0.9984 | -589.5% |
| subject_16 | 0.4470 | 0.1532 | -0.0001 | 1.0004 | -552.8% |
| subject_17 | 0.4371 | 0.0941 | -0.0040 | 0.9994 | -962.4% |
| subject_18 | 0.4970 | 0.0382 | 0.0007 | 1.0002 | -2517.4% |
| subject_19 | 0.4846 | 0.0874 | -0.0063 | 0.9936 | -1036.6% |
| subject_20 | 0.4801 | 0.0901 | -0.0010 | 0.9997 | -1010.0% |
| subject_21 | 0.4852 | 0.0711 | -0.0031 | 0.9991 | -1304.3% |
| subject_22 | 0.3967 | 0.1605 | 0.0043 | 0.9962 | -520.7% |
| subject_23 | 0.4700 | 0.0821 | 0.0030 | 0.9963 | -1113.0% |
| subject_24 | 0.4697 | 0.0745 | 0.0008 | 1.0014 | -1244.1% |
| subject_25 | 0.4705 | 0.0987 | 0.0010 | 0.9987 | -911.7% |
| subject_26 | 0.4478 | 0.1607 | -0.0008 | 0.9995 | -521.8% |
| subject_27 | 0.4766 | 0.1203 | 0.0010 | 0.9973 | -728.8% |
| subject_28 | 0.4620 | 0.1619 | -0.0014 | 0.9995 | -517.4% |
| subject_29 | 0.4206 | 0.1480 | 0.0052 | 0.9947 | -572.1% |
| subject_30 | 0.4863 | 0.1030 | -0.0001 | 0.9960 | -866.7% |
| subject_31 | 0.4932 | 0.0727 | -0.0005 | 1.0001 | -1276.0% |
| subject_32 | 0.4732 | 0.1465 | 0.0015 | 0.9982 | -581.2% |
| subject_33 | 0.4813 | 0.1491 | 0.0029 | 0.9990 | -569.9% |
| subject_34 | 0.4833 | 0.1608 | 0.0055 | 0.9987 | -520.9% |
| subject_35 | 0.4294 | 0.1632 | 0.0015 | 0.9972 | -511.1% |
| subject_36 | 0.4666 | 0.1441 | -0.0009 | 1.0004 | -594.1% |
| subject_37 | 0.4770 | 0.0944 | 0.0037 | 0.9953 | -954.5% |
| subject_38 | 0.4684 | 0.1446 | -0.0031 | 0.9984 | -590.6% |
| subject_39 | 1.9250 | 0.6061 | -0.0001 | 1.0017 | -65.3% |
| subject_40 | 0.4736 | 0.0743 | 0.0025 | 0.9967 | -1240.9% |
| subject_41 | 1.7370 | 0.6708 | 0.0028 | 0.9964 | -48.5% |
| subject_42 | 0.4832 | 0.1544 | 0.0048 | 0.9982 | -546.6% |
| subject_43 | 1.9286 | 0.3260 | -0.0006 | 0.9958 | -205.5% |
| subject_44 | 1.8828 | 0.6734 | -0.0000 | 0.9997 | -48.5% |
| subject_45 | 1.8416 | 0.6335 | -0.0029 | 0.9982 | -57.6% |
| subject_46 | 1.8292 | 0.2803 | 0.0065 | 0.9860 | -251.7% |
| subject_47 | 1.8374 | 0.5745 | 0.0039 | 0.9868 | -71.8% |
| subject_48 | 0.4415 | 0.1515 | -0.0003 | 1.0002 | -560.2% |
| subject_49 | 0.4932 | 0.0826 | 0.0009 | 0.9929 | -1101.6% |
| subject_50 | 0.4524 | 0.1472 | 0.0023 | 0.9978 | -577.9% |
| subject_51 | 1.7560 | 0.6090 | -0.0071 | 0.9932 | -63.1% |
| subject_52 | 1.8037 | 0.6054 | -0.0020 | 1.0003 | -65.2% |

#### Inter-Subject Variability (PPG)
| Metric | Raw Data | Preprocessed | Improvement |
|--------|----------|--------------|-------------|
| mean_std | 0.5174 | 0.0033 | 99.36% |
| std_std | 0.1736 | 0.0032 | 98.15% |
| median_std | 0.4833 | 0.0585 | 87.90% |

### Respiratory Signal Analysis

#### Inter-Subject Variability (Respiratory)
| Metric | Raw Data | Preprocessed | Improvement |
|--------|----------|--------------|-------------|
| mean_std | 0.1844 | 0.0023 | 98.76% |
| std_std | 0.1199 | 0.0016 | 98.69% |
| median_std | 0.2005 | 0.2084 | -3.97% |

### Distance Analysis
- **Mean Wasserstein Distance Improvement**: -0.9612 (-96.12%)
- **Subject pairs with improvement**: 585/1378 (42.5%)

⚠️ **Conclusion**: Preprocessing shows **MIXED RESULTS** for BIDMC dataset.

## Comparison with Other Datasets

This BIDMC analysis can be compared with the main dataset analysis to understand:
- Dataset-specific preprocessing effectiveness
- Generalizability of preprocessing approaches
- Optimal preprocessing parameters for different data sources

---
*BIDMC report generated on 2025-07-13 21:37:25*
