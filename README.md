# Epileptic Seizure Detection from EEG using Adaptive Hilbert-Huang Transform

Novel EEG classification system for detecting epileptic seizures using **AM-HHT (Adaptive Multi-feature Hilbert-Huang Transform)** — a custom feature extraction method that uses **Spectral Flatness** to adaptively filter noisy IMF components from Empirical Mode Decomposition.

Tested on the **Bonn University EEG dataset** (11,500 segments). Achieved **91% accuracy**, **95% sensitivity**, and **0.97 AUC**, outperforming FFT (+8%), DWT (+3%), and standard HHT (+5%).

---

## Original Contribution

Standard EMD generates IMF components that may contain noise rather than useful signal information. Existing methods filter IMFs based on energy or correlation — but noisy components can still have significant energy.

**AM-HHT introduces Spectral Flatness (Wiener Entropy) as an IMF selection criterion.** Spectral Flatness measures how uniform a signal's spectrum is: values near 1 indicate white noise (flat spectrum → discard), values near 0 indicate structured signal (peaked spectrum → keep). IMFs with SF > 0.7 are excluded.

This approach has not been previously applied to IMF selection in EEG seizure detection literature.

---

## Results

### Method Comparison

| Method | Accuracy | Sensitivity | Specificity | F1-Score | AUC |
|---|---|---|---|---|---|
| FFT (14 features) | 83.0% | 87.0% | 79.0% | 0.837 | 0.892 |
| DWT (15 features) | 88.2% | 92.5% | 84.0% | 0.887 | 0.950 |
| HHT Standard (15 features) | 86.0% | 90.5% | 81.5% | 0.866 | 0.933 |
| **AM-HHT (80 features)** | **91.0%** | **95.0%** | **87.0%** | **0.914** | **0.969** |

### Seizure Transition Detection

Sliding-window analysis using theta/beta ratio evolution detects seizure onset with **100% detection rate** (8/8 signals) and **0.96 average confidence**.

---

## Sample Outputs

### Raw EEG Signals: Seizure vs Healthy
![Raw Signals](screenshots/02_raw_signals.png)

### CEEMDAN Decomposition into IMFs
![CEEMDAN Decomposition](screenshots/04_CEEMDAN_decomposition.png)

### Hilbert-Huang Spectrogram: Normal vs Seizure
![HH Spectrogram](screenshots/05_HH_spectrogram_comparison.png)

### Full Method Comparison (FFT vs DWT vs HHT vs AM-HHT)
![Method Comparison](screenshots/10_FINAL_*.png)

### HHT Standard vs AM-HHT (Feature Importance)
![HHT vs AM-HHT](screenshots/08_comparison_*.png)

### SNR Analysis
![SNR Analysis](screenshots/04.5_snr_analysis.png)

### Seizure Transition Detection
![Seizure Transition](screenshots/07_seizure_transition_detection.png)

### Confusion Matrices
![Confusion Matrices](screenshots/06_confusion_matrices.png)

---

## Project Structure

```
├── 01_load_and_explore.py              # Data loading, EDA, signal visualization
├── 02_test_hht_single_sample.py        # CEEMDAN decomposition & HH spectrogram
├── 03_extract_hht_features.py          # HHT feature extraction (energy, entropy, amplitude)
├── 04_train_model.py                   # Random Forest & SVM classification
├── 04_5_snr_analysis.py                # Signal-to-Noise Ratio analysis
├── 05_adaptive_hht_ORIGINAL.py         # AM-HHT: Spectral Flatness IMF selection (original contribution)
├── 06_compare_methods.py               # HHT Standard vs AM-HHT comparison
├── 07_preictal_prediction.py           # Seizure onset transition detection
├── 10_FINAL_comparison_all_methods.py  # Full comparison: FFT vs DWT vs HHT vs AM-HHT
└── screenshots/
    ├── raw_signals.png
    ├── ceemdan_decomposition.png
    ├── hh_spectrogram.png
    ├── method_comparison.png
    ├── hht_vs_amhht.png
    ├── snr_analysis.png
    ├── seizure_transition.png
    └── confusion_matrices.png
```

---

## Pipeline Workflow

```
Phase 1 — Data Exploration
  1. Load Bonn EEG dataset (11,500 segments × 178 samples)
  2. Binary labeling: Normal (class 1) vs Seizure (class 5)
  3. Visualize raw signals and amplitude distributions

Phase 2 — Signal Decomposition
  4. CEEMDAN decomposition → Intrinsic Mode Functions (IMFs)
  5. Hilbert Transform → instantaneous frequency & amplitude
  6. Generate Hilbert-Huang spectrograms

Phase 3 — Feature Extraction
  7. Standard HHT: energy, entropy, mean amplitude per IMF (15 features)
  8. AM-HHT: Spectral Flatness IMF filtering + 80 multi-domain features
     - IMF features (energy, entropy, amplitude) — 24 features
     - Band ratios (θ/β, δ/α, γ/total) — 6 features
     - Frequency variance & zero-crossing rate — 10 features
     - Statistical moments (kurtosis, skewness) — 5 features
     - Hjorth parameters (activity, mobility, complexity) — 3 features
     - Sample entropy (m=2, m=3) — 2 features
     - Wavelet features (DWT db4, 6 levels) — 18 features
     - Time-domain (MAD, RMS, peak-to-peak, crest factor) — 7 features
     - Non-linear (Hurst, DFA, spectral flatness) — 5 features

Phase 4 — Classification & Evaluation
  9. Random Forest (500 trees, balanced weights) + SVM comparison
  10. Cross-validation + confusion matrices + ROC curves
  11. Full benchmark: FFT vs DWT vs HHT vs AM-HHT

Phase 5 — Seizure Onset Detection
  12. Sliding window → theta/beta ratio evolution → transition point detection
```

---

## How to Run

### Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn PyEMD tqdm
```

### Usage

Place `data.csv` (Bonn University EEG dataset) in the project directory, then run scripts in order:

```bash
python 01_load_and_explore.py           # EDA and visualization
python 02_test_hht_single_sample.py     # CEEMDAN + HH spectrogram
python 03_extract_hht_features.py       # Extract HHT features
python 04_train_model.py                # Train RF + SVM
python 04_5_snr_analysis.py             # SNR analysis
python 05_adaptive_hht_ORIGINAL.py      # Extract AM-HHT features (original method)
python 06_compare_methods.py            # Compare HHT vs AM-HHT
python 07_preictal_prediction.py        # Seizure onset detection
python 10_FINAL_comparison_all_methods.py  # Full benchmark
```

---

## Technical Details

### Spectral Flatness (Original Contribution)

```
SF = geometric_mean(spectrum) / arithmetic_mean(spectrum)
```

- **SF ≈ 1** → flat spectrum (white noise) → **discard IMF**
- **SF ≈ 0** → peaked spectrum (structured signal) → **keep IMF**
- **Threshold: 0.7** — IMFs with SF > 0.7 are excluded

### Feature Categories (80 total)

| Category | Count | Description |
|---|---|---|
| IMF Features | 24 | Energy, entropy, mean amplitude (8 IMFs) |
| Band Ratios | 6 | θ/β, δ/α, γ/total, α/β, θ/α, high-γ% |
| Frequency Variance | 5 | Instantaneous frequency variance per IMF |
| Zero-Crossing Rate | 5 | Per-IMF zero-crossing rate |
| Statistical Moments | 5 | Kurtosis, skewness, std |
| Hjorth Parameters | 3 | Activity, mobility, complexity |
| Sample Entropy | 2 | m=2 and m=3 |
| Wavelet (DWT) | 18 | db4 wavelet, 6 decomposition levels |
| Time-Domain | 7 | MAD, RMS, peak-to-peak, crest/shape/impulse factor |
| Non-linear + Spectral | 5 | Hurst exponent, DFA, spectral flatness, mean freq, bandwidth |

### Dataset

Bonn University EEG: 11,500 segments, 178 samples each (≈1 second at 173.61 Hz). Binary: 9,200 normal + 2,300 seizure (4:1 imbalance, handled with balanced class weights).

---

## Tech Stack

Python · NumPy · SciPy · Scikit-learn · PyEMD (CEEMDAN) · Matplotlib · Seaborn

---

## References

1. Huang, N.E. et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis.
2. Andrzejak, R.G. et al. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity.
3. Acharya, U.R. et al. (2013). Automated EEG analysis of epilepsy: A review.

---

## License

This project is available for reference and educational purposes.
