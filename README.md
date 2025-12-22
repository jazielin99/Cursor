# PSA Card Grading Model

An AI-powered PSA card grading prediction system using a **tiered** architecture, combining **CNN features + advanced engineered features**.

## Overview (Tiered System)

This repo now supports a **Tiered Model Architecture** (instead of the old PCA-compressed hybrid):
- **CNN Feature Extraction (Python)**: MobileNetV2 (1,280 dims)
- **Advanced Features v2 (Python)**: 6,298 engineered features (HOG/LBP/centering/corners/LoG + v2 upgrades)
- **No PCA Trap**: uses **feature-importance selection** (top 365) so critical signals aren’t discarded
- **Tier 1 (Low/Mid/High)**: routes the card into a specialist head
- **High-grade specialists**: dedicated 8/9/10 model + a PSA 9 vs 10 binary classifier

## Best Performance (Tiered System)

### Reported (from prior tiered-system run)

| Component / Metric | Accuracy |
|---|---:|
| Tier 1 (Low/Mid/High) | 88.3% |
| Overall Exact Match | 44.0% |
| Within 1 Grade | 77.8% |
| Within 2 Grades | 86.8% |
| High-grade specialist (8/9/10) exact | 50% |
| PSA 9 detection (within high-grade) | 96.2% |
| PSA 9 vs 10 binary | 73.2% |

### Reproduced in this repo (fast 2-fold CV on 8,123 images; will vary by split)

| Metric | Accuracy |
|---|---:|
| Tier 1 (Low/Mid/High) | 75.9% |
| Exact match | 55.4% |
| Within 1 grade | 73.0% |
| Within 2 grades | 86.1% |
| PSA 9 vs 10 accuracy | 58.2% |

## Quick Start

### 1. Extract Features (First Time Only)

```bash
# CNN features (MobileNetV2 bottleneck features, includes `path`)
python3 extract_cnn_features_batch.py

# Advanced features v2 (6,298 features, includes `path`)
python3 extract_advanced_features_v2.py
```

### 2. Train the Tiered Model (No PCA)

```bash
Rscript train_tiered_model.R
```

### 3. Make Predictions (Single / Batch)

```r
source("Prediction_New/predict_new.R")

# Single prediction
result <- predict_grade("path/to/card.jpg")

# Batch prediction
results <- predict_batch("folder/")
```

## Directory Structure

```
├── data/
│   └── training/           # Training images organized by grade
│       ├── PSA_1/
│       ├── PSA_1.5/
│       ├── PSA_2/
│       ...
│       └── PSA_10/
├── models/                  # Trained models and extracted features
├── R/                       # Core R functions
├── examples/                # Example usage scripts
├── Prediction_New/          # Prediction scripts
├── extract_advanced_features.py    # Legacy advanced features (v1)
├── extract_advanced_features_v2.py # Advanced features v2 (6,298)
├── extract_cnn_features_batch.py   # CNN feature extraction (dataset)
├── extract_cnn_features_single.py  # CNN feature extraction (single image)
├── train_tiered_model.R            # Tiered training (feature importance + SMOTE)
├── train_keras_generator.R         # TensorFlow/Keras CNN training
├── train_ensemble_cv.R             # RF + XGBoost ensemble
├── train_balanced_model.R          # Basic RF with balanced sampling
└── train_corner_model.R            # RF with corner-focused features
```

## Training Scripts

### Recommended: `train_tiered_model.R`
**Tiered system (no PCA trap)**
- Feature-importance selection (top 365 features)
- Tier 1 routing: Low/Mid/High
- Specialists: Low (1–4), Mid (5–7), High (8–10)
- PSA 9 vs 10 binary specialist
- Uses SMOTE (with a safe fallback if the SMOTE package is missing)

### Alternative: `train_keras_generator.R`
**Deep learning approach**
- MobileNetV2 transfer learning
- Memory-efficient data generators
- Good for GPU environments

### Legacy: `train_ensemble_cv.R`
**Traditional ML ensemble**
- Random Forest + XGBoost + Stacking
- 5-fold cross-validation
- ~49% exact accuracy (RF only)

## Feature Engineering

### CNN Features (1,280 features)
- MobileNetV2 bottleneck features
- Pre-trained on ImageNet
- Captures high-level visual patterns

### HOG Features (~6,000 features)
- Histogram of Oriented Gradients
- Detects edges and corner shapes
- PSA 10: sharp HOG signatures

### LBP Features (26 features)
- Local Binary Patterns
- Texture analysis for surface wear
- Detects microscopic defects

### Centering Features (11 features)
- Canny edge detection for borders
- Top/Bottom/Left/Right ratios
- Centering quality score

### Corner Sharpness (35 features)
- Contour area measurement
- Edge density per corner
- Gradient magnitude analysis

### LoG Features (15 features)
- Laplacian of Gaussian
- Surface defect detection
- "Blob" and "pit" detection

### v2 Additions (96 features)
- **LoG Kurtosis**: identifies scratch/glare outliers (PSA 10 = flatter distribution; scratched = higher kurtosis)
- **Corner Circularity**: \(4\pi \times \text{Area} / \text{Perimeter}^2\) detects rounding / fraying
- **High-resolution corner analysis**: corners analyzed at original resolution (micro-fraying sensitivity)

## Model Files

| File | Description | Size |
|------|-------------|------|
| `psa_ensemble_cv.rds` | RF + XGBoost ensemble | ~37MB |
| `psa_mobilenet_gen.keras` | Keras MobileNetV2 | ~14MB |
| `psa_rf_balanced.rds` | Balanced Random Forest | ~3MB |
| `psa_rf_3class_balanced.rds` | 3-class (Low/Mid/High) RF | ~6MB |
| `tiered_model.rds` | Tiered model (Low/Mid/High routing + specialists) | varies |
| `high_grade_specialist.rds` | Dedicated 8/9/10 specialist | varies |
| `psa_9_vs_10.rds` | PSA 9 vs PSA 10 binary classifier | varies |

## Requirements

### R Packages
```r
install.packages(c("randomForest", "ranger", "magick", "xgboost", "keras3", "smotefamily"))
```

### Python Packages
```bash
pip install -r requirements.txt
```

## Data Collection

Use the included scraping scripts to collect training data:
```bash
# Via curl
./scrape_comc_curl.sh

# Via R
Rscript scrape_comc.R
```

## Notes

- **Class Imbalance**: PSA 1.5 has only 80 samples vs 1,400+ for PSA 6
- **Best for**: Distinguishing high grades (7-9) from low grades (1-4)
- **Challenging**: Exact differentiation between adjacent grades (e.g., PSA 9 vs PSA 10)
- **3-Class Model**: Achieves 76% accuracy for Low (1-4) / Mid (5-7) / High (8-10)

## Real-World Sanity Tests (Recommended)

- **Rotation invariance test**: rotate a PSA 10 image ~5° and re-run `predict_grade()`. If the result flips, your HOG-like features may be too orientation-sensitive.
- **Lighting check**: compare “warm” vs “white” lighting. **LoG Kurtosis** can treat glare as scratch-like outliers if illumination changes.

## Output Improvement: Confidence / Upgrade Hints

Now that a `psa_9_vs_10` specialist exists, you can surface “borderline” predictions:
- Example: if PSA 9 vs 10 probability is near 50–55%, return **“PSA 9 (Potential 10 Upgrade)”** rather than a hard label.

## Next Enhancements (Roadmap)

1. **Cross-model weighted consensus voting**
   - Run the tiered model, high-grade specialist, and Keras CNN simultaneously.
   - Weight by confidence to reduce borderline “coin flip” errors.

2. **Strategic ROI zoom (micro-defect patches)**
   - Before final classification, extract 5 high-res patches: 4 corners + center holo area (e.g. 512×512 or original res).
   - Train a small “micro-defect” CNN on those patches (3-layer shallow net works well).

3. **Color space analysis (LAB & HSV)**
   - Add mean/variance of **L-channel** (lightness) for print-line/silvering/fading signals.

4. **Edge-to-border contrast ratio**
   - Measure border edge “step function” sharpness (Michelson contrast + gradient steepness).

5. **Automated gray-market label smoothing**
   - Replace hard 9 vs 10 labels with softened targets (e.g. 0.9/0.1) to reduce overfitting to unique defects.

### What to do first?

| Improvement | Effort | Impact on Exact Match |
|---|---:|---:|
| ROI patching (zoom) | High | Very High (helps 9 vs 10) |
| L-channel analysis | Low | Medium (surface/fading) |
| Weighted voting | Medium | High (reduces near-misses) |
| Edge contrast | Medium | Medium (centering/cut quality) |

## License

MIT License
