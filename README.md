# PSA Card Grading Model

An AI-powered PSA card grading prediction system using machine learning and deep learning techniques.

## Overview

This project uses a **Hybrid CNN-Random Forest** architecture that combines:
- **CNN Feature Extraction**: MobileNetV2 pre-trained on ImageNet extracts 1,280 visual features
- **Advanced Hand-Crafted Features**: HOG, LBP, centering analysis, corner sharpness, LoG surface detection (6,200+ features)
- **PCA Dimensionality Reduction**: Reduces to 200 components while retaining 95% variance
- **Ranger Random Forest**: Fast ordinal regression for grade prediction

## Best Model Performance (5-Fold Cross-Validation)

| Metric | Accuracy |
|--------|----------|
| **Exact Match** | 23.2% |
| **Within 1 Grade** | 60.5% |
| **Within 2 Grades** | 85.0% |

## Quick Start

### 1. Extract Features (First Time Only)

```bash
# Extract CNN features (MobileNetV2 bottleneck features)
python3 extract_cnn_features_batch.py

# Extract advanced features (HOG, LBP, centering, corners, LoG)
python3 extract_advanced_features.py
```

### 2. Train the Hybrid Model

```bash
Rscript train_hybrid_fast.R
```

### 3. Make Predictions

```r
source("Prediction_New/predict_grade.R")
predict_card("path/to/your/card.jpg")
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
├── extract_advanced_features.py    # HOG, LBP, centering extraction
├── extract_cnn_features_batch.py   # CNN feature extraction
├── train_hybrid_fast.R             # Main training script (recommended)
├── train_keras_generator.R         # TensorFlow/Keras CNN training
├── train_ensemble_cv.R             # RF + XGBoost ensemble
├── train_balanced_model.R          # Basic RF with balanced sampling
└── train_corner_model.R            # RF with corner-focused features
```

## Training Scripts

### Recommended: `train_hybrid_fast.R`
**Best accuracy with advanced features**
- Combines CNN + advanced features
- Uses PCA for dimensionality reduction
- Ranger RF with ordinal regression
- ~85% within 2 grades accuracy

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

## Model Files

| File | Description | Size |
|------|-------------|------|
| `hybrid_pca_model.rds` | Best model (CNN + Advanced + PCA) | ~700MB |
| `psa_ensemble_cv.rds` | RF + XGBoost ensemble | ~37MB |
| `psa_mobilenet_gen.keras` | Keras MobileNetV2 | ~14MB |
| `psa_rf_balanced.rds` | Balanced Random Forest | ~3MB |
| `psa_rf_3class_balanced.rds` | 3-class (Low/Mid/High) RF | ~6MB |

## Requirements

### R Packages
```r
install.packages(c("randomForest", "ranger", "magick", "xgboost", "keras3"))
```

### Python Packages
```bash
pip install tensorflow opencv-python-headless scikit-image scipy pandas
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

## License

MIT License
