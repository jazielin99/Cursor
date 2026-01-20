# PSA Card Grading Model

An AI-powered PSA card grading prediction system using a **tiered** architecture, combining **CNN features + advanced engineered features**.

## Performance Summary (Latest Evaluation)

Cross-validation results on 8,123 images (evaluation subset; repository may contain more training images):

| Metric | Accuracy |
|---|---:|
| **Tier 1 (Low/Mid/High)** | **77.5%** |
| **Exact Grade Match** | **57.6%** |
| **Within 1 Grade** | **73.8%** |
| **Within 2 Grades** | **86.9%** |
| **PSA 9 vs 10** | **59.6%** |

### Per-Fold Results (5-Fold CV, partial)

| Fold | Tier1 | Exact | Within-1 | Within-2 | 9v10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.7% | 56.7% | 73.4% | 86.9% | 60.6% |
| 2 | 77.2% | 58.5% | 74.3% | 87.0% | 58.5% |

## Project Structure

```
├── data/
│   └── training/           # Training images organized by grade
│       ├── PSA_1/          # 985 images
│       ├── PSA_1.5/        # 160 images
│       ├── PSA_2/          # 750 images
│       ├── PSA_3/          # 1,235 images
│       ├── PSA_4/          # 1,840 images
│       ├── PSA_5/          # 1,066 images
│       ├── PSA_6/          # 2,089 images
│       ├── PSA_7/          # 2,158 images
│       ├── PSA_8/          # 1,737 images
│       ├── PSA_9/          # 1,991 images
│       └── PSA_10/         # 1,759 images
│
├── models/                 # Trained models and extracted features
│   ├── tiered_model.rds           # Main tiered classifier
│   ├── high_grade_specialist.rds  # PSA 8/9/10 specialist
│   ├── psa_9_vs_10.rds            # Binary 9 vs 10 classifier
│   ├── advanced_features_v3.csv   # Extracted engineered features
│   └── cnn_features_mobilenetv2.csv  # CNN bottleneck features
│
├── scripts/                # Python scripts
│   ├── feature_extraction/
│   │   ├── extract_advanced_features_v3.py  # 6,312 engineered features
│   │   ├── extract_advanced_features_v2.py  # 6,298 features (legacy)
│   │   ├── extract_cnn_features_batch.py    # MobileNetV2 (1,280 features)
│   │   └── extract_cnn_features_single.py   # Single-image CNN extraction
│   └── data_collection/
│       ├── scrape_comc.R
│       └── scrape_comc_curl.sh
│
├── training/               # R training scripts
│   ├── train_tiered_model.R       # ★ Recommended: Tiered system
│   ├── train_balanced_model.R     # Basic balanced RF
│   ├── train_corner_model.R       # Corner-focused features
│   ├── train_ensemble_cv.R        # RF + XGBoost ensemble
│   ├── train_keras_generator.R    # Keras/TensorFlow CNN
│   └── train_hybrid_fast.R        # PCA-based hybrid (legacy)
│
├── evaluation/             # Evaluation scripts
│   └── evaluate_tiered_cv.R       # K-fold cross-validation
│
├── R/                      # Core R functions
│   ├── main.R                     # Entry point
│   ├── config.R                   # Configuration
│   ├── grading_standards.R        # PSA grade definitions
│   ├── 01_setup.R - 05_prediction.R  # Pipeline modules
│   └── crop_slabs.R               # Slab cropping utility
│
├── examples/               # Example usage scripts
│   ├── 01_basic_usage.R
│   ├── 02_advanced_training.R
│   ├── 03_image_collection_helper.R
│   └── 04_traditional_ml_fallback.R
│
└── Prediction_New/         # Prediction interface
    ├── predict_new.R              # Main prediction script
    └── predict_grade.R            # Alternative predictor
```

## Quick Start

### 1. Install Dependencies

**Python:**
```bash
pip install -r requirements.txt
```

**R:**
```r
install.packages(c("ranger", "randomForest", "magick", "xgboost", "keras3", "smotefamily"))
```

### 2. Extract Features (Required for Training)

```bash
# From project root:

# Advanced engineered features (6,312 dims) - takes ~10-15 minutes
python3 scripts/feature_extraction/extract_advanced_features_v3.py

# CNN features (MobileNetV2, 1,280 dims) - takes ~5-10 minutes
python3 scripts/feature_extraction/extract_cnn_features_batch.py
```

### 3. Train the Model

```bash
# Recommended: Tiered model (run from project root)
Rscript training/train_tiered_model.R
```

### 4. Make Predictions

```r
source("Prediction_New/predict_new.R")

# Single prediction
result <- predict_grade("path/to/card.jpg")
print(result)

# Batch prediction
results <- predict_batch("folder/")
```

## Model Architecture (Tiered System)

### Overview
- **No PCA Trap**: Uses feature-importance selection (top 365) to preserve critical grading signals
- **Tier 1**: Routes cards into Low (1-4) / Mid (5-7) / High (8-10)
- **Tier 2**: Specialist models per tier for exact grade prediction
- **9 vs 10 Specialist**: Dedicated binary classifier for the hardest distinction

### Feature Engineering

| Feature Type | Count | Description |
|---|---:|---|
| HOG | ~6,100 | Edge/corner shape patterns |
| LBP | 26 | Surface texture analysis |
| Centering | 11 | Border ratio detection |
| Corner Sharpness | 35 | Gradient magnitude per corner |
| LoG | 15 | Surface defect detection |
| LoG Kurtosis | 4 | Scratch/glare outlier detection |
| Corner Circularity | 12 | Corner rounding metric |
| High-res Corner | 80 | Original-resolution corner analysis |
| LAB Center | 2 | Perceptual lightness stats |
| Patch Features | 12 | Fixed-size corner attention |
| **CNN (MobileNetV2)** | 1,280 | Deep visual patterns |
| **Total** | **~7,600** | After merging all sources |

## Training Scripts Comparison

| Script | Model Type | Best For | Accuracy |
|---|---|---|---:|
| `train_tiered_model.R` | Tiered RF + Specialists | Production use | ~57% exact |
| `train_balanced_model.R` | Basic RF | Quick baseline | ~45% exact |
| `train_corner_model.R` | Corner-focused RF | Corner analysis | ~48% exact |
| `train_ensemble_cv.R` | RF + XGBoost | Ensemble learning | ~49% exact |
| `train_keras_generator.R` | MobileNetV2 CNN | GPU environments | Varies |

## Evaluation

Run cross-validation to evaluate model performance:

```bash
# 5-fold CV (recommended, but slow)
Rscript evaluation/evaluate_tiered_cv.R 5

# 2-fold CV (faster)
Rscript evaluation/evaluate_tiered_cv.R 2 models/my_eval_results
```

Results are saved to `models/` as CSV and TXT files.

## Dataset Statistics

| Grade | Images | % of Total |
|---|---:|---:|
| PSA 1 | 386 | 2.8% |
| PSA 1.5 | 160 | 1.2% |
| PSA 2 | 383 | 2.8% |
| PSA 3 | 415 | 3.0% |
| PSA 4 | 1,837 | 13.5% |
| PSA 5 | 1,066 | 7.8% |
| PSA 6 | 2,063 | 15.1% |
| PSA 7 | 1,722 | 12.6% |
| PSA 8 | 1,736 | 12.7% |
| PSA 9 | 1,990 | 14.6% |
| PSA 10 | 1,759 | 12.9% |
| **Total** | **13,517** | 100% |

Note: Some images may be excluded during feature extraction due to format issues.

## Key Insights

- **Best for**: Distinguishing high grades (7-10) from low grades (1-4)
- **Challenging**: Adjacent grade differentiation (e.g., PSA 9 vs 10)
- **Tier 1 accuracy**: 77.5% shows the model reliably categorizes card quality
- **Within-2-grades**: 86.9% means predictions are usually close to correct

## Real-World Sanity Tests

The prediction script includes built-in tests:

```r
source("Prediction_New/predict_new.R")

# Test rotation invariance (5° rotation shouldn't change prediction much)
test_result <- rotation_invariance_test("path/to/card.jpg", degrees = 5)

# Test lighting sensitivity (warm vs white lighting)
light_test <- lighting_check_test("path/to/card.jpg")
```

## Next Enhancements (Roadmap)

1. **Cross-model weighted consensus voting** - Combine tiered + CNN predictions
2. **Strategic ROI zoom** - High-res patches for micro-defect detection
3. **Color space analysis (LAB)** - Print-line/silvering/fading signals
4. **Edge-to-border contrast ratio** - Border edge sharpness measurement
5. **Label smoothing** - Reduce overfitting for 9 vs 10 distinction

## Requirements

- Python 3.8+
- R 4.0+
- TensorFlow 2.x
- See `requirements.txt` for Python packages

## License

MIT License
