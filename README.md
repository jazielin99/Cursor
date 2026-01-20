# PSA Card Grading Model

AI-powered PSA card grading prediction system using ensemble learning with adaptive features.

## Model Performance (5-Fold Cross-Validation)

**Current Results** (10,288 images, random splits):

| Metric | Performance |
|--------|-------------|
| **Exact Match** | **53.8%** (SD: 0.8%) |
| **Within 1 Grade** | **73.3%** (SD: 0.7%) |
| **Within 2 Grades** | **84.3%** (SD: 1.1%) |

### Per-Grade Exact Match Accuracy

| Grade | Accuracy | Correct/Total | Notes |
|-------|----------|---------------|-------|
| PSA 1 | **70.6%** | 591/837 | Strong - distinctive damage |
| PSA 2 | 34.5% | 225/652 | Often confused with 1, 3 |
| PSA 3 | **69.4%** | 784/1129 | Strong |
| PSA 4 | **74.4%** | 719/966 | Best performer |
| PSA 5 | 27.0% | 169/626 | Hardest grade - middle zone |
| PSA 6 | 56.9% | 813/1428 | Good - largest class |
| PSA 7 | 47.1% | 643/1365 | Confused with 6, 8 |
| PSA 8 | 28.7% | 290/1009 | Confused with 7, 9 |
| PSA 9 | 50.2% | 602/1199 | Moderate |
| PSA 10 | **65.3%** | 703/1077 | Strong - pristine is distinctive |

### Key Insights

- **Best performers**: PSA 4 (74.4%), PSA 1 (70.6%), PSA 3 (69.4%), PSA 10 (65.3%)
- **Challenging grades**: PSA 5 (27.0%), PSA 8 (28.7%), PSA 2 (34.5%)
- **Pattern**: Extreme grades (1-4, 10) are easier; middle grades (5-8) are harder
- **73% within-1**: Most errors are only off by one grade

### Model Components

| Component | Status | Purpose |
|-----------|--------|---------|
| Random Forest Ensemble | ✅ Active | 500 trees, 500 top features |
| Advanced Features (v4) | ✅ Active | HOG, LBP, corners, centering |
| 5-Fold CV | ✅ Active | Robust evaluation |

## Quick Start

### Step 1: Create Data Manifest (Prevents Leakage)

```bash
python scripts/data_management/create_data_manifest.py \
    --data-dir data/training \
    --output data/data_manifest.csv \
    --create-splits
```

This creates:
- `data_manifest.csv` - Full manifest with duplicate flags
- `data_manifest_clean.csv` - Deduplicated images only
- `data_manifest_splits.csv` - Grouped CV folds

### Step 2: Extract Features

```bash
# Advanced features (Adaptive ROI + Art-Box Centering)
python scripts/feature_extraction/extract_advanced_features.py

# CNN features (MobileNetV2 embeddings, 1,280 dims)
python scripts/feature_extraction/extract_cnn_features_batch.py
```

### Step 3: Train Models

**Option A: Ensemble Model (Recommended for best accuracy)**
```bash
Rscript training/train_ensemble_model.R
```

**Option B: Single Tiered Model (Faster)**
```bash
Rscript training/train_tiered_model.R
```

### Step 4: Make Predictions

```r
# Ensemble prediction (highest accuracy)
source("Prediction_New/predict_ensemble.R")
result <- predict_grade_ensemble("path/to/card.jpg", use_tta = TRUE)
print_ensemble_prediction(result)

# Single model prediction (faster)
source("Prediction_New/predict_new.R")
result <- predict_grade("path/to/card.jpg")
```

## Model Architecture

### Ensemble Model (`train_ensemble_model.R`)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
    │ Model 1 │     │ Model 2 │ ... │ Model 5 │  (Diverse configs)
    │seed=42  │     │seed=123 │     │seed=999 │
    │feat=365 │     │feat=300 │     │feat=350 │
    └────┬────┘     └────┬────┘     └────┬────┘
         │                │                │
         └────────────────┼────────────────┘
                          │ Average probabilities
                          ▼
              ┌───────────────────────┐
              │ Confusion-Pair Check  │  (If borderline 30-70%)
              │ 6↔7, 7↔8, 8↔9, 9↔10  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │ Temperature Scaling   │  (Per-tier calibration)
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │ Ordinal Post-Process  │  (Prefer adjacent grades)
              └───────────┬───────────┘
                          │
                    ┌─────▼─────┐
                    │  OUTPUT   │
                    │ PSA Grade │
                    └───────────┘
```

### Key Components

| Component | Purpose | Expected Impact |
|-----------|---------|-----------------|
| 5-Model Ensemble | Reduce variance, diverse views | +2-3% |
| Confusion-Pair Specialists | Better boundary decisions | +1-2% |
| Ordinal Loss | Prefer adjacent errors | +0.5-1% |
| Temperature Calibration | Better probability estimates | +0.5-1% |
| TTA (Test-Time Aug) | Reduce scan variance | +1-2% |
| Leakage-Free CV | Accurate metrics | True measurement |

## Feature Engineering

### Advanced Features (v4) - 6,400+ dimensions

| Category | Features | Description |
|----------|----------|-------------|
| HOG | ~6,100 | Edge/corner shape patterns |
| LBP | 26 | Surface texture analysis |
| Art-Box Centering | 11 | Pixel-perfect 55/45 ratio |
| Adaptive Corners | 36 | Contour-based corner crops + whitening |
| Corner Sharpness | 35 | Gradient magnitude per corner |
| LoG Kurtosis | 4 | Scratch/glare detection |
| High-res Corner | 80 | Original-resolution analysis |
| LAB Center | 2 | Perceptual lightness |

### CNN Features - 1,280 dimensions

MobileNetV2 embeddings provide deep visual patterns that complement engineered features.

## Data Management

### Preventing Leakage

The data manifest prevents common issues:

1. **Near-Duplicate Detection**: Perceptual hashing finds similar scans
2. **Grouped CV**: Same card (base_id) always in same fold
3. **Front/Back Pairing**: Tracks paired images for future penalty system

```bash
# Check your data quality
python scripts/data_management/create_data_manifest.py --data-dir data/training
```

### Confusion Analysis

After training, analyze errors for targeted improvement:

```bash
# Generate predictions first, then analyze
python scripts/analysis/confusion_analysis.py \
    --predictions results/predictions.csv \
    --output analysis/
```

Outputs:
- `confusion_matrix.png` - Visual heatmap
- `per_grade_accuracy.csv` - Breakdown by grade
- `confusion_report.md` - Recommendations

## Project Structure

```
├── data/
│   ├── training/              # Training images (PSA_1 through PSA_10)
│   ├── data_manifest.csv      # Full manifest with duplicate flags
│   └── data_manifest_clean.csv # Deduplicated images
│
├── models/
│   ├── ensemble_model.rds     # 5-model ensemble + specialists
│   ├── tiered_model.rds       # Single tiered model
│   ├── advanced_features.csv  # Extracted features
│   └── cnn_features_mobilenetv2.csv
│
├── scripts/
│   ├── feature_extraction/
│   │   ├── extract_advanced_features.py   # Main extractor (v4)
│   │   ├── extract_features_tta.py        # Test-time augmentation
│   │   └── extract_cnn_features_*.py      # CNN features
│   ├── data_management/
│   │   └── create_data_manifest.py        # Deduplication + CV splits
│   ├── analysis/
│   │   └── confusion_analysis.py          # Error analysis
│   └── llm_integration/
│       └── llm_grading_assistant.py       # LLM visual auditor
│
├── training/
│   ├── train_ensemble_model.R   # ★ Best accuracy (60%+ target)
│   └── train_tiered_model.R     # Single model (faster)
│
├── Prediction_New/
│   ├── predict_ensemble.R       # ★ Ensemble prediction
│   └── predict_new.R            # Single model prediction
│
├── evaluation/
│   └── evaluate_tiered_cv.R     # Cross-validation
│
├── ios_app/                     # iOS app for mobile grading
│   ├── backend/api_server.py
│   └── PSAGrader/*.swift
│
└── old_versions/                # Archived previous versions
```

## Running Evaluation

To get actual performance numbers:

```bash
# Train ensemble and see CV results
Rscript training/train_ensemble_model.R

# Results will show:
# - 5-fold cross-validation metrics
# - Per-grade exact match accuracy
# - Saved to models/ensemble_cv_results.csv
```

## iOS App

Mobile app for taking photos and getting predictions:

```bash
# Start backend
cd ios_app/backend
pip install -r requirements.txt
python api_server.py

# Then run iOS app in Xcode
```

See [ios_app/README.md](ios_app/README.md) for setup details.

## Improvement Roadmap

### Implemented ✅

- [x] Adaptive ROI patching (contour-based corners)
- [x] Art-Box mathematical centering
- [x] Binary Triage architecture
- [x] 5-model ensemble with diverse configs
- [x] Confusion-pair specialists
- [x] Ordinal-aware training
- [x] Temperature calibration
- [x] Test-time augmentation
- [x] Data manifest with deduplication
- [x] Grouped CV (leakage prevention)
- [x] LLM visual auditor integration
- [x] iOS mobile app

### High Priority (Likely +5-15% accuracy) 📋

- [ ] **CNN Feature Fusion**: Concatenate MobileNetV2 embeddings (1,280 dims) with engineered features
- [ ] **Back-of-card dataset**: Paired front/back images for penalty system
- [ ] **Card-type specialists**: Pokemon, sports, modern vs vintage
- [ ] **More training data**: Current dataset may have too much visual variance
- [ ] **Higher resolution analysis**: Extract corner features at higher resolution

### Medium Priority 📋

- [ ] Active learning loop (flag uncertain samples)
- [ ] Core ML model (offline iOS predictions)
- [ ] Gradient-based saliency maps (explainability)
- [ ] Fine-tuned CNN backbone on grading task

## Requirements

- **Python**: 3.8+
- **R**: 4.0+
- **TensorFlow**: 2.x

```bash
# Python dependencies
pip install -r requirements.txt

# R packages
install.packages(c("ranger", "randomForest"))
```

## License

MIT License
