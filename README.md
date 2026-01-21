# PSA Card Grading Model

AI-powered PSA card grading prediction system using machine learning with advanced features.

## Model Performance (5-Fold Cross-Validation)

**Python Model Results** (5,000 images):

| Metric | Performance |
|--------|-------------|
| **Exact Match** | **54.7%** |
| **Within 1 Grade** | **70.9%** |
| **Within 2 Grades** | **84.1%** |

### Per-Grade Exact Match Accuracy

| Grade | Accuracy | Correct/Total | Notes |
|-------|----------|---------------|-------|
| PSA 1 | 69.2% | 346/500 | Strong - distinctive damage |
| PSA 2 | 62.0% | 310/500 | Good |
| PSA 3 | 62.4% | 312/500 | Good |
| PSA 4 | 71.6% | 358/500 | Strong |
| PSA 5 | 52.2% | 261/500 | Moderate |
| PSA 6 | 40.2% | 201/500 | Hard class - mid-grade boundary |
| PSA 7 | 39.2% | 196/500 | Hard class - mid-grade boundary |
| PSA 8 | 39.6% | 198/500 | Hard class - high-grade boundary |
| PSA 9 | 41.8% | 209/500 | Confused with 8, 10 |
| PSA 10 | 68.4% | 342/500 | Strong |

### Model Components

| Component | Status | Purpose |
|-----------|--------|---------|
| Random Forest | ✅ Active | 500 trees, balanced class weights |
| Advanced Features (v4) | ✅ Active | HOG, LBP, corners, centering, art-box |
| Adaptive Corners | ✅ Active | Contour-based corner detection |
| Art-Box Centering | ✅ Active | Pixel-perfect 55/45 ratio measurement |

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

## Back-of-Card Setup

The model supports paired front/back card images. To add back-of-card data:

```bash
# Folder structure is ready:
data/
├── training_front/    # Front images (existing)
│   ├── PSA_1/
│   ├── PSA_2/
│   ...
└── training_back/     # Back images (add your images here)
    ├── PSA_1/
    ├── PSA_2/
    ...

# After adding back images, extract features:
python scripts/feature_extraction/extract_advanced_features.py \
    --data-dir data/training_back \
    --output-base models/advanced_features_back

# The training script will automatically detect and use back features
Rscript training/train_full_pipeline.R
```

## Project Structure

```
├── data/
│   ├── training_front/        # Front card images (PSA_1 through PSA_10)
│   ├── training_back/         # Back card images (same structure)
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

## Web App (Mobile-Friendly)

Access the grader from any device with a browser:

```bash
# Install dependencies
cd webapp
pip install -r requirements.txt

# Run the server
python app.py

# Access at http://localhost:5000
# Or from phone: http://<your-ip>:5000
```

Features:
- Camera capture or photo upload
- Real-time grade predictions with tiered specialist model
- Confidence scores and probability breakdown
- Works on iOS, Android, and desktop

### Retrain the Model

```bash
# Retrain with custom sample size
python webapp/predictor.py --train --samples 300

# Test prediction on a single image
python webapp/predictor.py --image path/to/card.jpg
```

The model is saved to `models/psa_python_model.pkl`

## iOS App (Native)

Native iOS app for offline predictions:

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
- [x] **Mobile web app** (Flask-based, works on any device)
- [x] **CNN Feature Fusion** (MobileNetV2 1,280 dims + engineered features)
- [x] **Back-of-card folder structure** (ready for paired images)
- [x] **Card type tagging** (sports, tcg, unknown in manifest)
- [x] **Cost-sensitive learning** (upweight hard classes)
- [x] **SMOTE oversampling** for PSA 2, 5, 8
- [x] **Uncertainty sampling** for active learning
- [x] **Confusion analysis** with recommendations

### High Priority (Next Steps) 📋

- [ ] **Balance hard class weights**: Current weights cause PSA 8 over-prediction
- [ ] **Collect back-of-card images**: Place in `data/training_back/PSA_X/`
- [ ] **Card-type specialists**: Separate models for sports vs TCG
- [ ] **Review uncertain samples**: Use `scripts/analysis/uncertainty_sampling.py`

### Medium Priority 📋

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
