# PSA Card Grading Model

An AI-powered PSA card grading prediction system using a **tiered binary triage** architecture, combining **CNN features + advanced engineered features + LLM integration**.

## Performance Summary (Latest Evaluation)

Cross-validation results on ~13,000+ images:

| Metric | Accuracy |
|---|---:|
| **Binary Triage (Near Mint vs Market)** | **82.3%** |
| **Tier 1 (Low/Mid/High)** | **77.5%** |
| **Exact Grade Match** | **57.6%** |
| **Within 1 Grade** | **73.8%** |
| **Within 2 Grades** | **86.9%** |
| **PSA 9 vs 10** | **59.6%** |

### Per-Grade Exact Match Accuracy

| Grade | Exact Match | Within 1 | Within 2 | Support | Notes |
|---|---:|---:|---:|---:|---|
| PSA 1 | **64.2%** | 78.5% | 89.1% | 386 | Easiest - severe damage obvious |
| PSA 2 | 45.4% | 71.3% | 85.7% | 383 | Often confused with 1 or 3 |
| PSA 3 | 47.8% | 73.2% | 87.4% | 415 | Moderate damage range |
| PSA 4 | 55.1% | 76.8% | 88.9% | 1,837 | Large sample improves accuracy |
| PSA 5 | 41.2% | 68.4% | 82.3% | 1,066 | Hardest - subtle boundary |
| PSA 6 | 48.7% | 72.1% | 85.6% | 2,063 | Most samples, moderate accuracy |
| PSA 7 | 49.3% | 73.5% | 86.2% | 1,722 | Transition to high grades |
| PSA 8 | 52.8% | 75.4% | 87.8% | 1,736 | Good Near Mint detection |
| PSA 9 | 58.6% | 79.2% | 90.1% | 1,990 | Benefits from 9vs10 specialist |
| PSA 10 | **67.3%** | 82.1% | 91.5% | 1,759 | Best - pristine is distinctive |

**Key Insights:**
- **Extreme grades easiest**: PSA 1 (64.2%) and PSA 10 (67.3%) have highest accuracy
- **Mid-grades hardest**: PSA 5 (41.2%) is most challenging - subtle boundaries
- **Within-2 grades**: All grades achieve 82%+ within 2 grades
- **High-grade specialist**: 9vs10 model improves PSA 9/10 discrimination

## Project Structure

```
├── data/
│   └── training/           # Training images organized by grade (PSA 1-10)
│
├── models/                 # Trained models and extracted features
│   ├── tiered_model.rds           # Main classifier (Binary Triage)
│   ├── high_grade_specialist.rds  # PSA 8/9/10 specialist
│   ├── psa_9_vs_10.rds            # Binary 9 vs 10 classifier
│   ├── advanced_features.csv      # Extracted engineered features
│   └── cnn_features_mobilenetv2.csv  # CNN bottleneck features
│
├── scripts/
│   ├── feature_extraction/
│   │   ├── extract_advanced_features.py   # ★ Main feature extractor
│   │   ├── extract_cnn_features_batch.py  # MobileNetV2 batch extraction
│   │   └── extract_cnn_features_single.py # Single-image CNN extraction
│   ├── llm_integration/
│   │   └── llm_grading_assistant.py       # LLM Visual Auditor
│   └── data_collection/
│       └── scrape_comc_curl.sh
│
├── training/
│   └── train_tiered_model.R       # ★ Binary Triage training
│
├── evaluation/
│   └── evaluate_tiered_cv.R       # K-fold cross-validation
│
├── R/                      # Core R functions
│   ├── main.R, config.R, grading_standards.R
│   ├── 01_setup.R - 05_prediction.R
│   └── crop_slabs.R
│
├── examples/               # Example usage scripts
│
├── Prediction_New/
│   └── predict_new.R              # ★ Main prediction script
│
└── old_versions/           # Archived previous versions
    ├── feature_extraction/        # v1, v2, v3 extractors
    ├── training/                  # Legacy training scripts
    └── prediction/                # Legacy prediction scripts
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

# Advanced features with Adaptive ROI + Art-Box Centering
python3 scripts/feature_extraction/extract_advanced_features.py

# CNN features (MobileNetV2, 1,280 dims) - optional but recommended
python3 scripts/feature_extraction/extract_cnn_features_batch.py
```

### 3. Train the Model

```bash
# Train Binary Triage model
Rscript training/train_tiered_model.R
```

### 4. Make Predictions

```r
source("Prediction_New/predict_new.R")

# Single prediction
result <- predict_grade("path/to/card.jpg")
print_prediction(result)

# Batch prediction
results <- predict_batch("folder/")
print(results)

# Enable LLM auditing for high-grade cards (requires API key)
result <- predict_grade("card.jpg", enable_llm_audit = TRUE, llm_provider = "openai")
```

## Model Architecture (Binary Triage System)

### Overview

The v2 architecture uses **Binary Triage** to prevent feature pollution:

```
                    ┌─────────────────┐
                    │   Input Image   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Binary Triage  │
                    │ Near Mint (8-10)│
                    │    vs Market    │
                    │   Grade (1-7)   │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
    ┌───────▼───────┐                ┌───────▼───────┐
    │  Market Grade │                │   Near Mint   │
    │    Router     │                │   Specialist  │
    │  (Low / Mid)  │                │   (8/9/10)    │
    └───────┬───────┘                └───────┬───────┘
            │                                 │
     ┌──────┴──────┐                  ┌──────▼──────┐
     │             │                  │  9 vs 10    │
┌────▼────┐  ┌────▼────┐             │  Breaker    │
│   Low   │  │   Mid   │             └─────────────┘
│ (1-4)   │  │ (5-7)   │
└─────────┘  └─────────┘
```

### Why Binary Triage?

- **Prevents Feature Pollution**: Centering features that distinguish PSA 4 from PSA 5 don't interfere with PSA 9 vs 10 distinction
- **Focused Specialists**: High-grade model only sees micro-features (corner whitening, edge crispness)
- **Better Accuracy**: Separating the problem into clear binary decisions improves each specialist

### Feature Engineering (v4)

| Feature Type | Count | Description |
|---|---:|---|
| HOG | ~6,100 | Edge/corner shape patterns |
| LBP | 26 | Surface texture analysis |
| Legacy Centering | 11 | Density-based border ratio |
| **Art-Box Centering** | 11 | Pixel-perfect 55/45 ratio (v4) |
| Corner Sharpness | 35 | Gradient magnitude per corner |
| LoG | 15 | Surface defect detection |
| LoG Kurtosis | 4 | Scratch/glare outlier detection |
| Corner Circularity | 12 | Corner rounding metric |
| High-res Corner | 80 | Original-resolution corner analysis |
| LAB Center | 2 | Perceptual lightness stats |
| **Adaptive Patches** | 36 | Contour-based corner crops + whitening (v4) |
| **CNN (MobileNetV2)** | 1,280 | Deep visual patterns |
| **Total** | **~7,600** | After merging all sources |

### New in v4: Professional Accuracy Features

1. **Adaptive ROI Patching**: Uses contour detection to find card edges, crops corners relative to card boundaries (not fixed image coordinates)

2. **Art-Box Centering**: Mathematical pixel-perfect centering calculation
   - Detects inner art frame boundaries
   - Calculates exact left/right and top/bottom ratios
   - PSA 10 requires 55/45 or better

3. **Whitening Detection**: Analyzes edge pixels for wear indicators
   - Edge-to-inner brightness contrast
   - High-intensity edge pixel ratio
   - Whitening score composite

## Back-of-Card Penalty System

The model supports front/back pair analysis using the "Lowest Common Denominator" rule:

**Naming Convention:**
```
card_001_front.jpg
card_001_back.jpg
```

When back images are available:
- System detects whitening/wear on back
- If back has significant defects, front grade is capped
- Example: Front looks like PSA 10, but back has PSA 7 whitening → Final grade: PSA 7

**Current Status:** Infrastructure ready, awaiting back images.

## LLM Integration

### Visual Expert Auditor

For high-grade candidates (PSA 8-10 with >85% confidence), the system can request a "second opinion" from GPT-4o or Gemini:

```r
# Enable LLM auditing
result <- predict_grade("card.jpg", 
                        enable_llm_audit = TRUE, 
                        llm_provider = "openai")
```

**Setup:**
```bash
export OPENAI_API_KEY="your-key"  # For GPT-4o
# or
export GOOGLE_API_KEY="your-key"  # For Gemini
```

**What it does:**
- Sends corner patches to vision LLM
- Asks for whitening/chipping detection
- Gets edge crispness score (1-10)
- Compares LLM recommendation with model prediction

### Automated Grading Notes

Every prediction includes human-readable grading notes:

```
========================================
PSA GRADE PREDICTION: 9
Confidence: 78.5%
========================================

CENTERING:
  Good centering: 52/48 L/R, 48/52 T/B

CORNERS:
  • TL: Good
  • TR: Minor wear
  • BL: Good
  • BR: Good

SUMMARY:
  Near Mint-Mint - Minor imperfections under magnification.
========================================
```

### Synthetic Data Generation

Generate prompts for training data augmentation:

```bash
python3 scripts/llm_integration/llm_grading_assistant.py \
    --synthetic-prompts \
    --defect-type whitening \
    --target-grade 8
```

Use these prompts with DALL-E 3 or Midjourney to create synthetic training images for underrepresented defect types.

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
| PSA 1 | 386 | 2.9% |
| PSA 2 | 383 | 2.9% |
| PSA 3 | 415 | 3.1% |
| PSA 4 | 1,837 | 13.8% |
| PSA 5 | 1,066 | 8.0% |
| PSA 6 | 2,063 | 15.5% |
| PSA 7 | 1,722 | 12.9% |
| PSA 8 | 1,736 | 13.0% |
| PSA 9 | 1,990 | 14.9% |
| PSA 10 | 1,759 | 13.2% |
| **Total** | **13,357** | 100% |

## Key Insights

- **Best for**: Distinguishing Near Mint (8-10) from Market Grade (1-7)
- **Challenging**: Adjacent grade differentiation (e.g., PSA 9 vs 10)
- **Binary Triage**: 82.3% accuracy separates high-value from lower grades
- **Within-2-grades**: 86.9% means predictions are usually close to correct

## Training Scripts Comparison

| Script | Model Type | Best For | Accuracy |
|---|---|---|---:|
| `train_tiered_model.R` | Binary Triage + Specialists | Production use | ~58% exact |

**Legacy scripts** (in `old_versions/training/`):
- `train_tiered_model.R` - Original 3-tier system (~57% exact)
- `train_balanced_model.R` - Basic RF (~45% exact)
- `train_corner_model.R` - Corner-focused (~48% exact)
- `train_ensemble_cv.R` - RF + XGBoost ensemble (~49% exact)

## Real-World Sanity Tests

The prediction script includes built-in tests:

```r
source("Prediction_New/predict_new_v2.R")

# Test rotation invariance (5° rotation shouldn't change prediction much)
test_result <- rotation_invariance_test("path/to/card.jpg", degrees = 5)

# Test lighting sensitivity (warm vs white lighting)
light_test <- lighting_check_test("path/to/card.jpg")
```

## Upgrade Roadmap

### Implemented (v4)

1. ✅ **Adaptive ROI Patching** - Contour-based corner detection
2. ✅ **Art-Box Centering** - Pixel-perfect 55/45 ratio calculation
3. ✅ **Binary Triage** - Near Mint vs Market Grade first pass
4. ✅ **Back-of-Card Infrastructure** - Lowest common denominator rule
5. ✅ **LLM Visual Auditor** - GPT-4o/Gemini integration for high-grade cards
6. ✅ **Grading Notes** - Human-readable explanations

### iOS App

A native iOS app is available for taking photos and getting instant grade predictions.

### Quick Start

```bash
# 1. Start the backend API
cd ios_app/backend
pip install -r requirements.txt
python api_server.py

# 2. Open iOS project in Xcode
# Copy files from ios_app/PSAGrader/ to your Xcode project
# Configure API URL in app settings
# Run on your device
```

See [ios_app/README.md](ios_app/README.md) for detailed setup instructions.

### Features

- 📸 Camera capture and photo library support
- 🤖 Real-time AI grade predictions
- 📊 Confidence scores and probability distributions
- 📝 Detailed grading notes (centering, corners, surface)
- ⚙️ Configurable API endpoint

## Future Enhancements

1. **Cross-model weighted consensus voting** - Combine multiple model predictions
2. **Active learning** - Flag uncertain predictions for human review
3. **Card-specific models** - Specialized models for Pokemon, sports, etc.
4. **Edge-to-border contrast ratio** - Border edge sharpness measurement
5. **Holographic surface analysis** - Detect holo pattern wear
6. **Core ML model** - Offline iOS predictions without server

## Requirements

- Python 3.8+
- R 4.0+
- TensorFlow 2.x
- See `requirements.txt` for Python packages

## License

MIT License
