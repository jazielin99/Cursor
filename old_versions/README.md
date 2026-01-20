# Old Versions Archive

This folder contains previous versions of scripts that have been superseded by newer implementations. These are kept for reference and backwards compatibility.

## Feature Extraction

| File | Description | Superseded By |
|------|-------------|---------------|
| `extract_advanced_features.py` | Original v1 feature extraction | `../scripts/feature_extraction/extract_advanced_features.py` |
| `extract_advanced_features_v2.py` | Added LoG kurtosis, corner circularity | Current version |
| `extract_advanced_features_v3.py` | Added LAB center, fixed corner patches | Current version |

**Current Version (v4)** includes:
- Adaptive ROI patching (contour-based corner detection)
- Art-Box mathematical centering (55/45 ratio)
- Whitening detection features

## Training Scripts

| File | Description | Superseded By |
|------|-------------|---------------|
| `train_tiered_model.R` | Original 3-tier system | `../training/train_tiered_model.R` (Binary Triage) |
| `train_balanced_model.R` | Basic balanced Random Forest | Current tiered model |
| `train_corner_model.R` | Corner-focused features | Current tiered model |
| `train_ensemble_cv.R` | RF + XGBoost ensemble | Current tiered model |
| `train_hybrid_fast.R` | PCA-based hybrid (legacy) | Current tiered model |
| `train_keras_generator.R` | Keras/TensorFlow CNN | CNN features extracted separately |

**Current Version** includes:
- Binary Triage (Near Mint vs Market Grade first)
- Back-of-card penalty infrastructure
- PSA_1.5 grade removed

## Prediction Scripts

| File | Description | Superseded By |
|------|-------------|---------------|
| `predict_new.R` | Original tiered prediction | `../Prediction_New/predict_new.R` |
| `predict_grade.R` | Alternative predictor | `../Prediction_New/predict_new.R` |

**Current Version** includes:
- Binary Triage architecture
- LLM visual auditor integration
- Automated grading notes
- Support for v4 features

## Usage

If you need to use an older version:

```bash
# Example: Use v3 feature extraction
python3 old_versions/feature_extraction/extract_advanced_features_v3.py --data-dir data/training --output-base models/advanced_features_v3

# Example: Use original training
Rscript old_versions/training/train_tiered_model.R
```

## Migration Notes

When upgrading from old versions:
1. Re-extract features using the current `extract_advanced_features.py`
2. Re-train models using `training/train_tiered_model.R`
3. Update predictions to use `Prediction_New/predict_new.R`
