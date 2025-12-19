# PSA Card Grading Model

An R-based image classification model for predicting PSA (Professional Sports Authenticator) card grades using deep learning.

## Overview

This project provides a complete framework for:
- Training a deep learning model to classify trading card images by PSA grade
- Making predictions on new card images
- Understanding PSA grading standards programmatically

### Supported Grades

| Grade | Name | Description |
|-------|------|-------------|
| PSA 10 | GEM-MT | Gem Mint - Virtually perfect card |
| PSA 9 | MINT | Mint - Superb condition, one minor flaw |
| PSA 8 | NM-MT | Near Mint-Mint - Super high-end card |
| PSA 7 | NM | Near Mint - Slight surface wear |
| PSA 6 | EX-MT | Excellent-Mint - Visible surface wear |
| PSA 5 | EX | Excellent - Minor corner rounding |
| PSA 4 | VG-EX | Very Good-Excellent - Noticeable wear |
| PSA 3 | VG | Very Good - Some corner rounding |
| PSA 2 | GOOD | Good - Obvious surface wear |
| PSA 1.5 | FR | Fair - Extreme wear, must be intact |
| PSA 1 | PR | Poor - Eye appeal nearly vanished |
| NO_GRADE | N/A | Cards that cannot be graded (N1-N9 codes) |

## Project Structure

```
workspace/
├── R/                          # R scripts
│   ├── main.R                  # Main entry point
│   ├── config.R                # Configuration settings
│   ├── grading_standards.R     # PSA grade definitions
│   ├── 01_setup.R              # Package installation
│   ├── 02_data_preparation.R   # Data loading & preprocessing
│   ├── 03_model_definition.R   # Model architectures
│   ├── 04_training.R           # Training functions
│   └── 05_prediction.R         # Prediction functions
├── data/
│   ├── training/               # Training images by grade
│   │   ├── PSA_10/
│   │   ├── PSA_9/
│   │   ├── PSA_8/
│   │   ├── PSA_7/
│   │   ├── PSA_6/
│   │   ├── PSA_5/
│   │   ├── PSA_4/
│   │   ├── PSA_3/
│   │   ├── PSA_2/
│   │   ├── PSA_1.5/
│   │   ├── PSA_1/
│   │   └── NO_GRADE/
│   ├── validation/             # Validation images
│   └── test/                   # Test images
├── models/                     # Saved models
├── examples/                   # Example scripts
└── README.md
```

## Quick Start

### 1. Setup Environment

```r
# Load the main script
source("R/main.R")

# Install required packages
setup_environment()
```

### 2. Prepare Training Data

Add card images to the appropriate grade folders:
- `data/training/PSA_10/` - Images of PSA 10 graded cards
- `data/training/PSA_9/` - Images of PSA 9 graded cards
- etc.

**Recommended**: At least 50-100 images per grade for good results.

**Image Requirements**:
- Format: JPG, JPEG, PNG, GIF, or BMP
- Resolution: Any size (will be resized to 224x224)
- Content: Clear, well-lit photos of the card front

### 3. Train the Model

```r
# Basic training
result <- train_psa_model()

# With custom settings
result <- train_psa_model(
  architecture = "mobilenet",  # "custom", "resnet", "efficientnet"
  epochs = 50,
  batch_size = 32,
  learning_rate = 0.0001
)
```

### 4. Make Predictions

```r
# Predict single card
predict_card("path/to/card_image.jpg")

# Predict all cards in a folder
results <- predict_cards("path/to/card_folder/")
```

## Detailed Usage

### Check Data Structure

```r
check_data_structure()
```

Output:
```
=== Data Directory Structure ===

Training directory: /workspace/data/training

  PSA_10: ✓ 150 images
  PSA_9: ✓ 120 images
  PSA_8: ✓ 180 images
  ...

Total training images: 1200
```

### View PSA Grading Standards

```r
# Show all grades
show_grading_standards()

# Get centering requirements
show_centering_requirements()

# Get detailed info for specific grade
get_grade_details(10)  # PSA 10
get_grade_details(7)   # PSA 7
```

### Model Architectures

The package supports multiple model architectures:

| Architecture | Description | Best For |
|-------------|-------------|----------|
| `mobilenet` | MobileNetV2 transfer learning | General use (recommended) |
| `resnet` | ResNet50 transfer learning | Large datasets |
| `efficientnet` | EfficientNetB0 transfer learning | Balanced accuracy/speed |
| `custom` | Custom CNN from scratch | Experimentation |

### Configuration

```r
# View current config
print_config()

# Get specific value
get_config("epochs")
get_config("batch_size")

# Set value
set_config("epochs", 100)
set_config("batch_size", 64)
```

## Collecting Training Data

Since the PSA website cannot be directly scraped (Cloudflare protection), here are ways to collect training data:

### Option 1: eBay Listings

1. Search for "PSA [grade] [card type]" on eBay
2. Download images of cards with visible PSA labels
3. Sort into appropriate grade folders

### Option 2: PSA Population Report

1. Use the PSA Population Report to find graded cards
2. Search for those specific cards on collector sites
3. Download verified graded card images

### Option 3: Personal Collection

1. Photograph your own PSA graded cards
2. Use consistent lighting and angle
3. Include both front and back if possible

### Option 4: Trading Card Forums

1. Visit collector forums and subreddits
2. Many collectors share photos of their graded cards
3. Always respect usage rights

### Image Collection Tips

- **Consistency**: Use similar lighting and angles
- **Quality**: Higher resolution is better
- **Variety**: Include different card types, years, and conditions
- **Balance**: Try to have similar numbers per grade
- **Focus**: Center the card in the image
- **Clean background**: Avoid cluttered backgrounds

## Model Training Best Practices

### Data Augmentation

The model automatically applies:
- Random horizontal flips
- Rotation (±15°)
- Brightness adjustments
- Zoom variations

### Handling Imbalanced Data

Class weights are automatically computed to handle imbalanced datasets.

### Transfer Learning

The recommended approach uses transfer learning:
1. **Phase 1**: Train classification head with frozen base
2. **Phase 2**: Fine-tune all layers with low learning rate

### Early Stopping

Training automatically stops when validation loss stops improving (default: 10 epochs patience).

## Evaluation

After training, evaluate your model:

```r
# Load model and test data
source("R/05_prediction.R")
loaded <- load_grading_model("models/psa_grading_model.keras")

# Load test dataset
test_data <- load_training_dataset("data/test")

# Evaluate
evaluation <- evaluate_model(
  loaded$model,
  test_data$images,
  test_data$labels,
  loaded$class_names
)

# Print results
print_evaluation_summary(evaluation)

# Plot confusion matrix
plot_confusion_matrix(evaluation$confusion_matrix, loaded$class_names)
```

## Troubleshooting

### "No images found in training directory"

Make sure:
1. Images are in the correct folders: `data/training/PSA_X/`
2. File extensions are correct (.jpg, .jpeg, .png, .gif, .bmp)
3. Files are not hidden (no leading dot)

### "TensorFlow not available"

Run:
```r
install.packages("tensorflow")
tensorflow::install_tensorflow()
```

Or install TensorFlow via pip:
```bash
pip install tensorflow
```

### "Out of memory"

Reduce batch size:
```r
set_config("batch_size", 16)
train_psa_model(batch_size = 16)
```

### Low accuracy

Try:
1. Add more training images
2. Use data augmentation
3. Try different architectures
4. Increase training epochs
5. Adjust learning rate

## Dependencies

### Required R Packages

```r
# Core
tidyverse, magrittr, caret, data.table, jsonlite

# Image processing
magick, jpeg, png

# Deep learning
keras3, tensorflow, reticulate

# Machine learning (fallback)
randomForest, xgboost, glmnet, e1071

# Visualization
ggplot2, gridExtra, viridis
```

### System Requirements

- R 4.0+
- Python 3.8+ (for TensorFlow)
- TensorFlow 2.x
- 8GB+ RAM recommended
- GPU optional but recommended for faster training

## PSA Grading Standards Reference

### Centering Requirements

| Grade | Front | Back |
|-------|-------|------|
| PSA 10 | 55/45 - 60/40 | 75/25 |
| PSA 9 | 60/40 or better | 90/10 or better |
| PSA 8 | 65/35 or better | 90/10 or better |
| PSA 7 | 70/30 or better | 90/10 or better |
| PSA 6 | 80/20 or better | 90/10 or better |
| PSA 5 | 85/15 or better | 90/10 or better |
| PSA 4 | 85/15 or better | 90/10 or better |
| PSA 3-1 | 90/10 or better | 90/10 or better |

### No Grade (N) Codes

| Code | Reason |
|------|--------|
| N1 | Evidence of Trimming |
| N2 | Evidence of Restoration |
| N3 | Evidence of Recoloration |
| N4 | Questionable Authenticity |
| N5 | Altered Stock |
| N6 | Minimum Size Requirement |
| N7 | Evidence of Cleaning |
| N8 | Miscut |
| N9 | Don't Grade |

### Half-Point Grades

Cards between PSA 2 and PSA 9 may receive half-point increases (e.g., PSA 8.5) if they exhibit high-end qualities within their grade. Centering is the primary factor.

### Eye Appeal

PSA graders may adjust grades based on subjective "eye appeal" - a card with exceptional visual presentation may receive a higher grade even if it's borderline on technical specs.

## License

This project is for educational and personal use. PSA grading standards are proprietary to Professional Sports Authenticator.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Disclaimer

This model is for educational purposes only. It should not be used as a substitute for professional grading services. Actual PSA grades can only be assigned by PSA's trained graders.
