# ==============================================================================
# PSA Card Grading Model - Configuration
# ==============================================================================
# This file contains all configuration settings for the model

# ------------------------------------------------------------------------------
# Directory Configuration
# ------------------------------------------------------------------------------
config <- list(
  # Base paths
  base_dir = getwd(),
  data_dir = file.path(getwd(), "data"),
  models_dir = file.path(getwd(), "models"),
  
  # Training data paths
  training_dir = file.path(getwd(), "data", "training"),
  validation_dir = file.path(getwd(), "data", "validation"),
  test_dir = file.path(getwd(), "data", "test"),
  
  # Image settings
  img_width = 224,
  img_height = 224,
  channels = 3,
  
  # Training settings
  batch_size = 32,
  epochs = 50,
  learning_rate = 0.0001,
  validation_split = 0.2,
  
  # Model settings
  dropout_rate = 0.5,
  use_data_augmentation = TRUE,
  
  # Early stopping
  patience = 10,
  min_delta = 0.001,
  
  # Grade classes (PSA grades)
  grade_classes = c(
    "PSA_1",    # Poor
    "PSA_1.5",  # Fair
    "PSA_2",    # Good
    "PSA_3",    # Very Good
    "PSA_4",    # Very Good-Excellent
    "PSA_5",    # Excellent
    "PSA_6",    # Excellent-Mint
    "PSA_7",    # Near Mint
    "PSA_8",    # Near Mint-Mint
    "PSA_9",    # Mint
    "PSA_10",   # Gem Mint
    "NO_GRADE"  # Cards that cannot be graded (N1-N9)
  ),
  
  # Numeric grade values for regression approach (optional)
  grade_values = c(
    "PSA_1" = 1,
    "PSA_1.5" = 1.5,
    "PSA_2" = 2,
    "PSA_3" = 3,
    "PSA_4" = 4,
    "PSA_5" = 5,
    "PSA_6" = 6,
    "PSA_7" = 7,
    "PSA_8" = 8,
    "PSA_9" = 9,
    "PSA_10" = 10,
    "NO_GRADE" = NA
  ),
  
  # Random seed for reproducibility
  seed = 42
)

# Function to get config value
get_config <- function(key) {
  if (key %in% names(config)) {
    return(config[[key]])
  } else {
    stop(paste("Configuration key not found:", key))
  }
}

# Function to update config value
set_config <- function(key, value) {
  config[[key]] <<- value
}

# Print configuration summary
print_config <- function() {
  cat("=== PSA Card Grading Model Configuration ===\n")
  cat("\nDirectories:\n")
  cat("  Base directory:", config$base_dir, "\n")
  cat("  Data directory:", config$data_dir, "\n")
  cat("  Models directory:", config$models_dir, "\n")
  cat("\nImage Settings:\n")
  cat("  Image size:", config$img_width, "x", config$img_height, "\n")
  cat("  Channels:", config$channels, "\n")
  cat("\nTraining Settings:\n")
  cat("  Batch size:", config$batch_size, "\n")
  cat("  Epochs:", config$epochs, "\n")
  cat("  Learning rate:", config$learning_rate, "\n")
  cat("  Validation split:", config$validation_split, "\n")
  cat("\nGrade Classes:", length(config$grade_classes), "classes\n")
  cat("  ", paste(config$grade_classes, collapse = ", "), "\n")
}
