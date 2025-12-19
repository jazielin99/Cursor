# ==============================================================================
# PSA Card Grading Model - Package Setup
# ==============================================================================
# This script installs and loads all required packages for the PSA grading model

# ------------------------------------------------------------------------------
# Install Required Packages
# ------------------------------------------------------------------------------

#' Install packages if not already installed
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) {
    cat("Installing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, repos = "https://cloud.r-project.org/")
  } else {
    cat("All CRAN packages already installed.\n")
  }
}

# Core packages for data manipulation and visualization
core_packages <- c(
  "tidyverse",      # Data manipulation (dplyr, ggplot2, tidyr, etc.)
  "magrittr",       # Pipe operators
  "caret",          # Machine learning utilities
  "data.table",     # Fast data manipulation
  "jsonlite",       # JSON handling
  "yaml"            # YAML configuration
)

# Image processing packages
image_packages <- c(
  "magick",         # Image processing
  "imager",         # Image manipulation
  "jpeg",           # JPEG handling
  "png",            # PNG handling
  "EBImage"         # Image processing (Bioconductor)
)

# Deep learning packages
dl_packages <- c(
  "keras3",         # Deep learning with TensorFlow backend
  "tensorflow",     # TensorFlow interface
  "reticulate"      # Python interface
)

# Additional ML packages
ml_packages <- c(
  "randomForest",   # Random Forest
  "xgboost",        # Gradient Boosting
  "glmnet",         # Regularized regression
  "e1071"           # SVM and other ML algorithms
)

# Visualization packages
viz_packages <- c(
  "ggplot2",        # Grammar of graphics
  "gridExtra",      # Grid arrangement
  "viridis",        # Color palettes
  "corrplot",       # Correlation plots
  "pheatmap"        # Heatmaps
)

# Reporting packages
report_packages <- c(
  "knitr",          # Report generation
  "rmarkdown"       # R Markdown
)

# Install CRAN packages
cat("=== Installing Core Packages ===\n")
install_if_missing(core_packages)

cat("\n=== Installing Image Processing Packages ===\n")
# Note: EBImage is from Bioconductor
cran_image_packages <- image_packages[image_packages != "EBImage"]
install_if_missing(cran_image_packages)

# Install EBImage from Bioconductor if not installed
if (!("EBImage" %in% installed.packages()[, "Package"])) {
  cat("Installing EBImage from Bioconductor...\n")
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org/")
  }
  BiocManager::install("EBImage", ask = FALSE, update = FALSE)
}

cat("\n=== Installing Deep Learning Packages ===\n")
install_if_missing(dl_packages)

cat("\n=== Installing ML Packages ===\n")
install_if_missing(ml_packages)

cat("\n=== Installing Visualization Packages ===\n")
install_if_missing(viz_packages)

cat("\n=== Installing Report Packages ===\n")
install_if_missing(report_packages)

# ------------------------------------------------------------------------------
# Setup TensorFlow/Keras
# ------------------------------------------------------------------------------

setup_tensorflow <- function() {
  cat("\n=== Setting up TensorFlow ===\n")
  
  # Check if Python is available
  tryCatch({
    library(reticulate)
    
    # Try to use existing Python installation
    python_path <- Sys.which("python3")
    if (python_path == "") {
      python_path <- Sys.which("python")
    }
    
    if (python_path != "") {
      cat("Found Python at:", python_path, "\n")
      use_python(python_path, required = FALSE)
    }
    
    # Check TensorFlow
    library(tensorflow)
    
    # Try to install TensorFlow if not available
    tryCatch({
      tf_version <- tf$version$VERSION
      cat("TensorFlow version:", tf_version, "\n")
    }, error = function(e) {
      cat("TensorFlow not found. Attempting to install...\n")
      cat("You may need to run: tensorflow::install_tensorflow()\n")
      cat("Or install TensorFlow manually via pip: pip install tensorflow\n")
    })
    
    # Setup Keras
    library(keras3)
    cat("Keras loaded successfully.\n")
    
    return(TRUE)
  }, error = function(e) {
    cat("Error setting up TensorFlow/Keras:\n")
    cat(e$message, "\n")
    cat("\nPlease install TensorFlow manually:\n")
    cat("1. Install Python (3.8+)\n")
    cat("2. pip install tensorflow\n")
    cat("3. Run tensorflow::install_tensorflow() in R\n")
    return(FALSE)
  })
}

# ------------------------------------------------------------------------------
# Load All Packages
# ------------------------------------------------------------------------------

load_packages <- function() {
  cat("\n=== Loading Required Packages ===\n")
  
  # Suppress startup messages
  suppressPackageStartupMessages({
    library(tidyverse)
    library(magrittr)
    library(caret)
    library(data.table)
    library(jsonlite)
    library(magick)
    library(jpeg)
    library(png)
  })
  
  # Load deep learning packages with error handling
  tryCatch({
    suppressPackageStartupMessages({
      library(reticulate)
      library(tensorflow)
      library(keras3)
    })
    cat("All packages loaded successfully (including deep learning).\n")
  }, error = function(e) {
    cat("Deep learning packages not available.\n")
    cat("The model will fall back to traditional ML methods.\n")
  })
}

# ------------------------------------------------------------------------------
# System Information
# ------------------------------------------------------------------------------

print_system_info <- function() {
  cat("\n=== System Information ===\n")
  cat("R version:", R.version$version.string, "\n")
  cat("Platform:", R.version$platform, "\n")
  cat("OS:", Sys.info()["sysname"], Sys.info()["release"], "\n")
  
  # Check available memory
  if (Sys.info()["sysname"] == "Linux") {
    mem_info <- system("free -h | grep Mem", intern = TRUE)
    cat("Memory:", mem_info, "\n")
  }
  
  # Check for GPU (CUDA)
  tryCatch({
    library(tensorflow)
    gpus <- tf$config$list_physical_devices("GPU")
    if (length(gpus) > 0) {
      cat("GPU available:", length(gpus), "device(s)\n")
    } else {
      cat("No GPU detected. Training will use CPU.\n")
    }
  }, error = function(e) {
    cat("TensorFlow not available for GPU check.\n")
  })
}

# ------------------------------------------------------------------------------
# Main Setup
# ------------------------------------------------------------------------------

run_setup <- function() {
  cat("========================================\n")
  cat("PSA Card Grading Model - Setup\n")
  cat("========================================\n\n")
  
  # Print system info
  print_system_info()
  
  # Setup TensorFlow
  tf_ready <- setup_tensorflow()
  
  # Load packages
  load_packages()
  
  cat("\n========================================\n")
  if (tf_ready) {
    cat("Setup complete! Deep learning ready.\n")
  } else {
    cat("Setup complete with limitations.\n")
    cat("Deep learning not available.\n")
  }
  cat("========================================\n")
  
  return(tf_ready)
}

# Run setup when sourced
if (interactive()) {
  cat("Run 'run_setup()' to install and configure all packages.\n")
} else {
  run_setup()
}
