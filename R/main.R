# ==============================================================================
# PSA Card Grading Model - Main Script
# ==============================================================================
# This is the main entry point for using the PSA Card Grading Model
#
# Usage:
#   1. First run the setup: source("R/main.R"); setup_environment()
#   2. Add images to data/training/<GRADE> directories
#   3. Train the model: train_psa_model()
#   4. Make predictions: predict_card("path/to/image.jpg")
#
# ==============================================================================

# Set working directory to project root if needed
if (basename(getwd()) == "R") {
  setwd("..")
}

# ------------------------------------------------------------------------------
# Source All Required Scripts
# ------------------------------------------------------------------------------

cat("Loading PSA Card Grading Model...\n")

# Source all R scripts
script_dir <- file.path(getwd(), "R")

source(file.path(script_dir, "config.R"))
source(file.path(script_dir, "grading_standards.R"))

# These will be sourced when needed to avoid loading keras if not installed
# source(file.path(script_dir, "01_setup.R"))
# source(file.path(script_dir, "02_data_preparation.R"))
# source(file.path(script_dir, "03_model_definition.R"))
# source(file.path(script_dir, "04_training.R"))
# source(file.path(script_dir, "05_prediction.R"))

# ------------------------------------------------------------------------------
# Main Functions
# ------------------------------------------------------------------------------

#' Setup the environment (install packages)
#' @export
setup_environment <- function() {
  source(file.path(getwd(), "R", "01_setup.R"))
  run_setup()
}

#' Train the PSA grading model
#' @param architecture Model architecture ("mobilenet", "resnet", "efficientnet", "custom")
#' @param epochs Number of training epochs
#' @param batch_size Batch size
#' @param learning_rate Learning rate
#' @export
train_psa_model <- function(architecture = "mobilenet",
                             epochs = 50,
                             batch_size = 32,
                             learning_rate = 0.0001) {
  
  # Source required scripts
  source(file.path(getwd(), "R", "02_data_preparation.R"))
  source(file.path(getwd(), "R", "03_model_definition.R"))
  source(file.path(getwd(), "R", "04_training.R"))
  
  # Run training pipeline
  result <- run_training_pipeline(
    architecture = architecture,
    epochs = epochs,
    batch_size = batch_size,
    learning_rate = learning_rate
  )
  
  return(result)
}

#' Predict PSA grade for a card image
#' @param image_path Path to the card image
#' @param model_path Optional path to trained model
#' @export
predict_card <- function(image_path, model_path = NULL) {
  
  # Source required scripts
  source(file.path(getwd(), "R", "02_data_preparation.R"))
  source(file.path(getwd(), "R", "05_prediction.R"))
  
  # Make prediction
  result <- quick_predict(image_path, model_path)
  
  return(invisible(result))
}

#' Predict PSA grades for all cards in a directory
#' @param image_dir Directory containing card images
#' @param model_path Optional path to trained model
#' @export
predict_cards <- function(image_dir, model_path = NULL) {
  
  # Source required scripts
  source(file.path(getwd(), "R", "02_data_preparation.R"))
  source(file.path(getwd(), "R", "05_prediction.R"))
  
  # Make predictions
  results <- quick_predict_directory(image_dir, model_path)
  
  return(results)
}

#' Show PSA grading standards
#' @export
show_grading_standards <- function() {
  print_grade_summary()
}

#' Show centering requirements table
#' @export
show_centering_requirements <- function() {
  table <- get_centering_table()
  print(table)
  return(invisible(table))
}

#' Get information about a specific PSA grade
#' @param grade Numeric grade value (1-10, or 1.5)
#' @export
get_grade_details <- function(grade) {
  info <- get_grade_info(grade)
  
  cat("\n")
  cat("=", rep("=", 50), "\n", sep = "")
  cat(sprintf("PSA %s (%s) - %s\n", info$grade, info$name, info$full_name))
  cat("=", rep("=", 50), "\n", sep = "")
  
  cat("\nDescription:\n")
  cat(strwrap(info$description, width = 60), sep = "\n")
  
  cat("\nCentering Requirements:\n")
  cat("  Front:", info$centering$front, "\n")
  cat("  Back:", info$centering$back, "\n")
  
  cat("\nKey Attributes:\n")
  for (attr in info$key_attributes) {
    cat("  •", attr, "\n")
  }
  
  cat("\nAllowed Defects:\n")
  for (defect in info$defects_allowed) {
    cat("  -", defect, "\n")
  }
  
  cat("\n")
  
  return(invisible(info))
}

#' Check data directory structure
#' @export
check_data_structure <- function() {
  
  training_dir <- get_config("training_dir")
  classes <- get_config("grade_classes")
  
  cat("\n=== Data Directory Structure ===\n\n")
  cat("Training directory:", training_dir, "\n\n")
  
  total_images <- 0
  
  for (class_name in classes) {
    class_dir <- file.path(training_dir, class_name)
    
    if (dir.exists(class_dir)) {
      files <- list.files(class_dir, pattern = "\\.(jpg|jpeg|png|gif|bmp)$", 
                          ignore.case = TRUE)
      n_files <- length(files)
      status <- ifelse(n_files > 0, 
                       sprintf("✓ %d images", n_files), 
                       "○ empty")
      total_images <- total_images + n_files
    } else {
      status <- "✗ missing"
    }
    
    cat(sprintf("  %s: %s\n", class_name, status))
  }
  
  cat(sprintf("\nTotal training images: %d\n", total_images))
  
  if (total_images == 0) {
    cat("\n⚠ No training images found!\n")
    cat("Please add card images to the appropriate grade folders.\n")
    cat("See README.md for instructions.\n")
  } else if (total_images < 100) {
    cat("\n⚠ Low number of training images.\n")
    cat("For best results, aim for at least 100 images per grade.\n")
  }
  
  cat("\n")
}

#' Print help information
#' @export
psa_help <- function() {
  cat("
========================================
PSA Card Grading Model - Help
========================================

SETUP:
  setup_environment()     Install required R packages

DATA PREPARATION:
  check_data_structure()  Check if training data is set up correctly
  
  Add images to: data/training/<GRADE>/
  Where <GRADE> is one of: PSA_1, PSA_1.5, PSA_2, ..., PSA_10, NO_GRADE

TRAINING:
  train_psa_model()       Train the model with default settings
  train_psa_model(
    architecture = 'mobilenet',  # 'custom', 'resnet', 'efficientnet'
    epochs = 50,
    batch_size = 32,
    learning_rate = 0.0001
  )

PREDICTION:
  predict_card('path/to/image.jpg')     Predict grade for single card
  predict_cards('path/to/folder/')      Predict grades for all cards in folder

GRADING STANDARDS:
  show_grading_standards()       Print all PSA grade definitions
  show_centering_requirements()  Show centering requirements table
  get_grade_details(10)          Get detailed info for PSA 10
  get_grade_details(7)           Get detailed info for PSA 7

CONFIGURATION:
  print_config()           Show current configuration
  get_config('epochs')     Get specific config value
  set_config('epochs', 100)  Set config value

For more details, see README.md
")
}

# ------------------------------------------------------------------------------
# Print Welcome Message
# ------------------------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("PSA Card Grading Model\n")
cat("========================================\n")
cat("\nType 'psa_help()' for usage instructions.\n")
cat("Type 'check_data_structure()' to verify your data setup.\n")
cat("\n")
