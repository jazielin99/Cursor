# ==============================================================================
# PSA Card Grading Model - Basic Usage Example
# ==============================================================================
# This script demonstrates the basic usage of the PSA Card Grading Model

# Set working directory to project root
setwd("/workspace")

# Load the main script
source("R/main.R")

# ------------------------------------------------------------------------------
# Step 1: Check Environment Setup
# ------------------------------------------------------------------------------

cat("=== Step 1: Checking Environment ===\n\n")

# Print current configuration
print_config()

# Check data directory structure
check_data_structure()

# ------------------------------------------------------------------------------
# Step 2: Review PSA Grading Standards
# ------------------------------------------------------------------------------

cat("\n=== Step 2: PSA Grading Standards ===\n\n")

# Show all grades summary
show_grading_standards()

# Show centering requirements
cat("\nCentering Requirements:\n")
show_centering_requirements()

# Get detailed info for PSA 10
cat("\nDetailed info for PSA 10:\n")
get_grade_details(10)

# Get detailed info for PSA 7
cat("\nDetailed info for PSA 7:\n")
get_grade_details(7)

# ------------------------------------------------------------------------------
# Step 3: Training (requires images in data/training/)
# ------------------------------------------------------------------------------

cat("\n=== Step 3: Training ===\n\n")

# First, check if we have training data
check_data_structure()

# If you have training images, uncomment this section:
# 
# # Train with default settings (MobileNet architecture)
# result <- train_psa_model()
# 
# # Or train with custom settings
# result <- train_psa_model(
#   architecture = "mobilenet",  # Options: "mobilenet", "resnet", "efficientnet", "custom"
#   epochs = 50,                 # Number of training epochs
#   batch_size = 32,             # Batch size
#   learning_rate = 0.0001       # Learning rate
# )
# 
# # View training history
# print(result$history)

cat("Training skipped - add images to data/training/ folders first.\n")
cat("See README.md for instructions on collecting training data.\n")

# ------------------------------------------------------------------------------
# Step 4: Making Predictions (requires trained model)
# ------------------------------------------------------------------------------

cat("\n=== Step 4: Making Predictions ===\n\n")

# If you have a trained model and test images, uncomment this section:
#
# # Predict single card
# result <- predict_card("path/to/your/card_image.jpg")
#
# # Predict all cards in a folder  
# results <- predict_cards("path/to/card_folder/")
# print(results)
#
# # Save predictions to CSV
# write.csv(results, "predictions.csv", row.names = FALSE)

cat("Prediction skipped - train a model first.\n")

# ------------------------------------------------------------------------------
# Step 5: Understanding Grade Info
# ------------------------------------------------------------------------------

cat("\n=== Step 5: Understanding Grades ===\n\n")

# Access grading standards programmatically
source("R/grading_standards.R")

# Get all allowed defects for PSA 8
cat("Defects allowed for PSA 8:\n")
defects <- get_allowed_defects(8)
for (d in defects) {
  cat("  -", d, "\n")
}

# Get centering requirements
cat("\nCentering for PSA 9:\n")
centering <- get_centering_requirements(9)
cat("  Front:", centering$front, "\n")
cat("  Back:", centering$back, "\n")

# Compare grades
cat("\n\nComparing PSA 10 vs PSA 9:\n")
psa10 <- get_grade_info(10)
psa9 <- get_grade_info(9)

cat("\nPSA 10 (", psa10$full_name, "):\n", sep = "")
cat("  - ", psa10$key_attributes[1], "\n", sep = "")
cat("  - ", psa10$key_attributes[2], "\n", sep = "")

cat("\nPSA 9 (", psa9$full_name, "):\n", sep = "")
cat("  - ", psa9$key_attributes[1], "\n", sep = "")
cat("  - Allows one minor flaw\n", sep = "")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("Basic Usage Example Complete\n")
cat("========================================\n")
cat("\n")
cat("Next steps:\n")
cat("1. Collect training images for each PSA grade\n")
cat("2. Place images in data/training/<GRADE>/ folders\n")
cat("3. Run train_psa_model() to train the model\n")
cat("4. Use predict_card() to classify new cards\n")
cat("\n")
cat("For help: psa_help()\n")
cat("\n")
