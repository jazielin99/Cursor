# ==============================================================================
# PSA Card Grading Model - Advanced Training Example
# ==============================================================================
# This script demonstrates advanced training options and customization

# Set working directory to project root
setwd("/workspace")

# Source all required scripts
source("R/config.R")
source("R/grading_standards.R")
source("R/02_data_preparation.R")
source("R/03_model_definition.R")
source("R/04_training.R")
source("R/05_prediction.R")

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Customize configuration
set_config("epochs", 100)
set_config("batch_size", 32)
set_config("learning_rate", 0.0001)
set_config("dropout_rate", 0.5)
set_config("validation_split", 0.2)

# Print configuration
print_config()

# ------------------------------------------------------------------------------
# Step 1: Load and Analyze Data
# ------------------------------------------------------------------------------

cat("\n=== Step 1: Loading Data ===\n\n")

# Load training dataset
dataset <- load_training_dataset()

# Print detailed summary
print_dataset_summary(dataset, dataset$class_names)

# Compute and display class weights
class_weights <- compute_class_weights(dataset$labels)
cat("\nClass weights for handling imbalance:\n")
print(round(class_weights, 3))

# ------------------------------------------------------------------------------
# Step 2: Split Data
# ------------------------------------------------------------------------------

cat("\n=== Step 2: Splitting Data ===\n\n")

# Split with stratification
split <- split_dataset(
  dataset$images,
  dataset$labels,
  validation_split = 0.2,
  stratify = TRUE,
  seed = 42
)

train_images <- split$train$images
train_labels <- split$train$labels
val_images <- split$validation$images
val_labels <- split$validation$labels

cat(sprintf("Training samples: %d\n", dim(train_images)[1]))
cat(sprintf("Validation samples: %d\n", dim(val_images)[1]))

# Verify stratification
cat("\nTraining label distribution:\n")
print(table(train_labels))

cat("\nValidation label distribution:\n")
print(table(val_labels))

# ------------------------------------------------------------------------------
# Step 3: Create Augmented Dataset (Optional)
# ------------------------------------------------------------------------------

cat("\n=== Step 3: Data Augmentation ===\n\n")

# Create augmented training data
# This creates 2 augmented versions of each training image
augmented <- create_augmented_dataset(train_images, train_labels, augmentation_factor = 2)

cat(sprintf("Original training images: %d\n", dim(train_images)[1]))
cat(sprintf("Augmented training images: %d\n", dim(augmented$images)[1]))

# Combine original and augmented
train_images_aug <- abind::abind(train_images, augmented$images, along = 1)
train_labels_aug <- c(train_labels, augmented$labels)

cat(sprintf("Total training images: %d\n", dim(train_images_aug)[1]))

# ------------------------------------------------------------------------------
# Step 4: Compare Model Architectures
# ------------------------------------------------------------------------------

cat("\n=== Step 4: Model Architecture Comparison ===\n\n")

# Get model parameters
num_classes <- length(dataset$class_names)
input_shape <- c(224, 224, 3)

# Create different architectures (without training)
architectures <- c("custom", "mobilenet")  # Add "resnet", "efficientnet" if needed

for (arch in architectures) {
  cat("\n--- ", arch, " ---\n")
  
  tryCatch({
    model <- create_psa_model(
      architecture = arch,
      input_shape = input_shape,
      num_classes = num_classes
    )
    
    # Count parameters
    trainable <- sum(sapply(model$trainable_weights, function(w) prod(dim(w))))
    non_trainable <- sum(sapply(model$non_trainable_weights, function(w) prod(dim(w))))
    
    cat(sprintf("  Trainable parameters: %s\n", format(trainable, big.mark = ",")))
    cat(sprintf("  Non-trainable parameters: %s\n", format(non_trainable, big.mark = ",")))
    
  }, error = function(e) {
    cat(sprintf("  Error: %s\n", e$message))
  })
}

# ------------------------------------------------------------------------------
# Step 5: Train with MobileNet (Recommended)
# ------------------------------------------------------------------------------

cat("\n=== Step 5: Training with MobileNet ===\n\n")

# Create MobileNet model
model <- create_mobilenet_model(
  input_shape = input_shape,
  num_classes = num_classes,
  trainable_layers = NULL,  # Freeze all base layers initially
  dropout_rate = 0.5
)

# Compile with low learning rate
model <- compile_model(model, learning_rate = 0.001)

# Print summary
print_model_summary(model)

# Create callbacks
callbacks <- create_callbacks(
  model_path = file.path(get_config("models_dir"), "psa_model_mobilenet.keras"),
  log_dir = file.path(get_config("models_dir"), "logs", "mobilenet"),
  patience = 10
)

# Phase 1: Train with frozen base (transfer learning)
cat("\n--- Phase 1: Training with frozen base ---\n\n")

history1 <- train_model(
  model,
  train_images, train_labels,
  val_images, val_labels,
  epochs = 10,
  batch_size = 32,
  class_weights = class_weights,
  callbacks = unname(callbacks)
)

# Phase 2: Fine-tune all layers
cat("\n--- Phase 2: Fine-tuning ---\n\n")

# Unfreeze all layers
for (layer in model$layers) {
  layer$trainable <- TRUE
}

# Recompile with lower learning rate
model <- compile_model(model, learning_rate = 0.00001)

history2 <- train_model(
  model,
  train_images, train_labels,
  val_images, val_labels,
  epochs = 20,
  batch_size = 32,
  class_weights = class_weights,
  callbacks = unname(callbacks)
)

# ------------------------------------------------------------------------------
# Step 6: Evaluate Model
# ------------------------------------------------------------------------------

cat("\n=== Step 6: Model Evaluation ===\n\n")

# Evaluate on validation set
evaluation <- evaluate_model(model, val_images, val_labels, dataset$class_names)

# Print summary
print_evaluation_summary(evaluation)

# Plot confusion matrix
plot_confusion_matrix(
  evaluation$confusion_matrix, 
  dataset$class_names,
  save_path = file.path(get_config("models_dir"), "confusion_matrix.png")
)

# ------------------------------------------------------------------------------
# Step 7: Save Model
# ------------------------------------------------------------------------------

cat("\n=== Step 7: Saving Model ===\n\n")

# Save complete model
model_path <- file.path(get_config("models_dir"), "psa_grading_model.keras")
save_model(model, model_path)

# Save class names
classes_path <- gsub("\\.keras$", "_classes.rds", model_path)
saveRDS(dataset$class_names, classes_path)

cat("Model saved to:", model_path, "\n")
cat("Classes saved to:", classes_path, "\n")

# ------------------------------------------------------------------------------
# Step 8: Visualize Training History
# ------------------------------------------------------------------------------

cat("\n=== Step 8: Training Visualization ===\n\n")

# Plot training history
plot_training_history(
  history2,
  save_path = file.path(get_config("models_dir"), "training_history.png")
)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("Advanced Training Complete!\n")
cat("========================================\n")
cat("\n")
cat("Final Results:\n")
cat(sprintf("  Accuracy: %.2f%%\n", evaluation$accuracy * 100))
cat(sprintf("  Loss: %.4f\n", evaluation$loss))
cat("\n")
cat("Model saved to:", model_path, "\n")
cat("\n")
cat("To make predictions:\n")
cat('  predict_card("path/to/image.jpg")\n')
cat("\n")
