# ============================================
# Train Keras/TensorFlow CNN Model
# Memory-Efficient Version with Generators
# ============================================

library(keras3)
library(tensorflow)

cat("========================================\n")
cat("Training Keras CNN Model (Generator)\n")
cat("========================================\n\n")

cat("TensorFlow version:", as.character(tf$version$VERSION), "\n\n")

# --- Configuration ---
IMG_SIZE <- 224L
BATCH_SIZE <- 32L
EPOCHS <- 25L
LEARNING_RATE <- 0.0001

# --- Use Keras image_dataset_from_directory ---
cat("Loading datasets from directories...\n")

training_dir <- "data/training"

# Create training dataset
train_ds <- image_dataset_from_directory(
  training_dir,
  validation_split = 0.2,
  subset = "training",
  seed = 42L,
  image_size = c(IMG_SIZE, IMG_SIZE),
  batch_size = BATCH_SIZE,
  label_mode = "int"
)

# Create validation dataset
val_ds <- image_dataset_from_directory(
  training_dir,
  validation_split = 0.2,
  subset = "validation",
  seed = 42L,
  image_size = c(IMG_SIZE, IMG_SIZE),
  batch_size = BATCH_SIZE,
  label_mode = "int"
)

class_names <- train_ds$class_names
num_classes <- length(class_names)

cat("Classes:", paste(as.character(class_names), collapse = ", "), "\n")
cat("Number of classes:", num_classes, "\n")

# Prefetch for performance
train_ds <- train_ds$prefetch(buffer_size = tf$data$AUTOTUNE)
val_ds <- val_ds$prefetch(buffer_size = tf$data$AUTOTUNE)

# --- Build MobileNetV2 Model with Data Augmentation ---
cat("\n========================================\n")
cat("Building MobileNetV2 model with augmentation...\n")
cat("========================================\n")

# Data augmentation layers
data_augmentation <- keras_model_sequential(name = "data_augmentation") %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.1) %>%
  layer_random_brightness(0.1)

# Load pre-trained MobileNetV2
base_model <- application_mobilenet_v2(
  input_shape = c(IMG_SIZE, IMG_SIZE, 3L),
  include_top = FALSE,
  weights = "imagenet"
)

# Freeze base model
base_model$trainable <- FALSE

# Build model
inputs <- layer_input(shape = c(IMG_SIZE, IMG_SIZE, 3L))
x <- inputs

# Rescale (MobileNet expects [-1, 1])
x <- layer_rescaling(x, scale = 1/127.5, offset = -1)

# Data augmentation (only during training)
x <- data_augmentation(x)

# Base model
x <- base_model(x, training = FALSE)

# Global pooling and classification head
x <- layer_global_average_pooling_2d(x)
x <- layer_dropout(x, rate = 0.3)
x <- layer_dense(x, units = 256, activation = "relu")
x <- layer_dropout(x, rate = 0.3)
outputs <- layer_dense(x, units = num_classes, activation = "softmax")

model <- keras_model(inputs, outputs)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

cat("Model built successfully.\n")
model

# --- Create callbacks ---
cat("\n========================================\n")
cat("Setting up callbacks...\n")
cat("========================================\n")

callbacks <- list(
  callback_model_checkpoint(
    filepath = "models/psa_mobilenet_gen.keras",
    monitor = "val_accuracy",
    save_best_only = TRUE,
    mode = "max",
    verbose = 1
  ),
  callback_early_stopping(
    monitor = "val_loss",
    patience = 6,
    restore_best_weights = TRUE,
    verbose = 1
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 3,
    min_lr = 1e-7,
    verbose = 1
  )
)

# --- Train Phase 1: Frozen base ---
cat("\n========================================\n")
cat("Phase 1: Training with frozen base (10 epochs)\n")
cat("========================================\n")

history1 <- model %>% fit(
  train_ds,
  validation_data = val_ds,
  epochs = 10L,
  callbacks = callbacks,
  verbose = 2
)

# --- Train Phase 2: Fine-tuning ---
cat("\n========================================\n")
cat("Phase 2: Fine-tuning (unfreeze last 30 layers)\n")
cat("========================================\n")

# Unfreeze last 30 layers
base_model$trainable <- TRUE
n_layers <- length(base_model$layers)
for (i in seq_len(max(1, n_layers - 30))) {
  base_model$layers[[i]]$trainable <- FALSE
}

# Recompile with lower learning rate
model %>% compile(
  optimizer = optimizer_adam(learning_rate = LEARNING_RATE / 10),
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

history2 <- model %>% fit(
  train_ds,
  validation_data = val_ds,
  epochs = 15L,
  callbacks = callbacks,
  verbose = 2
)

# --- Final evaluation ---
cat("\n========================================\n")
cat("Final Evaluation\n")
cat("========================================\n")

eval_result <- model %>% evaluate(val_ds, verbose = 0)
cat(sprintf("Validation Loss: %.4f\n", eval_result[[1]]))
cat(sprintf("Validation Accuracy: %.4f (%.1f%%)\n", eval_result[[2]], eval_result[[2]] * 100))

# --- Save final model ---
cat("\n========================================\n")
cat("Saving model...\n")
cat("========================================\n")

save_model(model, "models/psa_cnn_generator.keras")
saveRDS(as.character(class_names), "models/psa_cnn_generator_classes.rds")

cat("Model saved: models/psa_cnn_generator.keras\n")
cat("Classes saved: models/psa_cnn_generator_classes.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
