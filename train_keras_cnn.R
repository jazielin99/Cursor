# ============================================
# Train Keras/TensorFlow CNN Model
# With Upsampling for Class Balance
# ============================================

library(keras3)
library(tensorflow)
library(magick)

cat("========================================\n")
cat("Training Keras CNN Model\n")
cat("========================================\n\n")

cat("TensorFlow version:", as.character(tf$version$VERSION), "\n\n")

# --- Configuration ---
IMG_SIZE <- 224L
BATCH_SIZE <- 32L
EPOCHS <- 30L
LEARNING_RATE <- 0.0001

# --- Load and preprocess images ---
load_image <- function(path, target_size = IMG_SIZE) {
  tryCatch({
    img <- image_read(path)
    img <- image_resize(img, paste0(target_size, "x", target_size, "!"))
    img_data <- image_data(img, channels = "rgb")
    img_array <- as.integer(img_data)
    img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1)) / 255.0
    rm(img, img_data); gc(verbose = FALSE)
    return(img_array)
  }, error = function(e) return(NULL))
}

# --- Load training data ---
cat("Loading training data...\n")
training_dir <- "data/training"
class_dirs <- list.dirs(training_dir, recursive = FALSE)
class_dirs <- class_dirs[!grepl("NO_GRADE|backup", class_dirs)]

all_paths <- c()
all_labels <- c()

for (class_dir in class_dirs) {
  class_name <- basename(class_dir)
  files <- list.files(class_dir, pattern = "\\.(jpg|jpeg|png)$", 
                      ignore.case = TRUE, full.names = TRUE)
  files <- files[!grepl("originals_backup", files)]
  if (length(files) > 0) {
    all_paths <- c(all_paths, files)
    all_labels <- c(all_labels, rep(class_name, length(files)))
  }
}

# Sort class names properly
class_names <- sort(unique(all_labels))
class_names <- class_names[order(as.numeric(gsub("PSA_", "", class_names)))]
cat("Classes:", paste(class_names, collapse = ", "), "\n")
cat("Total images:", length(all_paths), "\n")

# --- Upsample minority classes ---
cat("\n========================================\n")
cat("Upsampling minority classes\n")
cat("========================================\n")

class_counts <- table(all_labels)
print(class_counts)

# Target: 80% of max class size
target_count <- floor(max(class_counts) * 0.7)
cat("\nTarget samples per class:", target_count, "\n")

set.seed(42)
upsampled_paths <- all_paths
upsampled_labels <- all_labels

for (class_name in names(class_counts)) {
  current_count <- class_counts[class_name]
  if (current_count < target_count) {
    need <- target_count - current_count
    class_idx <- which(all_labels == class_name)
    sample_idx <- sample(class_idx, need, replace = TRUE)
    upsampled_paths <- c(upsampled_paths, all_paths[sample_idx])
    upsampled_labels <- c(upsampled_labels, rep(class_name, need))
    cat(sprintf("  %s: %d -> %d (+%d)\n", class_name, current_count, current_count + need, need))
  }
}

cat("\nUpsampled total:", length(upsampled_paths), "images\n")

# --- Load images ---
cat("\n========================================\n")
cat("Loading images into memory...\n")
cat("========================================\n")

n_images <- length(upsampled_paths)
X <- array(0, dim = c(n_images, IMG_SIZE, IMG_SIZE, 3))
y <- integer(n_images)

label_map <- setNames(seq_along(class_names) - 1L, class_names)

for (i in seq_len(n_images)) {
  if (i %% 500 == 0) cat("  ", i, "/", n_images, "\n")
  img <- load_image(upsampled_paths[i])
  if (!is.null(img)) {
    X[i, , , ] <- img
    y[i] <- label_map[upsampled_labels[i]]
  }
  if (i %% 2000 == 0) gc(verbose = FALSE)
}

cat("Images loaded:", n_images, "\n")

# --- Split train/validation ---
cat("\n========================================\n")
cat("Splitting train/validation...\n")
cat("========================================\n")

set.seed(42)
n <- nrow(X)
val_idx <- sample(seq_len(n), floor(n * 0.2))
train_idx <- setdiff(seq_len(n), val_idx)

X_train <- X[train_idx, , , , drop = FALSE]
y_train <- y[train_idx]
X_val <- X[val_idx, , , , drop = FALSE]
y_val <- y[val_idx]

cat("Training samples:", nrow(X_train), "\n")
cat("Validation samples:", nrow(X_val), "\n")

rm(X, y); gc()

# --- Build MobileNetV2 Model ---
cat("\n========================================\n")
cat("Building MobileNetV2 model...\n")
cat("========================================\n")

num_classes <- length(class_names)

# Load pre-trained MobileNetV2
base_model <- application_mobilenet_v2(
  input_shape = c(IMG_SIZE, IMG_SIZE, 3L),
  include_top = FALSE,
  weights = "imagenet",
  pooling = "avg"
)

# Freeze base model initially
base_model$trainable <- FALSE

# Build model
inputs <- layer_input(shape = c(IMG_SIZE, IMG_SIZE, 3L))
x <- inputs

# MobileNet preprocessing
x <- tf$keras$applications$mobilenet_v2$preprocess_input(x)
x <- base_model(x, training = FALSE)
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

# --- Compute class weights ---
cat("\n========================================\n")
cat("Computing class weights...\n")
cat("========================================\n")

class_counts_train <- table(y_train)
total <- sum(class_counts_train)
n_classes <- length(class_counts_train)
weights <- total / (n_classes * class_counts_train)
weights <- weights / min(weights)
class_weights <- as.list(weights)
names(class_weights) <- as.character(seq_along(class_weights) - 1)

cat("Class weights:\n")
for (i in seq_along(class_names)) {
  cat(sprintf("  %s: %.2f\n", class_names[i], weights[i]))
}

# --- Create callbacks ---
cat("\n========================================\n")
cat("Setting up callbacks...\n")
cat("========================================\n")

callbacks <- list(
  callback_model_checkpoint(
    filepath = "models/psa_mobilenet_new.keras",
    monitor = "val_accuracy",
    save_best_only = TRUE,
    mode = "max",
    verbose = 1
  ),
  callback_early_stopping(
    monitor = "val_loss",
    patience = 8,
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
  x = X_train,
  y = y_train,
  validation_data = list(X_val, y_val),
  epochs = 10L,
  batch_size = BATCH_SIZE,
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1
)

# --- Train Phase 2: Fine-tuning ---
cat("\n========================================\n")
cat("Phase 2: Fine-tuning (unfreeze last 30 layers)\n")
cat("========================================\n")

# Unfreeze last 30 layers of base model
base_model$trainable <- TRUE
for (i in seq_len(length(base_model$layers) - 30)) {
  base_model$layers[[i]]$trainable <- FALSE
}

# Recompile with lower learning rate
model %>% compile(
  optimizer = optimizer_adam(learning_rate = LEARNING_RATE / 10),
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

history2 <- model %>% fit(
  x = X_train,
  y = y_train,
  validation_data = list(X_val, y_val),
  epochs = 20L,
  batch_size = BATCH_SIZE,
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1
)

# --- Final evaluation ---
cat("\n========================================\n")
cat("Final Evaluation\n")
cat("========================================\n")

eval_result <- model %>% evaluate(X_val, y_val, verbose = 0)
cat(sprintf("Validation Loss: %.4f\n", eval_result[[1]]))
cat(sprintf("Validation Accuracy: %.4f (%.1f%%)\n", eval_result[[2]], eval_result[[2]] * 100))

# Get predictions for confusion matrix
predictions <- model %>% predict(X_val, verbose = 0)
pred_classes <- max.col(predictions) - 1

# Calculate per-class accuracy
cat("\nPer-class accuracy:\n")
for (i in seq_along(class_names)) {
  class_idx <- which(y_val == (i - 1))
  if (length(class_idx) > 0) {
    acc <- mean(pred_classes[class_idx] == y_val[class_idx])
    cat(sprintf("  %s: %.1f%% (%d samples)\n", class_names[i], acc * 100, length(class_idx)))
  }
}

# Within 1/2 grade accuracy
pred_numeric <- as.numeric(gsub("PSA_", "", class_names[pred_classes + 1]))
true_numeric <- as.numeric(gsub("PSA_", "", class_names[y_val + 1]))

within_1 <- mean(abs(pred_numeric - true_numeric) <= 1, na.rm = TRUE)
within_2 <- mean(abs(pred_numeric - true_numeric) <= 2, na.rm = TRUE)

cat(sprintf("\nWithin 1 grade: %.1f%%\n", within_1 * 100))
cat(sprintf("Within 2 grades: %.1f%%\n", within_2 * 100))

# --- Save final model ---
cat("\n========================================\n")
cat("Saving model...\n")
cat("========================================\n")

save_model(model, "models/psa_cnn_upsampled.keras")
saveRDS(class_names, "models/psa_cnn_upsampled_classes.rds")

cat("Model saved: models/psa_cnn_upsampled.keras\n")
cat("Classes saved: models/psa_cnn_upsampled_classes.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
