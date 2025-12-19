# ==============================================================================
# PSA Card Grading Model - Training
# ==============================================================================
# Functions for training the PSA grading model

# Source dependencies
source(file.path(getwd(), "R", "config.R"))
source(file.path(getwd(), "R", "02_data_preparation.R"))
source(file.path(getwd(), "R", "03_model_definition.R"))

# ------------------------------------------------------------------------------
# Training Callbacks
# ------------------------------------------------------------------------------

#' Create training callbacks
#' @param model_path Path to save best model
#' @param log_dir Directory for TensorBoard logs
#' @param patience Early stopping patience
#' @param min_delta Minimum change to qualify as improvement
#' @return List of Keras callbacks
create_callbacks <- function(model_path = NULL,
                              log_dir = NULL,
                              patience = 10,
                              min_delta = 0.001) {
  
  library(keras3)
  
  callbacks <- list()
  
  # Model checkpoint - save best model
  if (!is.null(model_path)) {
    callbacks$checkpoint <- callback_model_checkpoint(
      filepath = model_path,
      monitor = "val_accuracy",
      save_best_only = TRUE,
      save_weights_only = FALSE,
      mode = "max",
      verbose = 1
    )
  }
  
  # Early stopping
  callbacks$early_stopping <- callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    min_delta = min_delta,
    restore_best_weights = TRUE,
    verbose = 1
  )
  
  # Learning rate reduction on plateau
  callbacks$lr_reduction <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-7,
    verbose = 1
  )
  
  # TensorBoard logging
  if (!is.null(log_dir)) {
    if (!dir.exists(log_dir)) {
      dir.create(log_dir, recursive = TRUE)
    }
    callbacks$tensorboard <- callback_tensorboard(
      log_dir = log_dir,
      histogram_freq = 0,
      write_graph = TRUE,
      update_freq = "epoch"
    )
  }
  
  # Progress callback
  callbacks$progress <- callback_progbar_logger()
  
  return(callbacks)
}

# ------------------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------------------

#' Train PSA grading model
#' @param model Compiled Keras model
#' @param train_images Training images array
#' @param train_labels Training labels (0-indexed integers)
#' @param val_images Validation images array
#' @param val_labels Validation labels
#' @param epochs Number of training epochs
#' @param batch_size Batch size
#' @param class_weights Optional class weights for imbalanced data
#' @param callbacks List of callbacks
#' @param verbose Verbosity level (0, 1, or 2)
#' @return Training history
train_model <- function(model,
                        train_images,
                        train_labels,
                        val_images = NULL,
                        val_labels = NULL,
                        epochs = 50,
                        batch_size = 32,
                        class_weights = NULL,
                        callbacks = NULL,
                        verbose = 1) {
  
  library(keras3)
  
  # Convert labels to one-hot encoding
  num_classes <- length(unique(c(train_labels, val_labels)))
  train_labels_onehot <- to_categorical(train_labels, num_classes)
  
  # Prepare validation data
  validation_data <- NULL
  if (!is.null(val_images) && !is.null(val_labels)) {
    val_labels_onehot <- to_categorical(val_labels, num_classes)
    validation_data <- list(val_images, val_labels_onehot)
  }
  
  # Convert class weights to named list if provided
  if (!is.null(class_weights)) {
    class_weights <- as.list(class_weights)
    names(class_weights) <- as.character(seq_along(class_weights) - 1)
  }
  
  cat("\n=== Training Model ===\n")
  cat("Training samples:", dim(train_images)[1], "\n")
  if (!is.null(val_images)) {
    cat("Validation samples:", dim(val_images)[1], "\n")
  }
  cat("Epochs:", epochs, "\n")
  cat("Batch size:", batch_size, "\n")
  cat("\n")
  
  # Train the model
  history <- model %>% fit(
    x = train_images,
    y = train_labels_onehot,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = validation_data,
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = verbose
  )
  
  return(history)
}

#' Fine-tune a pre-trained model
#' @param model Pre-trained model
#' @param train_images Training images
#' @param train_labels Training labels
#' @param val_images Validation images
#' @param val_labels Validation labels
#' @param initial_epochs Epochs for training with frozen base
#' @param fine_tune_epochs Additional epochs for fine-tuning
#' @param initial_lr Initial learning rate
#' @param fine_tune_lr Fine-tuning learning rate (lower)
#' @param batch_size Batch size
#' @param callbacks Callbacks
#' @return Combined training history
fine_tune_model <- function(model,
                             train_images,
                             train_labels,
                             val_images = NULL,
                             val_labels = NULL,
                             initial_epochs = 10,
                             fine_tune_epochs = 20,
                             initial_lr = 0.001,
                             fine_tune_lr = 0.00001,
                             batch_size = 32,
                             callbacks = NULL) {
  
  library(keras3)
  
  cat("=== Phase 1: Training with frozen base ===\n")
  
  # Compile with initial learning rate
  model <- compile_model(model, learning_rate = initial_lr)
  
  # Train with frozen base
  history1 <- train_model(
    model,
    train_images, train_labels,
    val_images, val_labels,
    epochs = initial_epochs,
    batch_size = batch_size,
    callbacks = callbacks
  )
  
  cat("\n=== Phase 2: Fine-tuning ===\n")
  
  # Unfreeze base model layers
  for (layer in model$layers) {
    layer$trainable <- TRUE
  }
  
  # Recompile with lower learning rate
  model <- compile_model(model, learning_rate = fine_tune_lr)
  
  # Continue training
  history2 <- train_model(
    model,
    train_images, train_labels,
    val_images, val_labels,
    epochs = fine_tune_epochs,
    batch_size = batch_size,
    callbacks = callbacks
  )
  
  # Combine histories
  combined_history <- list(
    phase1 = history1,
    phase2 = history2
  )
  
  return(combined_history)
}

# ------------------------------------------------------------------------------
# Training with Data Augmentation
# ------------------------------------------------------------------------------

#' Train model with on-the-fly data augmentation
#' @param model Compiled model
#' @param train_images Training images
#' @param train_labels Training labels
#' @param val_images Validation images
#' @param val_labels Validation labels
#' @param epochs Number of epochs
#' @param batch_size Batch size
#' @param augmentation_params List of augmentation parameters
#' @param callbacks Callbacks
#' @return Training history
train_with_augmentation <- function(model,
                                     train_images,
                                     train_labels,
                                     val_images = NULL,
                                     val_labels = NULL,
                                     epochs = 50,
                                     batch_size = 32,
                                     augmentation_params = NULL,
                                     callbacks = NULL) {
  
  library(keras3)
  
  # Default augmentation parameters
  if (is.null(augmentation_params)) {
    augmentation_params <- list(
      rotation_range = 15,
      width_shift_range = 0.1,
      height_shift_range = 0.1,
      horizontal_flip = TRUE,
      zoom_range = 0.1,
      brightness_range = c(0.8, 1.2),
      fill_mode = "nearest"
    )
  }
  
  # Create data augmentation layer
  data_augmentation <- keras_model_sequential() %>%
    layer_random_flip("horizontal") %>%
    layer_random_rotation(factor = augmentation_params$rotation_range / 180) %>%
    layer_random_zoom(height_factor = augmentation_params$zoom_range) %>%
    layer_random_brightness(factor = 0.2)
  
  # Augment training data
  cat("Applying data augmentation...\n")
  
  # Convert labels to one-hot
  num_classes <- length(unique(c(train_labels, val_labels)))
  train_labels_onehot <- to_categorical(train_labels, num_classes)
  
  validation_data <- NULL
  if (!is.null(val_images) && !is.null(val_labels)) {
    val_labels_onehot <- to_categorical(val_labels, num_classes)
    validation_data <- list(val_images, val_labels_onehot)
  }
  
  # Train with augmentation
  history <- model %>% fit(
    x = train_images,
    y = train_labels_onehot,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = validation_data,
    callbacks = callbacks,
    verbose = 1
  )
  
  return(history)
}

# ------------------------------------------------------------------------------
# Complete Training Pipeline
# ------------------------------------------------------------------------------

#' Run complete training pipeline
#' @param training_dir Path to training data directory
#' @param architecture Model architecture to use
#' @param epochs Number of training epochs
#' @param batch_size Batch size
#' @param learning_rate Learning rate
#' @param validation_split Proportion for validation
#' @param use_augmentation Whether to use data augmentation
#' @param save_model_path Path to save trained model
#' @return List with model and training history
run_training_pipeline <- function(training_dir = NULL,
                                   architecture = "mobilenet",
                                   epochs = 50,
                                   batch_size = 32,
                                   learning_rate = 0.0001,
                                   validation_split = 0.2,
                                   use_augmentation = TRUE,
                                   save_model_path = NULL) {
  
  library(keras3)
  
  # Set defaults from config
  if (is.null(training_dir)) {
    training_dir <- get_config("training_dir")
  }
  if (is.null(save_model_path)) {
    save_model_path <- file.path(get_config("models_dir"), "psa_grading_model.keras")
  }
  
  cat("========================================\n")
  cat("PSA Card Grading Model Training\n")
  cat("========================================\n\n")
  
  # Step 1: Load data
  cat("Step 1: Loading training data...\n")
  dataset <- load_training_dataset(training_dir)
  print_dataset_summary(dataset, dataset$class_names)
  
  # Step 2: Split data
  cat("\nStep 2: Splitting into train/validation sets...\n")
  split <- split_dataset(
    dataset$images,
    dataset$labels,
    validation_split = validation_split
  )
  
  train_images <- split$train$images
  train_labels <- split$train$labels
  val_images <- split$validation$images
  val_labels <- split$validation$labels
  
  cat(sprintf("Training samples: %d\n", dim(train_images)[1]))
  cat(sprintf("Validation samples: %d\n", dim(val_images)[1]))
  
  # Step 3: Create model
  cat("\nStep 3: Creating model...\n")
  num_classes <- length(dataset$class_names)
  input_shape <- c(get_config("img_height"), get_config("img_width"), get_config("channels"))
  
  model <- create_psa_model(
    architecture = architecture,
    input_shape = input_shape,
    num_classes = num_classes,
    dropout_rate = get_config("dropout_rate")
  )
  
  model <- compile_model(model, learning_rate = learning_rate)
  print_model_summary(model)
  
  # Step 4: Create callbacks
  cat("\nStep 4: Setting up callbacks...\n")
  
  # Ensure model directory exists
  model_dir <- dirname(save_model_path)
  if (!dir.exists(model_dir)) {
    dir.create(model_dir, recursive = TRUE)
  }
  
  callbacks <- create_callbacks(
    model_path = save_model_path,
    log_dir = file.path(model_dir, "logs"),
    patience = get_config("patience")
  )
  
  # Step 5: Compute class weights
  cat("\nStep 5: Computing class weights...\n")
  class_weights <- compute_class_weights(train_labels)
  cat("Class weights:\n")
  print(round(class_weights, 3))
  
  # Step 6: Train model
  cat("\nStep 6: Training model...\n")
  history <- train_model(
    model,
    train_images, train_labels,
    val_images, val_labels,
    epochs = epochs,
    batch_size = batch_size,
    class_weights = class_weights,
    callbacks = unname(callbacks)
  )
  
  # Step 7: Evaluate final model
  cat("\nStep 7: Final evaluation...\n")
  num_classes <- length(dataset$class_names)
  val_labels_onehot <- to_categorical(val_labels, num_classes)
  
  final_eval <- model %>% evaluate(val_images, val_labels_onehot, verbose = 0)
  cat(sprintf("Final Validation Loss: %.4f\n", final_eval[[1]]))
  cat(sprintf("Final Validation Accuracy: %.4f\n", final_eval[[2]]))
  
  # Save class names for later use
  class_names_path <- gsub("\\.keras$", "_classes.rds", save_model_path)
  saveRDS(dataset$class_names, class_names_path)
  cat("\nClass names saved to:", class_names_path, "\n")
  
  cat("\n========================================\n")
  cat("Training Complete!\n")
  cat("Model saved to:", save_model_path, "\n")
  cat("========================================\n")
  
  return(list(
    model = model,
    history = history,
    class_names = dataset$class_names,
    evaluation = final_eval
  ))
}

# ------------------------------------------------------------------------------
# Training Visualization
# ------------------------------------------------------------------------------

#' Plot training history
#' @param history Training history object
#' @param save_path Optional path to save plot
plot_training_history <- function(history, save_path = NULL) {
  
  library(ggplot2)
  library(gridExtra)
  
  # Extract metrics from history
  if ("metrics" %in% names(history)) {
    metrics <- history$metrics
  } else {
    metrics <- history
  }
  
  epochs <- seq_len(length(metrics$loss))
  
  # Create data frame for plotting
  df_loss <- data.frame(
    epoch = c(epochs, epochs),
    value = c(metrics$loss, metrics$val_loss),
    type = c(rep("Training", length(epochs)), rep("Validation", length(epochs)))
  )
  
  df_acc <- data.frame(
    epoch = c(epochs, epochs),
    value = c(metrics$accuracy, metrics$val_accuracy),
    type = c(rep("Training", length(epochs)), rep("Validation", length(epochs)))
  )
  
  # Plot loss
  p1 <- ggplot(df_loss, aes(x = epoch, y = value, color = type)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(title = "Model Loss", x = "Epoch", y = "Loss") +
    theme_minimal() +
    theme(legend.title = element_blank())
  
  # Plot accuracy
  p2 <- ggplot(df_acc, aes(x = epoch, y = value, color = type)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(title = "Model Accuracy", x = "Epoch", y = "Accuracy") +
    theme_minimal() +
    theme(legend.title = element_blank()) +
    scale_y_continuous(labels = scales::percent_format())
  
  # Combine plots
  combined <- grid.arrange(p1, p2, ncol = 2)
  
  # Save if path provided
  if (!is.null(save_path)) {
    ggsave(save_path, combined, width = 12, height = 5)
    cat("Training history plot saved to:", save_path, "\n")
  }
  
  return(combined)
}
