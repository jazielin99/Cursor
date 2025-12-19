# ==============================================================================
# PSA Card Grading Model - Model Definition
# ==============================================================================
# Deep learning model architectures for PSA card grading

# Source configuration
source(file.path(getwd(), "R", "config.R"))

# ------------------------------------------------------------------------------
# Model Architecture Functions
# ------------------------------------------------------------------------------

#' Create a custom CNN model for PSA grading
#' @param input_shape Input image shape (height, width, channels)
#' @param num_classes Number of output classes
#' @param dropout_rate Dropout rate for regularization
#' @return Keras model
create_custom_cnn <- function(input_shape = c(224, 224, 3),
                               num_classes = 12,
                               dropout_rate = 0.5) {
  
  library(keras3)
  
  model <- keras_model_sequential(input_shape = input_shape) %>%
    
    # Block 1
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    
    # Block 2
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    
    # Block 3
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    
    # Block 4
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    
    # Flatten and Dense layers
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  return(model)
}

#' Create a model using transfer learning with MobileNetV2
#' @param input_shape Input image shape (height, width, channels)
#' @param num_classes Number of output classes
#' @param trainable_layers Number of top layers to make trainable (NULL = freeze all)
#' @param dropout_rate Dropout rate for regularization
#' @return Keras model
create_mobilenet_model <- function(input_shape = c(224, 224, 3),
                                    num_classes = 12,
                                    trainable_layers = NULL,
                                    dropout_rate = 0.5) {
  
  library(keras3)
  
  # Load pre-trained MobileNetV2 (without top classification layer)
  base_model <- application_mobilenet_v2(
    input_shape = input_shape,
    include_top = FALSE,
    weights = "imagenet",
    pooling = "avg"
  )
  
  # Freeze base model layers
  base_model$trainable <- FALSE
  
  # If trainable_layers specified, unfreeze top N layers
  if (!is.null(trainable_layers) && trainable_layers > 0) {
    n_layers <- length(base_model$layers)
    for (i in seq(max(1, n_layers - trainable_layers + 1), n_layers)) {
      base_model$layers[[i]]$trainable <- TRUE
    }
  }
  
  # Create the full model
  inputs <- layer_input(shape = input_shape)
  
  # Preprocess for MobileNet
  x <- inputs %>%
    layer_rescaling(scale = 1/127.5, offset = -1)  # Scale to [-1, 1]
  
  # Base model features
  x <- base_model(x, training = FALSE)
  
  # Classification head
  x <- x %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = dropout_rate / 2)
  
  outputs <- x %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  model <- keras_model(inputs, outputs)
  
  return(model)
}

#' Create a model using transfer learning with ResNet50
#' @param input_shape Input image shape (height, width, channels)
#' @param num_classes Number of output classes
#' @param dropout_rate Dropout rate for regularization
#' @return Keras model
create_resnet_model <- function(input_shape = c(224, 224, 3),
                                 num_classes = 12,
                                 dropout_rate = 0.5) {
  
  library(keras3)
  
  # Load pre-trained ResNet50
  base_model <- application_resnet50(
    input_shape = input_shape,
    include_top = FALSE,
    weights = "imagenet",
    pooling = "avg"
  )
  
  # Freeze base model layers
  base_model$trainable <- FALSE
  
  # Create the full model
  inputs <- layer_input(shape = input_shape)
  
  # Preprocess for ResNet
  x <- inputs %>%
    layer_rescaling(scale = 255)  # ResNet expects [0, 255]
  
  # Apply ResNet preprocessing
  x <- tf$keras$applications$resnet50$preprocess_input(x)
  
  # Base model features
  x <- base_model(x, training = FALSE)
  
  # Classification head
  x <- x %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dropout(rate = dropout_rate / 2)
  
  outputs <- x %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  model <- keras_model(inputs, outputs)
  
  return(model)
}

#' Create EfficientNetB0-based model
#' @param input_shape Input image shape (height, width, channels)
#' @param num_classes Number of output classes
#' @param dropout_rate Dropout rate for regularization
#' @return Keras model
create_efficientnet_model <- function(input_shape = c(224, 224, 3),
                                       num_classes = 12,
                                       dropout_rate = 0.5) {
  
  library(keras3)
  
  # Load pre-trained EfficientNetB0
  base_model <- application_efficientnet_b0(
    input_shape = input_shape,
    include_top = FALSE,
    weights = "imagenet",
    pooling = "avg"
  )
  
  # Freeze base model layers
  base_model$trainable <- FALSE
  
  # Create the full model
  inputs <- layer_input(shape = input_shape)
  
  # Base model features (EfficientNet expects [0, 255])
  x <- inputs %>%
    layer_rescaling(scale = 255)
  
  x <- base_model(x, training = FALSE)
  
  # Classification head
  x <- x %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = dropout_rate / 2)
  
  outputs <- x %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  model <- keras_model(inputs, outputs)
  
  return(model)
}

# ------------------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------------------

#' Create a PSA grading model
#' @param architecture Model architecture ("custom", "mobilenet", "resnet", "efficientnet")
#' @param input_shape Input image shape
#' @param num_classes Number of output classes
#' @param dropout_rate Dropout rate
#' @return Compiled Keras model
create_psa_model <- function(architecture = "mobilenet",
                              input_shape = c(224, 224, 3),
                              num_classes = 12,
                              dropout_rate = 0.5) {
  
  cat("Creating", architecture, "model...\n")
  
  model <- switch(
    architecture,
    "custom" = create_custom_cnn(input_shape, num_classes, dropout_rate),
    "mobilenet" = create_mobilenet_model(input_shape, num_classes, dropout_rate = dropout_rate),
    "resnet" = create_resnet_model(input_shape, num_classes, dropout_rate),
    "efficientnet" = create_efficientnet_model(input_shape, num_classes, dropout_rate),
    stop(paste("Unknown architecture:", architecture))
  )
  
  cat("Model created successfully.\n")
  
  return(model)
}

#' Compile model with optimizer and loss function
#' @param model Keras model
#' @param learning_rate Learning rate
#' @param metrics Metrics to track
#' @return Compiled model
compile_model <- function(model, 
                          learning_rate = 0.0001,
                          metrics = c("accuracy")) {
  
  library(keras3)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "categorical_crossentropy",
    metrics = metrics
  )
  
  cat("Model compiled with:\n")
  cat("  - Optimizer: Adam (lr =", learning_rate, ")\n")
  cat("  - Loss: Categorical Crossentropy\n")
  cat("  - Metrics:", paste(metrics, collapse = ", "), "\n")
  
  return(model)
}

# ------------------------------------------------------------------------------
# Model Utilities
# ------------------------------------------------------------------------------

#' Print model summary
#' @param model Keras model
print_model_summary <- function(model) {
  cat("\n=== Model Summary ===\n")
  summary(model)
  
  # Count parameters
  trainable <- sum(sapply(model$trainable_weights, function(w) prod(dim(w))))
  non_trainable <- sum(sapply(model$non_trainable_weights, function(w) prod(dim(w))))
  total <- trainable + non_trainable
  
  cat("\n")
  cat(sprintf("Total parameters: %s\n", format(total, big.mark = ",")))
  cat(sprintf("Trainable parameters: %s\n", format(trainable, big.mark = ",")))
  cat(sprintf("Non-trainable parameters: %s\n", format(non_trainable, big.mark = ",")))
}

#' Save model to file
#' @param model Keras model
#' @param file_path Path to save model (without extension)
#' @param save_weights_only Whether to save only weights
save_model <- function(model, file_path, save_weights_only = FALSE) {
  
  if (save_weights_only) {
    weights_path <- paste0(file_path, ".weights.h5")
    save_model_weights(model, weights_path)
    cat("Model weights saved to:", weights_path, "\n")
  } else {
    # Save full model
    keras_path <- paste0(file_path, ".keras")
    save_model(model, keras_path)
    cat("Model saved to:", keras_path, "\n")
  }
}

#' Load model from file
#' @param file_path Path to model file
#' @return Loaded Keras model
load_psa_model <- function(file_path) {
  
  library(keras3)
  
  if (!file.exists(file_path)) {
    stop(paste("Model file not found:", file_path))
  }
  
  cat("Loading model from:", file_path, "\n")
  model <- load_model(file_path)
  cat("Model loaded successfully.\n")
  
  return(model)
}

# ------------------------------------------------------------------------------
# Alternative: Traditional ML Models (fallback if deep learning not available)
# ------------------------------------------------------------------------------

#' Extract features from images using color histograms and basic statistics
#' @param images Images array (n x height x width x channels)
#' @return Feature matrix
extract_basic_features <- function(images) {
  
  n <- dim(images)[1]
  cat("Extracting basic features from", n, "images...\n")
  
  features <- matrix(0, nrow = n, ncol = 0)
  
  pb <- txtProgressBar(min = 0, max = n, style = 3)
  
  feature_list <- list()
  
  for (i in seq_len(n)) {
    img <- images[i, , , ]
    
    # Color channel statistics
    r <- img[, , 1]
    g <- img[, , 2]
    b <- img[, , 3]
    
    # Mean and std for each channel
    channel_stats <- c(
      mean(r), sd(r), mean(g), sd(g), mean(b), sd(b)
    )
    
    # Color histogram (16 bins per channel)
    r_hist <- hist(r, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts
    g_hist <- hist(g, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts
    b_hist <- hist(b, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts
    
    # Edge detection (simple gradient)
    grad_x <- diff(apply(img, c(1, 3), mean))
    grad_y <- diff(apply(img, c(2, 3), mean))
    edge_features <- c(
      mean(abs(grad_x)), sd(abs(grad_x)),
      mean(abs(grad_y)), sd(abs(grad_y))
    )
    
    # Brightness and contrast
    gray <- 0.299 * r + 0.587 * g + 0.114 * b
    brightness_contrast <- c(mean(gray), sd(gray))
    
    # Corner analysis (sample corner regions)
    h <- dim(img)[1]
    w <- dim(img)[2]
    corner_size <- floor(min(h, w) / 8)
    
    corners <- list(
      img[1:corner_size, 1:corner_size, ],  # Top-left
      img[1:corner_size, (w-corner_size+1):w, ],  # Top-right
      img[(h-corner_size+1):h, 1:corner_size, ],  # Bottom-left
      img[(h-corner_size+1):h, (w-corner_size+1):w, ]  # Bottom-right
    )
    
    corner_stats <- unlist(lapply(corners, function(c) c(mean(c), sd(c))))
    
    # Centering analysis (compare left/right, top/bottom)
    left_half <- img[, 1:(w/2), ]
    right_half <- img[, (w/2+1):w, ]
    top_half <- img[1:(h/2), , ]
    bottom_half <- img[(h/2+1):h, , ]
    
    centering_features <- c(
      abs(mean(left_half) - mean(right_half)),
      abs(mean(top_half) - mean(bottom_half))
    )
    
    # Combine all features
    all_features <- c(
      channel_stats,
      r_hist / sum(r_hist),  # Normalize histograms
      g_hist / sum(g_hist),
      b_hist / sum(b_hist),
      edge_features,
      brightness_contrast,
      corner_stats,
      centering_features
    )
    
    feature_list[[i]] <- all_features
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  # Combine into matrix
  features <- do.call(rbind, feature_list)
  
  cat("Extracted", ncol(features), "features per image.\n")
  
  return(features)
}

#' Train a Random Forest model for PSA grading
#' @param features Feature matrix
#' @param labels Labels vector
#' @param n_trees Number of trees
#' @return Trained randomForest model
train_random_forest <- function(features, labels, n_trees = 500) {
  
  library(randomForest)
  
  cat("Training Random Forest model with", n_trees, "trees...\n")
  
  # Convert labels to factor
  labels_factor <- factor(labels)
  
  model <- randomForest(
    x = features,
    y = labels_factor,
    ntree = n_trees,
    importance = TRUE,
    do.trace = 100
  )
  
  cat("\nRandom Forest training complete.\n")
  print(model)
  
  return(model)
}

#' Train an XGBoost model for PSA grading
#' @param features Feature matrix
#' @param labels Labels vector
#' @param num_classes Number of classes
#' @param n_rounds Number of boosting rounds
#' @return Trained xgboost model
train_xgboost <- function(features, labels, num_classes = 12, n_rounds = 100) {
  
  library(xgboost)
  
  cat("Training XGBoost model...\n")
  
  # Create DMatrix
  dtrain <- xgb.DMatrix(data = as.matrix(features), label = labels)
  
  # Parameters for multi-class classification
  params <- list(
    objective = "multi:softprob",
    num_class = num_classes,
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = n_rounds,
    verbose = 1,
    print_every_n = 20
  )
  
  cat("\nXGBoost training complete.\n")
  
  return(model)
}
