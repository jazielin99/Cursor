# ==============================================================================
# PSA Card Grading Model - Traditional ML Fallback
# ==============================================================================
# This script provides alternative machine learning models when TensorFlow/Keras
# is not available. Uses Random Forest and XGBoost with handcrafted features.

# Set working directory
setwd("/workspace")

# Load required packages
required_packages <- c("randomForest", "xgboost", "caret", "e1071", "magick")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Source dependencies
source("R/config.R")
source("R/02_data_preparation.R")

# ------------------------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------------------------

#' Extract comprehensive features from a card image
#' @param img Image array (height x width x channels)
#' @return Named vector of features
extract_card_features <- function(img) {
  
  # Ensure image is in [0, 1] range
  if (max(img) > 1) {
    img <- img / 255
  }
  
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  # Extract color channels
  r <- img[, , 1]
  g <- img[, , 2]
  b <- img[, , 3]
  
  # Convert to grayscale
  gray <- 0.299 * r + 0.587 * g + 0.114 * b
  
  features <- c()
  
  # ==========================================================================
  # 1. COLOR STATISTICS
  # ==========================================================================
  
  # Mean and standard deviation per channel
  features <- c(features,
    r_mean = mean(r), r_sd = sd(r),
    g_mean = mean(g), g_sd = sd(g),
    b_mean = mean(b), b_sd = sd(b),
    gray_mean = mean(gray), gray_sd = sd(gray)
  )
  
  # Color histograms (16 bins each, normalized)
  r_hist <- hist(r, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts / length(r)
  g_hist <- hist(g, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts / length(g)
  b_hist <- hist(b, breaks = seq(0, 1, length.out = 17), plot = FALSE)$counts / length(b)
  
  names(r_hist) <- paste0("r_hist_", 1:16)
  names(g_hist) <- paste0("g_hist_", 1:16)
  names(b_hist) <- paste0("b_hist_", 1:16)
  
  features <- c(features, r_hist, g_hist, b_hist)
  
  # ==========================================================================
  # 2. CORNER ANALYSIS (important for PSA grading)
  # ==========================================================================
  
  corner_size <- floor(min(h, w) / 10)
  
  # Extract corners
  tl <- gray[1:corner_size, 1:corner_size]  # Top-left
  tr <- gray[1:corner_size, (w-corner_size+1):w]  # Top-right
  bl <- gray[(h-corner_size+1):h, 1:corner_size]  # Bottom-left
  br <- gray[(h-corner_size+1):h, (w-corner_size+1):w]  # Bottom-right
  
  # Corner statistics
  features <- c(features,
    corner_tl_mean = mean(tl), corner_tl_sd = sd(tl),
    corner_tr_mean = mean(tr), corner_tr_sd = sd(tr),
    corner_bl_mean = mean(bl), corner_bl_sd = sd(bl),
    corner_br_mean = mean(br), corner_br_sd = sd(br)
  )
  
  # Corner sharpness (gradient magnitude)
  corner_grad <- function(corner) {
    gx <- diff(apply(corner, 1, mean))
    gy <- diff(apply(corner, 2, mean))
    return(sqrt(mean(gx^2) + mean(gy^2)))
  }
  
  features <- c(features,
    corner_tl_sharp = corner_grad(tl),
    corner_tr_sharp = corner_grad(tr),
    corner_bl_sharp = corner_grad(bl),
    corner_br_sharp = corner_grad(br)
  )
  
  # ==========================================================================
  # 3. EDGE ANALYSIS
  # ==========================================================================
  
  edge_size <- floor(min(h, w) / 20)
  
  # Extract edges
  top_edge <- gray[1:edge_size, ]
  bottom_edge <- gray[(h-edge_size+1):h, ]
  left_edge <- gray[, 1:edge_size]
  right_edge <- gray[, (w-edge_size+1):w]
  
  features <- c(features,
    edge_top_mean = mean(top_edge), edge_top_sd = sd(top_edge),
    edge_bottom_mean = mean(bottom_edge), edge_bottom_sd = sd(bottom_edge),
    edge_left_mean = mean(left_edge), edge_left_sd = sd(left_edge),
    edge_right_mean = mean(right_edge), edge_right_sd = sd(right_edge)
  )
  
  # ==========================================================================
  # 4. CENTERING ANALYSIS
  # ==========================================================================
  
  # Compare left vs right halves
  left_half <- gray[, 1:(w/2)]
  right_half <- gray[, (w/2+1):w]
  
  # Compare top vs bottom halves
  top_half <- gray[1:(h/2), ]
  bottom_half <- gray[(h/2+1):h, ]
  
  features <- c(features,
    center_lr_diff = abs(mean(left_half) - mean(right_half)),
    center_tb_diff = abs(mean(top_half) - mean(bottom_half)),
    center_lr_ratio = mean(left_half) / (mean(right_half) + 0.001),
    center_tb_ratio = mean(top_half) / (mean(bottom_half) + 0.001)
  )
  
  # ==========================================================================
  # 5. SURFACE ANALYSIS
  # ==========================================================================
  
  # Gradient magnitude (surface uniformity)
  gx <- matrix(0, h, w)
  gy <- matrix(0, h, w)
  
  gx[, 2:(w-1)] <- gray[, 3:w] - gray[, 1:(w-2)]
  gy[2:(h-1), ] <- gray[3:h, ] - gray[1:(h-2), ]
  
  gradient_mag <- sqrt(gx^2 + gy^2)
  
  features <- c(features,
    surface_grad_mean = mean(gradient_mag),
    surface_grad_sd = sd(gradient_mag),
    surface_grad_max = max(gradient_mag)
  )
  
  # Texture analysis (variance in local windows)
  window_size <- 16
  local_vars <- c()
  
  for (i in seq(1, h - window_size, by = window_size)) {
    for (j in seq(1, w - window_size, by = window_size)) {
      window <- gray[i:(i+window_size-1), j:(j+window_size-1)]
      local_vars <- c(local_vars, var(as.vector(window)))
    }
  }
  
  features <- c(features,
    texture_var_mean = mean(local_vars),
    texture_var_sd = sd(local_vars),
    texture_var_max = max(local_vars)
  )
  
  # ==========================================================================
  # 6. GLOSS/BRIGHTNESS ANALYSIS
  # ==========================================================================
  
  # Brightness distribution
  features <- c(features,
    brightness_mean = mean(gray),
    brightness_median = median(gray),
    brightness_q25 = quantile(gray, 0.25),
    brightness_q75 = quantile(gray, 0.75),
    brightness_range = diff(range(gray))
  )
  
  # Contrast
  features <- c(features,
    contrast = sd(gray),
    contrast_rms = sqrt(mean((gray - mean(gray))^2))
  )
  
  # ==========================================================================
  # 7. DEFECT DETECTION
  # ==========================================================================
  
  # Look for potential scratches (high gradient lines)
  horiz_grads <- apply(abs(gx), 1, mean)
  vert_grads <- apply(abs(gy), 2, mean)
  
  features <- c(features,
    scratch_horiz_max = max(horiz_grads),
    scratch_vert_max = max(vert_grads),
    scratch_horiz_mean = mean(horiz_grads),
    scratch_vert_mean = mean(vert_grads)
  )
  
  # Look for potential creases (continuous lines)
  high_grad_thresh <- quantile(gradient_mag, 0.95)
  high_grad_pct <- mean(gradient_mag > high_grad_thresh)
  
  features <- c(features,
    high_grad_pct = high_grad_pct
  )
  
  # ==========================================================================
  # 8. COLOR UNIFORMITY (for staining detection)
  # ==========================================================================
  
  # Hue calculation (simplified)
  max_c <- pmax(r, g, b)
  min_c <- pmin(r, g, b)
  delta <- max_c - min_c
  
  # Saturation
  saturation <- ifelse(max_c > 0, delta / max_c, 0)
  
  features <- c(features,
    saturation_mean = mean(saturation),
    saturation_sd = sd(saturation),
    saturation_max = max(saturation)
  )
  
  # Color uniformity (lower is more uniform)
  color_diff <- abs(r - g) + abs(g - b) + abs(r - b)
  
  features <- c(features,
    color_uniformity = mean(color_diff),
    color_uniformity_sd = sd(color_diff)
  )
  
  return(features)
}

#' Extract features from all images in a dataset
#' @param images Images array (n x height x width x channels)
#' @return Feature matrix
extract_all_features <- function(images) {
  
  n <- dim(images)[1]
  cat("Extracting features from", n, "images...\n")
  
  features_list <- list()
  pb <- txtProgressBar(min = 0, max = n, style = 3)
  
  for (i in seq_len(n)) {
    features_list[[i]] <- extract_card_features(images[i, , , ])
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  # Combine into matrix
  features <- do.call(rbind, features_list)
  
  cat("Extracted", ncol(features), "features per image.\n")
  
  return(features)
}

# ------------------------------------------------------------------------------
# Random Forest Model
# ------------------------------------------------------------------------------

#' Train a Random Forest classifier
#' @param features Feature matrix
#' @param labels Labels vector
#' @param n_trees Number of trees
#' @param mtry Number of variables to try at each split
#' @return Trained model and important metrics
train_rf_model <- function(features, labels, n_trees = 500, mtry = NULL) {
  
  library(randomForest)
  
  # Convert labels to factor
  labels_factor <- factor(labels)
  
  # Set default mtry
  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(features)))
  }
  
  cat("\n=== Training Random Forest ===\n")
  cat("Number of trees:", n_trees, "\n")
  cat("Variables per split:", mtry, "\n")
  cat("Training samples:", nrow(features), "\n")
  cat("Features:", ncol(features), "\n\n")
  
  # Train model
  model <- randomForest(
    x = features,
    y = labels_factor,
    ntree = n_trees,
    mtry = mtry,
    importance = TRUE,
    do.trace = 100
  )
  
  cat("\nTraining complete!\n")
  
  # Get feature importance
  importance <- importance(model)
  importance_df <- data.frame(
    feature = rownames(importance),
    mean_decrease_accuracy = importance[, "MeanDecreaseAccuracy"],
    mean_decrease_gini = importance[, "MeanDecreaseGini"]
  )
  importance_df <- importance_df[order(-importance_df$mean_decrease_accuracy), ]
  
  cat("\nTop 10 most important features:\n")
  print(head(importance_df, 10))
  
  return(list(
    model = model,
    importance = importance_df
  ))
}

# ------------------------------------------------------------------------------
# XGBoost Model
# ------------------------------------------------------------------------------

#' Train an XGBoost classifier
#' @param features Feature matrix
#' @param labels Labels vector
#' @param num_classes Number of classes
#' @param n_rounds Number of boosting rounds
#' @return Trained model
train_xgb_model <- function(features, labels, num_classes = NULL, n_rounds = 200) {
  
  library(xgboost)
  
  if (is.null(num_classes)) {
    num_classes <- length(unique(labels))
  }
  
  cat("\n=== Training XGBoost ===\n")
  cat("Boosting rounds:", n_rounds, "\n")
  cat("Number of classes:", num_classes, "\n\n")
  
  # Create DMatrix
  dtrain <- xgb.DMatrix(data = as.matrix(features), label = labels)
  
  # Parameters
  params <- list(
    objective = "multi:softprob",
    num_class = num_classes,
    max_depth = 8,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    eval_metric = "mlogloss"
  )
  
  # Train with cross-validation first
  cat("Running 5-fold cross-validation...\n")
  cv_result <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = n_rounds,
    nfold = 5,
    early_stopping_rounds = 20,
    print_every_n = 50,
    verbose = 1
  )
  
  best_round <- cv_result$best_iteration
  cat("\nBest iteration:", best_round, "\n")
  
  # Train final model
  cat("\nTraining final model...\n")
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_round,
    verbose = 1,
    print_every_n = 50
  )
  
  # Feature importance
  importance <- xgb.importance(model = model)
  
  cat("\nTop 10 most important features:\n")
  print(head(importance, 10))
  
  return(list(
    model = model,
    best_round = best_round,
    importance = importance
  ))
}

# ------------------------------------------------------------------------------
# Prediction Functions
# ------------------------------------------------------------------------------

#' Predict using trained Random Forest model
#' @param model Trained RF model
#' @param features Feature matrix
#' @return Predictions
predict_rf <- function(model, features) {
  predictions <- predict(model, features)
  probabilities <- predict(model, features, type = "prob")
  
  return(list(
    predictions = predictions,
    probabilities = probabilities
  ))
}

#' Predict using trained XGBoost model
#' @param model Trained XGB model
#' @param features Feature matrix
#' @param class_names Vector of class names
#' @return Predictions
predict_xgb <- function(model, features, class_names = NULL) {
  
  library(xgboost)
  
  dtest <- xgb.DMatrix(data = as.matrix(features))
  probabilities <- predict(model, dtest, reshape = TRUE)
  predictions <- apply(probabilities, 1, which.max) - 1
  
  if (!is.null(class_names)) {
    predictions <- class_names[predictions + 1]
    colnames(probabilities) <- class_names
  }
  
  return(list(
    predictions = predictions,
    probabilities = probabilities
  ))
}

# ------------------------------------------------------------------------------
# Complete Pipeline
# ------------------------------------------------------------------------------

#' Run complete traditional ML pipeline
#' @param method Model method ("rf" or "xgb")
#' @return Trained model and evaluation results
run_traditional_ml_pipeline <- function(method = "rf") {
  
  cat("========================================\n")
  cat("Traditional ML Pipeline for PSA Grading\n")
  cat("========================================\n\n")
  
  # Load data
  cat("Step 1: Loading data...\n")
  dataset <- load_training_dataset()
  print_dataset_summary(dataset, dataset$class_names)
  
  # Split data
  cat("\nStep 2: Splitting data...\n")
  split <- split_dataset(dataset$images, dataset$labels, validation_split = 0.2)
  
  train_images <- split$train$images
  train_labels <- split$train$labels
  val_images <- split$validation$images
  val_labels <- split$validation$labels
  
  # Extract features
  cat("\nStep 3: Extracting features...\n")
  train_features <- extract_all_features(train_images)
  val_features <- extract_all_features(val_images)
  
  # Handle any NA values
  train_features[is.na(train_features)] <- 0
  val_features[is.na(val_features)] <- 0
  
  # Train model
  cat("\nStep 4: Training model...\n")
  
  if (method == "rf") {
    result <- train_rf_model(train_features, train_labels)
    model <- result$model
    
    # Predict
    predictions <- predict_rf(model, val_features)
    pred_labels <- as.numeric(as.character(predictions$predictions))
    
  } else if (method == "xgb") {
    num_classes <- length(dataset$class_names)
    result <- train_xgb_model(train_features, train_labels, num_classes)
    model <- result$model
    
    # Predict
    predictions <- predict_xgb(model, val_features, dataset$class_names)
    pred_labels <- match(predictions$predictions, dataset$class_names) - 1
  }
  
  # Evaluate
  cat("\nStep 5: Evaluation...\n")
  accuracy <- mean(pred_labels == val_labels)
  
  cat("\nValidation Accuracy:", round(accuracy * 100, 2), "%\n")
  
  # Confusion matrix
  confusion <- table(
    Actual = factor(val_labels, levels = 0:(length(dataset$class_names)-1)),
    Predicted = factor(pred_labels, levels = 0:(length(dataset$class_names)-1))
  )
  
  cat("\nConfusion Matrix:\n")
  print(confusion)
  
  # Save model
  model_path <- file.path(get_config("models_dir"), paste0("psa_", method, "_model.rds"))
  saveRDS(list(model = model, class_names = dataset$class_names), model_path)
  cat("\nModel saved to:", model_path, "\n")
  
  cat("\n========================================\n")
  cat("Pipeline Complete!\n")
  cat("========================================\n")
  
  return(list(
    model = model,
    class_names = dataset$class_names,
    accuracy = accuracy,
    confusion = confusion
  ))
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if (interactive()) {
  cat("\n=== Traditional ML Fallback ===\n")
  cat("\nUse this when TensorFlow/Keras is not available.\n")
  cat("\nRun: run_traditional_ml_pipeline('rf')  # Random Forest\n")
  cat("     run_traditional_ml_pipeline('xgb') # XGBoost\n")
  cat("\n")
}
