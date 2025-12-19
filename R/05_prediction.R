# ==============================================================================
# PSA Card Grading Model - Prediction
# ==============================================================================
# Functions for making predictions with trained PSA grading models

# Source dependencies
source(file.path(getwd(), "R", "config.R"))
source(file.path(getwd(), "R", "02_data_preparation.R"))
source(file.path(getwd(), "R", "grading_standards.R"))

# ------------------------------------------------------------------------------
# Prediction Functions
# ------------------------------------------------------------------------------

#' Load a trained PSA grading model
#' @param model_path Path to saved model (.keras or .h5)
#' @param classes_path Optional path to saved class names (.rds)
#' @return List with model and class_names
load_grading_model <- function(model_path, classes_path = NULL) {
  
  library(keras3)
  
  # Load model
  if (!file.exists(model_path)) {
    stop(paste("Model file not found:", model_path))
  }
  
  cat("Loading model from:", model_path, "\n")
  model <- load_model(model_path)
  
  # Load class names if available
  if (is.null(classes_path)) {
    classes_path <- gsub("\\.keras$|\\.h5$", "_classes.rds", model_path)
  }
  
  class_names <- get_config("grade_classes")  # Default
  if (file.exists(classes_path)) {
    class_names <- readRDS(classes_path)
    cat("Loaded class names from:", classes_path, "\n")
  }
  
  return(list(
    model = model,
    class_names = class_names
  ))
}

#' Predict PSA grade for a single image
#' @param model Loaded Keras model
#' @param image_path Path to image file
#' @param class_names Vector of class names
#' @param top_n Number of top predictions to return
#' @return Data frame with predictions
predict_single_image <- function(model, 
                                  image_path, 
                                  class_names = NULL,
                                  top_n = 3) {
  
  library(keras3)
  
  # Load and preprocess image
  target_size <- c(get_config("img_width"), get_config("img_height"))
  img <- load_and_preprocess_image(image_path, target_size)
  
  if (is.null(img)) {
    stop(paste("Failed to load image:", image_path))
  }
  
  # Add batch dimension
  img_batch <- array(img, dim = c(1, dim(img)))
  
  # Make prediction
  predictions <- model %>% predict(img_batch, verbose = 0)
  
  # Get class names
  if (is.null(class_names)) {
    class_names <- get_config("grade_classes")
  }
  
  # Create results data frame
  results <- data.frame(
    class = class_names,
    probability = as.vector(predictions),
    stringsAsFactors = FALSE
  )
  
  # Sort by probability
  results <- results[order(-results$probability), ]
  rownames(results) <- NULL
  
  # Add rank
  results$rank <- seq_len(nrow(results))
  
  # Return top N
  top_results <- head(results, top_n)
  
  return(list(
    top_predictions = top_results,
    all_predictions = results,
    predicted_class = results$class[1],
    confidence = results$probability[1]
  ))
}

#' Predict PSA grades for multiple images
#' @param model Loaded Keras model
#' @param image_paths Vector of image paths
#' @param class_names Vector of class names
#' @param batch_size Batch size for predictions
#' @return Data frame with predictions for all images
predict_batch <- function(model, 
                          image_paths, 
                          class_names = NULL,
                          batch_size = 32) {
  
  library(keras3)
  
  target_size <- c(get_config("img_width"), get_config("img_height"))
  
  if (is.null(class_names)) {
    class_names <- get_config("grade_classes")
  }
  
  # Load all images
  cat("Loading", length(image_paths), "images...\n")
  images <- list()
  valid_paths <- character(0)
  
  pb <- txtProgressBar(min = 0, max = length(image_paths), style = 3)
  for (i in seq_along(image_paths)) {
    img <- load_and_preprocess_image(image_paths[i], target_size)
    if (!is.null(img)) {
      images[[length(images) + 1]] <- img
      valid_paths <- c(valid_paths, image_paths[i])
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  if (length(images) == 0) {
    stop("No valid images to predict!")
  }
  
  # Stack into array
  images_array <- array(0, dim = c(length(images), target_size[2], target_size[1], 3))
  for (i in seq_along(images)) {
    images_array[i, , , ] <- images[[i]]
  }
  
  # Make predictions
  cat("Making predictions...\n")
  predictions <- model %>% predict(images_array, batch_size = batch_size, verbose = 1)
  
  # Get predicted classes and confidences
  predicted_indices <- apply(predictions, 1, which.max)
  confidences <- apply(predictions, 1, max)
  
  # Create results data frame
  results <- data.frame(
    image = valid_paths,
    predicted_grade = class_names[predicted_indices],
    confidence = confidences,
    stringsAsFactors = FALSE
  )
  
  # Add all class probabilities
  for (i in seq_along(class_names)) {
    results[[class_names[i]]] <- predictions[, i]
  }
  
  return(results)
}

#' Predict from a directory of images
#' @param model Loaded Keras model
#' @param image_dir Directory containing images
#' @param class_names Vector of class names
#' @param extensions Valid image extensions
#' @return Data frame with predictions
predict_directory <- function(model, 
                               image_dir, 
                               class_names = NULL,
                               extensions = c("jpg", "jpeg", "png", "gif", "bmp")) {
  
  if (!dir.exists(image_dir)) {
    stop(paste("Directory not found:", image_dir))
  }
  
  # Get all image files
  pattern <- paste0("\\.(", paste(extensions, collapse = "|"), ")$")
  image_paths <- list.files(image_dir, pattern = pattern, 
                            full.names = TRUE, ignore.case = TRUE)
  
  if (length(image_paths) == 0) {
    stop(paste("No images found in:", image_dir))
  }
  
  cat("Found", length(image_paths), "images\n")
  
  return(predict_batch(model, image_paths, class_names))
}

# ------------------------------------------------------------------------------
# Detailed Prediction Analysis
# ------------------------------------------------------------------------------

#' Get detailed prediction with PSA grade information
#' @param model Loaded Keras model
#' @param image_path Path to image file
#' @param class_names Vector of class names
#' @return List with detailed prediction information
get_detailed_prediction <- function(model, image_path, class_names = NULL) {
  
  # Get basic prediction
  pred <- predict_single_image(model, image_path, class_names)
  
  # Get grade information
  predicted_grade <- pred$predicted_class
  
  # Parse grade number
  grade_num <- as.numeric(gsub("PSA_", "", predicted_grade))
  
  # Get grade standards if not NO_GRADE
  grade_info <- NULL
  if (!is.na(grade_num)) {
    tryCatch({
      grade_info <- get_grade_info(grade_num)
    }, error = function(e) {
      grade_info <- NULL
    })
  }
  
  # Create detailed result
  result <- list(
    image = image_path,
    predicted_grade = predicted_grade,
    confidence = pred$confidence,
    confidence_pct = paste0(round(pred$confidence * 100, 1), "%"),
    top_3_predictions = pred$top_predictions,
    all_predictions = pred$all_predictions
  )
  
  if (!is.null(grade_info)) {
    result$grade_name <- grade_info$full_name
    result$grade_abbreviation <- grade_info$name
    result$grade_description <- grade_info$description
    result$centering_requirements <- grade_info$centering
    result$defects_allowed <- grade_info$defects_allowed
  }
  
  return(result)
}

#' Print detailed prediction
#' @param detailed_pred Result from get_detailed_prediction
print_detailed_prediction <- function(detailed_pred) {
  
  cat("\n")
  cat("=" , rep("=", 50), "\n", sep = "")
  cat("PSA Card Grading Prediction\n")
  cat("=", rep("=", 50), "\n", sep = "")
  
  cat("\nImage:", detailed_pred$image, "\n")
  cat("\nPredicted Grade:", detailed_pred$predicted_grade, "\n")
  cat("Confidence:", detailed_pred$confidence_pct, "\n")
  
  if (!is.null(detailed_pred$grade_name)) {
    cat("\nGrade Name:", detailed_pred$grade_name, 
        "(", detailed_pred$grade_abbreviation, ")\n")
    cat("\nDescription:\n")
    cat(strwrap(detailed_pred$grade_description, width = 60), sep = "\n")
    
    cat("\nCentering Requirements:\n")
    cat("  Front:", detailed_pred$centering_requirements$front, "\n")
    cat("  Back:", detailed_pred$centering_requirements$back, "\n")
    
    cat("\nAllowed Defects:\n")
    for (defect in detailed_pred$defects_allowed[1:min(5, length(detailed_pred$defects_allowed))]) {
      cat("  -", defect, "\n")
    }
    if (length(detailed_pred$defects_allowed) > 5) {
      cat("  ... and", length(detailed_pred$defects_allowed) - 5, "more\n")
    }
  }
  
  cat("\nTop 3 Predictions:\n")
  for (i in seq_len(nrow(detailed_pred$top_3_predictions))) {
    row <- detailed_pred$top_3_predictions[i, ]
    cat(sprintf("  %d. %s: %.1f%%\n", i, row$class, row$probability * 100))
  }
  
  cat("\n")
}

# ------------------------------------------------------------------------------
# Evaluation Functions
# ------------------------------------------------------------------------------

#' Evaluate model on test set
#' @param model Loaded Keras model
#' @param test_images Test images array
#' @param test_labels Test labels (0-indexed integers)
#' @param class_names Vector of class names
#' @return List with evaluation metrics
evaluate_model <- function(model, test_images, test_labels, class_names = NULL) {
  
  library(keras3)
  
  if (is.null(class_names)) {
    class_names <- get_config("grade_classes")
  }
  
  num_classes <- length(class_names)
  test_labels_onehot <- to_categorical(test_labels, num_classes)
  
  # Get predictions
  predictions <- model %>% predict(test_images, verbose = 1)
  predicted_classes <- apply(predictions, 1, which.max) - 1
  
  # Calculate metrics
  correct <- sum(predicted_classes == test_labels)
  accuracy <- correct / length(test_labels)
  
  # Confusion matrix
  confusion <- table(
    Actual = factor(test_labels, levels = 0:(num_classes-1)),
    Predicted = factor(predicted_classes, levels = 0:(num_classes-1))
  )
  
  # Per-class metrics
  class_metrics <- data.frame(
    class = class_names,
    support = as.vector(table(factor(test_labels, levels = 0:(num_classes-1)))),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_len(num_classes)) {
    class_idx <- i - 1
    tp <- sum(predicted_classes == class_idx & test_labels == class_idx)
    fp <- sum(predicted_classes == class_idx & test_labels != class_idx)
    fn <- sum(predicted_classes != class_idx & test_labels == class_idx)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1 <- ifelse(precision + recall > 0, 
                 2 * precision * recall / (precision + recall), 0)
    
    class_metrics$precision[i] <- precision
    class_metrics$recall[i] <- recall
    class_metrics$f1[i] <- f1
  }
  
  # Loss
  loss_value <- model %>% evaluate(test_images, test_labels_onehot, verbose = 0)
  
  return(list(
    accuracy = accuracy,
    loss = loss_value[[1]],
    confusion_matrix = confusion,
    class_metrics = class_metrics,
    predictions = predictions,
    predicted_classes = predicted_classes,
    actual_classes = test_labels
  ))
}

#' Plot confusion matrix
#' @param confusion_matrix Confusion matrix from evaluate_model
#' @param class_names Vector of class names
#' @param save_path Optional path to save plot
plot_confusion_matrix <- function(confusion_matrix, class_names = NULL, save_path = NULL) {
  
  library(ggplot2)
  
  # Convert to data frame
  cm_df <- as.data.frame(as.table(confusion_matrix))
  colnames(cm_df) <- c("Actual", "Predicted", "Count")
  
  # Add labels
  if (!is.null(class_names)) {
    levels(cm_df$Actual) <- class_names
    levels(cm_df$Predicted) <- class_names
  }
  
  # Create heatmap
  p <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Confusion Matrix", x = "Predicted Grade", y = "Actual Grade") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 8)
    cat("Confusion matrix plot saved to:", save_path, "\n")
  }
  
  print(p)
  return(p)
}

#' Print evaluation summary
#' @param evaluation Result from evaluate_model
print_evaluation_summary <- function(evaluation) {
  
  cat("\n")
  cat("=", rep("=", 50), "\n", sep = "")
  cat("Model Evaluation Summary\n")
  cat("=", rep("=", 50), "\n", sep = "")
  
  cat(sprintf("\nOverall Accuracy: %.2f%%\n", evaluation$accuracy * 100))
  cat(sprintf("Overall Loss: %.4f\n", evaluation$loss))
  
  cat("\nPer-Class Metrics:\n")
  metrics <- evaluation$class_metrics
  cat(sprintf("%-12s %8s %9s %8s %8s\n", 
              "Class", "Support", "Precision", "Recall", "F1"))
  cat(rep("-", 50), "\n", sep = "")
  
  for (i in seq_len(nrow(metrics))) {
    cat(sprintf("%-12s %8d %9.2f%% %8.2f%% %8.2f%%\n",
                metrics$class[i],
                metrics$support[i],
                metrics$precision[i] * 100,
                metrics$recall[i] * 100,
                metrics$f1[i] * 100))
  }
  
  # Macro averages
  cat(rep("-", 50), "\n", sep = "")
  cat(sprintf("%-12s %8s %9.2f%% %8.2f%% %8.2f%%\n",
              "Macro Avg", "",
              mean(metrics$precision) * 100,
              mean(metrics$recall) * 100,
              mean(metrics$f1) * 100))
  
  # Weighted averages
  weights <- metrics$support / sum(metrics$support)
  cat(sprintf("%-12s %8s %9.2f%% %8.2f%% %8.2f%%\n",
              "Weighted Avg", "",
              sum(metrics$precision * weights) * 100,
              sum(metrics$recall * weights) * 100,
              sum(metrics$f1 * weights) * 100))
  
  cat("\n")
}

# ------------------------------------------------------------------------------
# Quick Prediction Functions
# ------------------------------------------------------------------------------

#' Quick predict: Load model and predict single image
#' @param image_path Path to image
#' @param model_path Path to model (uses default if NULL)
#' @return Prediction result
quick_predict <- function(image_path, model_path = NULL) {
  
  if (is.null(model_path)) {
    model_path <- file.path(get_config("models_dir"), "psa_grading_model.keras")
  }
  
  # Load model
  loaded <- load_grading_model(model_path)
  
  # Get detailed prediction
  result <- get_detailed_prediction(loaded$model, image_path, loaded$class_names)
  
  # Print results
  print_detailed_prediction(result)
  
  return(result)
}

#' Quick predict: Predict all images in a directory
#' @param image_dir Directory containing images
#' @param model_path Path to model (uses default if NULL)
#' @return Data frame with predictions
quick_predict_directory <- function(image_dir, model_path = NULL) {
  
  if (is.null(model_path)) {
    model_path <- file.path(get_config("models_dir"), "psa_grading_model.keras")
  }
  
  # Load model
  loaded <- load_grading_model(model_path)
  
  # Predict directory
  results <- predict_directory(loaded$model, image_dir, loaded$class_names)
  
  return(results)
}
