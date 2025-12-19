# K-Fold Cross-Validation for PSA Grading Model
suppressPackageStartupMessages({
  library(magick)
  library(tidyverse)
  library(randomForest)
  library(caret)
})

source("R/config.R")
source("R/02_data_preparation.R")

cat("========================================\n")
cat("PSA Card Grading - 5-Fold CV\n")
cat("========================================\n\n")

# Load data
cat("Loading images...\n")
dataset <- load_training_dataset()
cat("\nTotal images:", dim(dataset$images)[1], "\n")

# Feature extraction function
extract_features <- function(images) {
  n <- dim(images)[1]
  features_list <- list()
  
  for (i in seq_len(n)) {
    img <- images[i, , , ]
    r <- img[, , 1]; g <- img[, , 2]; b <- img[, , 3]
    gray <- 0.299 * r + 0.587 * g + 0.114 * b
    h <- dim(img)[1]; w <- dim(img)[2]
    
    features <- c(
      r_mean = mean(r), r_sd = sd(r),
      g_mean = mean(g), g_sd = sd(g),
      b_mean = mean(b), b_sd = sd(b),
      gray_mean = mean(gray), gray_sd = sd(gray)
    )
    
    r_hist <- hist(r, breaks = seq(0, 1, length.out = 9), plot = FALSE)$counts / length(r)
    g_hist <- hist(g, breaks = seq(0, 1, length.out = 9), plot = FALSE)$counts / length(g)
    b_hist <- hist(b, breaks = seq(0, 1, length.out = 9), plot = FALSE)$counts / length(b)
    names(r_hist) <- paste0("r_h", 1:8)
    names(g_hist) <- paste0("g_h", 1:8)
    names(b_hist) <- paste0("b_h", 1:8)
    
    cs <- floor(min(h, w) / 10)
    corner_stats <- c(
      c_tl_m = mean(gray[1:cs, 1:cs]), c_tl_s = sd(gray[1:cs, 1:cs]),
      c_tr_m = mean(gray[1:cs, (w-cs+1):w]), c_tr_s = sd(gray[1:cs, (w-cs+1):w]),
      c_bl_m = mean(gray[(h-cs+1):h, 1:cs]), c_bl_s = sd(gray[(h-cs+1):h, 1:cs]),
      c_br_m = mean(gray[(h-cs+1):h, (w-cs+1):w]), c_br_s = sd(gray[(h-cs+1):h, (w-cs+1):w])
    )
    
    es <- floor(min(h, w) / 15)
    edge_stats <- c(
      e_top = mean(gray[1:es, ]), e_bot = mean(gray[(h-es+1):h, ]),
      e_left = mean(gray[, 1:es]), e_right = mean(gray[, (w-es+1):w])
    )
    
    center_stats <- c(
      lr_diff = abs(mean(gray[, 1:(w/2)]) - mean(gray[, (w/2+1):w])),
      tb_diff = abs(mean(gray[1:(h/2), ]) - mean(gray[(h/2+1):h, ]))
    )
    
    gx <- abs(diff(apply(gray, 1, mean)))
    gy <- abs(diff(apply(gray, 2, mean)))
    grad_stats <- c(
      grad_x_mean = mean(gx), grad_x_max = max(gx),
      grad_y_mean = mean(gy), grad_y_max = max(gy)
    )
    
    features_list[[i]] <- c(features, r_hist, g_hist, b_hist, corner_stats, edge_stats, center_stats, grad_stats)
  }
  do.call(rbind, features_list)
}

# Extract all features
cat("Extracting features...\n")
all_features <- extract_features(dataset$images)
all_labels <- dataset$labels
actual_levels <- sort(unique(all_labels))
all_labels_f <- factor(all_labels, levels = actual_levels)

cat("Features per image:", ncol(all_features), "\n")
cat("Number of classes:", length(actual_levels), "\n")

# K-Fold Cross-Validation
K <- 5
cat("\n========================================\n")
cat(K, "-Fold Cross-Validation\n", sep="")
cat("========================================\n")

set.seed(42)
folds <- createFolds(all_labels_f, k = K, list = TRUE, returnTrain = FALSE)

fold_accuracies <- numeric(K)
all_predictions <- numeric(length(all_labels))
all_actuals <- numeric(length(all_labels))

for (k in 1:K) {
  test_idx <- folds[[k]]
  train_idx <- setdiff(1:nrow(all_features), test_idx)
  
  train_features <- all_features[train_idx, ]
  train_labels <- all_labels_f[train_idx]
  test_features <- all_features[test_idx, ]
  test_labels <- all_labels_f[test_idx]
  
  rf_model <- randomForest(x = train_features, y = train_labels, ntree = 500)
  predictions <- predict(rf_model, test_features)
  
  all_predictions[test_idx] <- as.numeric(as.character(predictions))
  all_actuals[test_idx] <- as.numeric(as.character(test_labels))
  
  fold_accuracies[k] <- mean(predictions == test_labels)
  cat(sprintf("Fold %d: %.1f%%\n", k, fold_accuracies[k] * 100))
}

mean_acc <- mean(fold_accuracies)
sd_acc <- sd(fold_accuracies)

cat("\n========================================\n")
cat("CV RESULTS\n")
cat("========================================\n")
cat(sprintf("Mean Accuracy: %.1f%% (+/- %.1f%%)\n", mean_acc * 100, sd_acc * 100))

# Confusion matrix
cat("\nConfusion Matrix (all folds combined):\n")
all_pred_f <- factor(all_predictions, levels = actual_levels)
all_act_f <- factor(all_actuals, levels = actual_levels)
conf_mat <- table(Actual = all_act_f, Predicted = all_pred_f)
print(conf_mat)

# Class names
class_names <- dataset$class_names[actual_levels + 1]
cat("\nClass labels: ", paste(paste0(actual_levels, "=", class_names), collapse=", "), "\n")

# Per-class accuracy
cat("\n========================================\n")
cat("Per-Class Accuracy\n")
cat("========================================\n")
for (i in seq_along(actual_levels)) {
  lvl <- as.character(actual_levels[i])
  total <- sum(conf_mat[lvl, ])
  correct <- conf_mat[lvl, lvl]
  acc <- if(total > 0) correct / total * 100 else 0
  cat(sprintf("%-8s: %5.1f%% (%d/%d)\n", class_names[i], acc, correct, total))
}

# Adjacent accuracy
cat("\n========================================\n")
cat("Relaxed Accuracy Metrics\n")
cat("========================================\n")
within1 <- sum(abs(all_predictions - all_actuals) <= 1)
within2 <- sum(abs(all_predictions - all_actuals) <= 2)
n <- length(all_predictions)

cat(sprintf("Exact match:      %5.1f%% (%d/%d)\n", mean_acc * 100, sum(all_predictions == all_actuals), n))
cat(sprintf("Within 1 grade:   %5.1f%% (%d/%d)\n", within1/n*100, within1, n))
cat(sprintf("Within 2 grades:  %5.1f%% (%d/%d)\n", within2/n*100, within2, n))

# Train final model on all data
cat("\n========================================\n")
cat("Training Final Model (all data)\n")
cat("========================================\n")
final_model <- randomForest(x = all_features, y = all_labels_f, ntree = 500, importance = TRUE)

model_data <- list(
  model = final_model,
  class_names = class_names,
  class_levels = actual_levels,
  feature_names = colnames(all_features),
  cv_accuracy = mean_acc,
  cv_sd = sd_acc
)
saveRDS(model_data, "models/psa_rf_model_cv.rds")
cat("Model saved: models/psa_rf_model_cv.rds\n")

cat("\n========================================\n")
cat("SUMMARY\n")
cat("========================================\n")
cat(sprintf("5-Fold CV Accuracy: %.1f%% (+/- %.1f%%)\n", mean_acc * 100, sd_acc * 100))
cat(sprintf("Within 1 grade:     %.1f%%\n", within1/n*100))
cat(sprintf("Within 2 grades:    %.1f%%\n", within2/n*100))
