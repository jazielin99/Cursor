# ============================================
# Ensemble Model with 5-Fold Cross-Validation
# RF + XGBoost + Stacking with Upsampling
# ============================================

library(randomForest)
library(magick)
library(xgboost)

cat("========================================\n")
cat("Ensemble Training with Cross-Validation\n")
cat("========================================\n\n")

# --- Load cached features ---
cache_file <- "models/features_cache.rds"
if (file.exists(cache_file)) {
  cat("Loading cached features...\n")
  cache <- readRDS(cache_file)
  X <- cache$X
  y <- cache$y
  class_levels <- cache$class_levels
} else {
  stop("Feature cache not found. Run train_balanced_model.R first.")
}

cat("Dataset:", nrow(X), "images,", ncol(X), "features\n")
cat("Classes:", paste(class_levels, collapse = ", "), "\n\n")

# Convert to numeric grades
grade_to_num <- function(g) {
  g <- gsub("PSA_", "", as.character(g))
  as.numeric(g)
}
y_num <- sapply(as.character(y), grade_to_num)
num_classes <- length(class_levels)

# --- 5-Fold Cross-Validation ---
cat("========================================\n")
cat("5-Fold Cross-Validation\n")
cat("========================================\n\n")

set.seed(42)
n <- nrow(X)
K <- 5
folds <- sample(rep(1:K, length.out = n))

# Storage for predictions
all_rf_pred <- rep(NA, n)
all_xgb_pred <- rep(NA, n)
all_rf_reg_pred <- rep(NA, n)
all_ensemble_pred <- rep(NA, n)

fold_results <- data.frame(
  Fold = integer(),
  RF_Acc = numeric(),
  XGB_Acc = numeric(),
  RF_Reg_Acc = numeric(),
  Ensemble_Acc = numeric()
)

for (k in 1:K) {
  cat("=== Fold", k, "/", K, "===\n")
  
  train_idx <- which(folds != k)
  test_idx <- which(folds == k)
  
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  y_train_num <- y_num[train_idx]
  X_test <- X[test_idx, ]
  y_test <- y[test_idx]
  y_test_num <- y_num[test_idx]
  
  # --- Upsample training data ---
  cat("  Upsampling training data...\n")
  class_counts <- table(y_train)
  target_count <- floor(max(class_counts) * 0.7)
  
  X_up <- X_train
  y_up <- as.character(y_train)
  y_up_num <- y_train_num
  
  for (class_name in names(class_counts)) {
    current_count <- class_counts[class_name]
    if (current_count < target_count) {
      need <- min(target_count - current_count, current_count * 2)
      class_idx <- which(y_train == class_name)
      sample_idx <- sample(class_idx, need, replace = TRUE)
      
      new_X <- X_train[sample_idx, , drop = FALSE]
      # Add small noise
      for (j in 1:ncol(new_X)) {
        noise_sd <- sd(X_train[, j]) * 0.01
        new_X[, j] <- new_X[, j] + rnorm(nrow(new_X), 0, noise_sd)
      }
      
      X_up <- rbind(X_up, new_X)
      y_up <- c(y_up, rep(class_name, need))
      y_up_num <- c(y_up_num, rep(grade_to_num(class_name), need))
    }
  }
  y_up <- factor(y_up, levels = class_levels)
  cat("  Training samples:", nrow(X_up), "\n")
  
  # --- Train RF Classification ---
  cat("  Training RF Classification...\n")
  rf_model <- randomForest(x = X_up, y = y_up, ntree = 200)
  rf_pred <- predict(rf_model, X_test)
  rf_pred_num <- sapply(as.character(rf_pred), grade_to_num)
  all_rf_pred[test_idx] <- rf_pred_num
  
  # --- Train RF Regression ---
  cat("  Training RF Regression...\n")
  rf_reg_model <- randomForest(x = X_up, y = y_up_num, ntree = 200)
  rf_reg_pred <- predict(rf_reg_model, X_test)
  rf_reg_pred_rounded <- pmax(1, pmin(10, round(rf_reg_pred)))
  all_rf_reg_pred[test_idx] <- rf_reg_pred_rounded
  
  # --- Train XGBoost ---
  cat("  Training XGBoost...\n")
  y_up_xgb <- as.integer(y_up) - 1
  dtrain <- xgb.DMatrix(data = X_up, label = y_up_xgb)
  dtest <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = num_classes,
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
  xgb_prob <- predict(xgb_model, dtest)
  xgb_prob_matrix <- matrix(xgb_prob, ncol = num_classes, byrow = TRUE)
  xgb_pred_class <- max.col(xgb_prob_matrix)
  xgb_pred_num <- sapply(class_levels[xgb_pred_class], grade_to_num)
  all_xgb_pred[test_idx] <- xgb_pred_num
  
  # --- Simple Ensemble (Average) ---
  # Combine RF classification, RF regression, and XGBoost
  ensemble_pred <- (rf_pred_num + rf_reg_pred_rounded + xgb_pred_num) / 3
  ensemble_pred_rounded <- pmax(1, pmin(10, round(ensemble_pred)))
  all_ensemble_pred[test_idx] <- ensemble_pred_rounded
  
  # --- Fold Results ---
  rf_acc <- mean(rf_pred_num == y_test_num)
  xgb_acc <- mean(xgb_pred_num == y_test_num)
  rf_reg_acc <- mean(rf_reg_pred_rounded == y_test_num)
  ens_acc <- mean(ensemble_pred_rounded == y_test_num)
  
  cat(sprintf("  RF: %.1f%%, XGB: %.1f%%, RF_Reg: %.1f%%, Ensemble: %.1f%%\n\n",
              rf_acc * 100, xgb_acc * 100, rf_reg_acc * 100, ens_acc * 100))
  
  fold_results <- rbind(fold_results, data.frame(
    Fold = k,
    RF_Acc = rf_acc,
    XGB_Acc = xgb_acc,
    RF_Reg_Acc = rf_reg_acc,
    Ensemble_Acc = ens_acc
  ))
  
  gc(verbose = FALSE)
}

# --- Overall CV Results ---
cat("========================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("========================================\n\n")

cat("Per-Fold Accuracy (%):\n")
print(round(fold_results[, -1] * 100, 1))

cat("\n--- Summary Statistics ---\n")
cat(sprintf("RF Classification:  Mean = %.1f%% (SD = %.1f%%)\n",
            mean(fold_results$RF_Acc) * 100, sd(fold_results$RF_Acc) * 100))
cat(sprintf("XGBoost:            Mean = %.1f%% (SD = %.1f%%)\n",
            mean(fold_results$XGB_Acc) * 100, sd(fold_results$XGB_Acc) * 100))
cat(sprintf("RF Regression:      Mean = %.1f%% (SD = %.1f%%)\n",
            mean(fold_results$RF_Reg_Acc) * 100, sd(fold_results$RF_Reg_Acc) * 100))
cat(sprintf("Ensemble (Avg):     Mean = %.1f%% (SD = %.1f%%)\n",
            mean(fold_results$Ensemble_Acc) * 100, sd(fold_results$Ensemble_Acc) * 100))

# --- Within-Grade Analysis ---
cat("\n--- Grade Proximity Analysis ---\n")
cat(sprintf("Exact match:      %.1f%%\n", mean(all_ensemble_pred == y_num, na.rm = TRUE) * 100))
cat(sprintf("Within 1 grade:   %.1f%%\n", mean(abs(all_ensemble_pred - y_num) <= 1, na.rm = TRUE) * 100))
cat(sprintf("Within 2 grades:  %.1f%%\n", mean(abs(all_ensemble_pred - y_num) <= 2, na.rm = TRUE) * 100))

# --- Per-Class CV Accuracy ---
cat("\n--- Per-Class Accuracy (Ensemble) ---\n")
for (lvl in class_levels) {
  grade_num <- grade_to_num(lvl)
  idx <- which(y_num == grade_num)
  if (length(idx) > 0) {
    acc <- mean(all_ensemble_pred[idx] == y_num[idx], na.rm = TRUE)
    cat(sprintf("  %s: %.1f%% (%d samples)\n", lvl, acc * 100, length(idx)))
  }
}

# --- Train Final Models on All Data with Upsampling ---
cat("\n========================================\n")
cat("Training Final Ensemble on All Data\n")
cat("========================================\n")

# Upsample all data
class_counts <- table(y)
target_count <- floor(max(class_counts) * 0.7)

X_up <- X
y_up <- as.character(y)
y_up_num <- y_num

for (class_name in names(class_counts)) {
  current_count <- class_counts[class_name]
  if (current_count < target_count) {
    need <- min(target_count - current_count, current_count * 2)
    class_idx <- which(y == class_name)
    sample_idx <- sample(class_idx, need, replace = TRUE)
    
    new_X <- X[sample_idx, , drop = FALSE]
    for (j in 1:ncol(new_X)) {
      noise_sd <- sd(X[, j]) * 0.01
      new_X[, j] <- new_X[, j] + rnorm(nrow(new_X), 0, noise_sd)
    }
    
    X_up <- rbind(X_up, new_X)
    y_up <- c(y_up, rep(class_name, need))
    y_up_num <- c(y_up_num, rep(grade_to_num(class_name), need))
  }
}
y_up <- factor(y_up, levels = class_levels)
cat("Final training samples:", nrow(X_up), "\n")

cat("Training final RF Classification (300 trees)...\n")
rf_final <- randomForest(x = X_up, y = y_up, ntree = 300, importance = TRUE)

cat("Training final RF Regression (300 trees)...\n")
rf_reg_final <- randomForest(x = X_up, y = y_up_num, ntree = 300)

cat("Training final XGBoost (150 rounds)...\n")
y_up_xgb <- as.integer(y_up) - 1
dall <- xgb.DMatrix(data = X_up, label = y_up_xgb)
params <- list(
  objective = "multi:softprob",
  num_class = num_classes,
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)
xgb_final <- xgb.train(params = params, data = dall, nrounds = 150, verbose = 0)

# Save ensemble
cat("\nSaving models...\n")
ensemble_cv <- list(
  rf_class = rf_final,
  rf_reg = rf_reg_final,
  xgb = xgb_final,
  class_levels = class_levels,
  cv_results = fold_results
)
saveRDS(ensemble_cv, "models/psa_ensemble_cv.rds")

cat("Ensemble saved: models/psa_ensemble_cv.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
