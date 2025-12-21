# ============================================
# Quick Ensemble Training (Skip CV)
# Trains RF + XGBoost + Stacking directly
# ============================================

library(randomForest)
library(magick)
library(xgboost)

cat("========================================\n")
cat("Quick Ensemble Training\n")
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

# --- Light Upsampling (no CV) ---
cat("\n========================================\n")
cat("Upsampling minority classes\n")
cat("========================================\n")

class_counts <- table(y)
target_count <- floor(max(class_counts) * 0.6)  # Less aggressive

set.seed(42)
X_up <- X
y_up <- as.character(y)

for (class_name in names(class_counts)) {
  current_count <- class_counts[class_name]
  if (current_count < target_count) {
    need <- min(target_count - current_count, current_count * 2)  # Cap upsampling
    class_idx <- which(y == class_name)
    sample_idx <- sample(class_idx, need, replace = TRUE)
    new_samples <- X[sample_idx, , drop = FALSE]
    
    for (j in 1:ncol(new_samples)) {
      noise_sd <- sd(X[, j]) * 0.01
      new_samples[, j] <- new_samples[, j] + rnorm(nrow(new_samples), 0, noise_sd)
    }
    
    X_up <- rbind(X_up, new_samples)
    y_up <- c(y_up, rep(class_name, need))
  }
}

y_up <- factor(y_up, levels = class_levels)
cat("Upsampled:", nrow(X_up), "samples\n")

# Convert to numeric grades
grade_to_num <- function(g) {
  g <- gsub("PSA_", "", as.character(g))
  as.numeric(g)
}
y_num <- sapply(y_up, grade_to_num)
num_classes <- length(class_levels)

# --- Train Final Models ---
cat("\n========================================\n")
cat("Training Final Models\n")
cat("========================================\n")

cat("Training RF Classification (300 trees)...\n")
rf_final <- randomForest(x = X_up, y = y_up, ntree = 300, importance = TRUE)

cat("Training RF Regression (300 trees)...\n")
rf_reg_final <- randomForest(x = X_up, y = y_num, ntree = 300)

cat("Training XGBoost (150 rounds)...\n")
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

cat("Training Stacking Meta-learner...\n")
rf_all_prob <- predict(rf_final, X_up, type = "prob")
rf_all_class <- sapply(class_levels[max.col(rf_all_prob)], grade_to_num)
rf_all_reg <- predict(rf_reg_final, X_up)
xgb_all_prob <- predict(xgb_final, dall)
xgb_all_prob_matrix <- matrix(xgb_all_prob, ncol = num_classes, byrow = TRUE)
xgb_all_class <- sapply(class_levels[max.col(xgb_all_prob_matrix)], grade_to_num)

meta_all <- cbind(
  rf_class = rf_all_class,
  rf_reg = rf_all_reg,
  xgb_class = xgb_all_class,
  rf_conf = apply(rf_all_prob, 1, max),
  xgb_conf = apply(xgb_all_prob_matrix, 1, max)
)

meta_final <- randomForest(x = meta_all, y = y_num, ntree = 100)

# Save ensemble
cat("\nSaving models...\n")
ensemble <- list(
  rf_class = rf_final,
  rf_reg = rf_reg_final,
  xgb = xgb_final,
  meta = meta_final,
  class_levels = class_levels
)
saveRDS(ensemble, "models/psa_ensemble.rds")
cat("Ensemble saved: models/psa_ensemble.rds\n")

# --- Quick Validation on Training Data ---
cat("\n========================================\n")
cat("Training Set Performance (Reference)\n")
cat("========================================\n")

# Get predictions
rf_pred <- predict(rf_final, X_up)
rf_pred_num <- sapply(as.character(rf_pred), grade_to_num)

xgb_pred_prob <- predict(xgb_final, dall)
xgb_pred_matrix <- matrix(xgb_pred_prob, ncol = num_classes, byrow = TRUE)
xgb_pred_class <- class_levels[max.col(xgb_pred_matrix)]
xgb_pred_num <- sapply(xgb_pred_class, grade_to_num)

# Meta predictions
meta_pred <- predict(meta_final, meta_all)
meta_pred_rounded <- pmax(1, pmin(10, round(meta_pred)))

cat("RF Classification Accuracy:", round(mean(rf_pred_num == y_num) * 100, 1), "%\n")
cat("XGBoost Accuracy:", round(mean(xgb_pred_num == y_num) * 100, 1), "%\n")
cat("Stacking Ensemble Accuracy:", round(mean(meta_pred_rounded == y_num) * 100, 1), "%\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
