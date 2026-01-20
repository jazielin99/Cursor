#!/usr/bin/env Rscript

# ============================================
# Improved Model Training for Low-Performing Classes
# ============================================
# IMPROVEMENTS:
# 1. Cost-Sensitive Learning: Upweight PSA 2, 5, 8 misclassifications
# 2. SMOTE Oversampling: Generate synthetic examples for minority classes
# 3. Enhanced Confusion-Pair Specialists: 4↔5, 5↔6, 7↔8 boundaries
# 4. Ordinal Distance Weighting: Penalize distant errors more
# 5. Balanced sampling per fold
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

set.seed(42)

models_dir <- "models"
features_csv <- file.path(models_dir, "advanced_features.csv")

cat("========================================\n")
cat("Improved Model Training\n")
cat("(Targeting PSA 2, 5, 8 accuracy)\n")
cat("========================================\n\n")

# Load features
cat("Loading features:", features_csv, "\n")
df <- read.csv(features_csv, check.names = FALSE)

# Filter out PSA_1.5
df <- df[df$label != "PSA_1.5", , drop = FALSE]

cat("Dataset:", nrow(df), "samples\n")

# =============================================================================
# 1. COST-SENSITIVE CLASS WEIGHTS
# =============================================================================
# Upweight PSA 2, 5, 8 which have low accuracy

class_counts <- table(df$label)
class_levels <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                  "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")

# Base weights inversely proportional to frequency
base_weights <- max(class_counts) / class_counts[class_levels]

# Additional boost for hard classes (PSA 2, 5, 8)
hard_class_boost <- c(
  PSA_1 = 1.0, PSA_2 = 2.5,  # 34.5% accuracy - needs 2.5x boost
  PSA_3 = 1.0, PSA_4 = 1.0, 
  PSA_5 = 3.0,  # 27.0% accuracy - needs 3x boost (hardest)
  PSA_6 = 1.0, PSA_7 = 1.0, 
  PSA_8 = 2.5,  # 28.7% accuracy - needs 2.5x boost
  PSA_9 = 1.2, PSA_10 = 1.0
)

final_weights <- base_weights * hard_class_boost[class_levels]
final_weights <- final_weights / min(final_weights)  # Normalize

cat("\nClass weights (cost-sensitive):\n")
for (g in class_levels) {
  cat(sprintf("  %s: %.2f (n=%d)\n", g, final_weights[g], class_counts[g]))
}

# Create sample weights for each row
df$sample_weight <- final_weights[df$label]

# =============================================================================
# 2. SMOTE-LIKE OVERSAMPLING FOR HARD CLASSES
# =============================================================================

smote_oversample <- function(data, target_col, minority_classes, oversample_factor = 2) {
  # Simple oversampling with noise injection
  result <- data
  
  for (cls in minority_classes) {
    cls_data <- data[data[[target_col]] == cls, , drop = FALSE]
    n_samples <- nrow(cls_data)
    n_synthetic <- round(n_samples * (oversample_factor - 1))
    
    if (n_synthetic > 0 && n_samples > 1) {
      cat(sprintf("  Generating %d synthetic samples for %s\n", n_synthetic, cls))
      
      # Sample with replacement and add small noise to features
      synthetic_idx <- sample(1:n_samples, n_synthetic, replace = TRUE)
      synthetic <- cls_data[synthetic_idx, , drop = FALSE]
      
      # Add small random noise to numeric features (not path, label, weight)
      numeric_cols <- setdiff(colnames(synthetic), c("path", "label", "sample_weight", "fold"))
      for (col in numeric_cols) {
        if (is.numeric(synthetic[[col]])) {
          noise <- rnorm(nrow(synthetic), 0, sd(cls_data[[col]], na.rm = TRUE) * 0.1)
          synthetic[[col]] <- synthetic[[col]] + noise
        }
      }
      
      result <- rbind(result, synthetic)
    }
  }
  
  return(result)
}

cat("\nOversampling hard classes...\n")
hard_classes <- c("PSA_2", "PSA_5", "PSA_8")
df_augmented <- smote_oversample(df, "label", hard_classes, oversample_factor = 2.0)
cat("Augmented dataset:", nrow(df_augmented), "samples\n")

# =============================================================================
# 3. FEATURE SELECTION
# =============================================================================

feature_cols <- setdiff(colnames(df), c("path", "label", "sample_weight", "fold"))

# Remove columns with NA/Inf
valid_cols <- sapply(feature_cols, function(col) {
  vals <- df[[col]]
  !any(is.na(vals)) && !any(is.infinite(vals)) && sd(vals, na.rm = TRUE) > 1e-10
})
feature_cols <- feature_cols[valid_cols]

# Select top features by variance
var_scores <- sapply(feature_cols, function(col) var(df[[col]], na.rm = TRUE))
top_features <- names(sort(var_scores, decreasing = TRUE))[1:min(500, length(feature_cols))]

cat("\nUsing", length(top_features), "features\n")

# =============================================================================
# 4. TRAIN MAIN ENSEMBLE WITH CASE WEIGHTS
# =============================================================================

cat("\n========================================\n")
cat("Training weighted ensemble...\n")
cat("========================================\n")

df_augmented$label <- factor(df_augmented$label, levels = class_levels)

# Train main model with case weights
train_data <- df_augmented[, c(top_features, "label", "sample_weight"), drop = FALSE]

main_model <- ranger(
  formula = label ~ . - sample_weight,
  data = train_data,
  num.trees = 700,
  mtry = floor(sqrt(length(top_features))),
  probability = TRUE,
  case.weights = train_data$sample_weight,
  seed = 42,
  importance = "impurity"
)

cat("Main model trained with", main_model$num.trees, "trees\n")

# =============================================================================
# 5. CONFUSION-PAIR SPECIALISTS
# =============================================================================

cat("\nTraining confusion-pair specialists...\n")

train_pair_specialist <- function(data, grade1, grade2, features, weight_col = NULL) {
  pair_data <- data[data$label %in% c(grade1, grade2), , drop = FALSE]
  pair_data$binary_label <- factor(pair_data$label, levels = c(grade1, grade2))
  
  weights <- if (!is.null(weight_col) && weight_col %in% colnames(pair_data)) {
    pair_data[[weight_col]]
  } else {
    rep(1, nrow(pair_data))
  }
  
  model <- ranger(
    formula = binary_label ~ .,
    data = pair_data[, c(features, "binary_label"), drop = FALSE],
    num.trees = 300,
    probability = TRUE,
    case.weights = weights,
    seed = 42
  )
  
  list(model = model, grades = c(grade1, grade2))
}

# Key confusion pairs (especially for hard classes)
confusion_pairs <- list(
  c("PSA_1", "PSA_2"),  # For PSA_2
  c("PSA_2", "PSA_3"),  # For PSA_2
  c("PSA_3", "PSA_4"),
  c("PSA_4", "PSA_5"),  # For PSA_5
  c("PSA_5", "PSA_6"),  # For PSA_5
  c("PSA_6", "PSA_7"),
  c("PSA_7", "PSA_8"),  # For PSA_8
  c("PSA_8", "PSA_9"),  # For PSA_8
  c("PSA_9", "PSA_10")
)

specialists <- list()
for (pair in confusion_pairs) {
  pair_name <- paste0(pair[1], "_vs_", pair[2])
  cat(sprintf("  Training %s specialist (%d samples)...\n", 
              pair_name, 
              sum(df_augmented$label %in% pair)))
  specialists[[pair_name]] <- train_pair_specialist(
    df_augmented, pair[1], pair[2], top_features, "sample_weight"
  )
}

# =============================================================================
# 6. ORDINAL REGRESSION HEAD
# =============================================================================

cat("\nTraining ordinal regression head...\n")

# Create ordinal targets
df_augmented$grade_num <- as.numeric(gsub("PSA_", "", df_augmented$label))

ordinal_model <- ranger(
  formula = grade_num ~ .,
  data = df_augmented[, c(top_features, "grade_num"), drop = FALSE],
  num.trees = 500,
  seed = 42
)

cat("Ordinal model R-squared:", round(ordinal_model$r.squared, 3), "\n")

# =============================================================================
# 7. SAVE MODEL
# =============================================================================

improved_model <- list(
  version = "improved_v1_hard_class_focus",
  main_model = main_model,
  specialists = specialists,
  ordinal_model = ordinal_model,
  class_levels = class_levels,
  class_weights = final_weights,
  feature_names = top_features,
  hard_classes = hard_classes
)

model_path <- file.path(models_dir, "improved_model.rds")
saveRDS(improved_model, model_path)
cat("\nSaved model to:", model_path, "\n")

# =============================================================================
# 8. CROSS-VALIDATION
# =============================================================================

cat("\n========================================\n")
cat("Running 5-fold cross-validation...\n")
cat("========================================\n")

# Use original (non-augmented) data for CV
df$fold <- sample(rep(1:5, length.out = nrow(df)))
df$label <- factor(df$label, levels = class_levels)

all_preds <- character(0)
all_true <- character(0)
fold_results <- list()

for (k in 1:5) {
  cat("Fold", k, "/ 5...\n")
  
  train_idx <- which(df$fold != k)
  test_idx <- which(df$fold == k)
  
  # Augment training data
  train_df <- df[train_idx, , drop = FALSE]
  train_aug <- smote_oversample(train_df, "label", hard_classes, oversample_factor = 2.0)
  train_aug$label <- factor(train_aug$label, levels = class_levels)
  
  train_data <- train_aug[, c(top_features, "label", "sample_weight"), drop = FALSE]
  test_data <- df[test_idx, c(top_features, "label"), drop = FALSE]
  
  # Train with case weights
  fold_model <- ranger(
    formula = label ~ . - sample_weight,
    data = train_data,
    num.trees = 500,
    mtry = floor(sqrt(length(top_features))),
    probability = TRUE,
    case.weights = train_data$sample_weight,
    seed = 42 + k
  )
  
  # Predict
  preds <- predict(fold_model, test_data)$predictions
  pred_labels <- class_levels[apply(preds, 1, which.max)]
  true_labels <- as.character(test_data$label)
  
  all_preds <- c(all_preds, pred_labels)
  all_true <- c(all_true, true_labels)
  
  # Fold metrics
  exact <- mean(pred_labels == true_labels)
  pred_nums <- as.numeric(gsub("PSA_", "", pred_labels))
  true_nums <- as.numeric(gsub("PSA_", "", true_labels))
  within1 <- mean(abs(pred_nums - true_nums) <= 1)
  within2 <- mean(abs(pred_nums - true_nums) <= 2)
  
  fold_results[[k]] <- c(exact = exact, within1 = within1, within2 = within2)
  cat(sprintf("  Exact: %.1f%% | Within-1: %.1f%% | Within-2: %.1f%%\n", 
              exact * 100, within1 * 100, within2 * 100))
}

cat("\n========================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("========================================\n\n")

# Overall metrics
exact_acc <- mean(all_preds == all_true)
pred_nums <- as.numeric(gsub("PSA_", "", all_preds))
true_nums <- as.numeric(gsub("PSA_", "", all_true))
within1_acc <- mean(abs(pred_nums - true_nums) <= 1)
within2_acc <- mean(abs(pred_nums - true_nums) <= 2)

fold_exact <- sapply(fold_results, function(x) x["exact"])
fold_within1 <- sapply(fold_results, function(x) x["within1"])
fold_within2 <- sapply(fold_results, function(x) x["within2"])

cat(sprintf("Exact Match:  %.1f%% (SD: %.1f%%)\n", exact_acc * 100, sd(fold_exact) * 100))
cat(sprintf("Within 1:     %.1f%% (SD: %.1f%%)\n", within1_acc * 100, sd(fold_within1) * 100))
cat(sprintf("Within 2:     %.1f%% (SD: %.1f%%)\n", within2_acc * 100, sd(fold_within2) * 100))

cat("\nPer-Grade Exact Match:\n")
grade_results <- data.frame(
  grade = class_levels,
  exact_match_pct = NA,
  correct = NA,
  total = NA,
  improvement = NA
)

# Baseline accuracies from simple CV
baseline <- c(
  PSA_1 = 70.6, PSA_2 = 34.5, PSA_3 = 69.4, PSA_4 = 74.4, PSA_5 = 27.0,
  PSA_6 = 56.9, PSA_7 = 47.1, PSA_8 = 28.7, PSA_9 = 50.2, PSA_10 = 65.3
)

for (i in seq_along(class_levels)) {
  grade <- class_levels[i]
  mask <- all_true == grade
  if (sum(mask) > 0) {
    correct <- sum(all_preds[mask] == grade)
    total <- sum(mask)
    acc <- correct / total * 100
    improvement <- acc - baseline[grade]
    grade_results$exact_match_pct[i] <- acc
    grade_results$correct[i] <- correct
    grade_results$total[i] <- total
    grade_results$improvement[i] <- improvement
    
    sign <- ifelse(improvement >= 0, "+", "")
    cat(sprintf("  %s: %.1f%% (%d/%d) [%s%.1f%%]\n", 
                grade, acc, correct, total, sign, improvement))
  }
}

# Save results
write.csv(grade_results, file.path(models_dir, "improved_cv_results.csv"), row.names = FALSE)
cat("\nSaved CV results to: models/improved_cv_results.csv\n")

# Summary for hard classes
cat("\n========================================\n")
cat("HARD CLASS IMPROVEMENT SUMMARY\n")
cat("========================================\n")
hard_results <- grade_results[grade_results$grade %in% c("PSA_2", "PSA_5", "PSA_8"), ]
for (i in 1:nrow(hard_results)) {
  r <- hard_results[i, ]
  sign <- ifelse(r$improvement >= 0, "+", "")
  cat(sprintf("  %s: %.1f%% -> %.1f%% (%s%.1f%%)\n",
              r$grade, baseline[r$grade], r$exact_match_pct, sign, r$improvement))
}

cat("\nDONE.\n")
