#!/usr/bin/env Rscript

# ============================================
# Balanced Ensemble: Best of Both Worlds
# ============================================
# Strategy:
# 1. Train BASE model (no upweighting) for good overall accuracy
# 2. Train SPECIALIST model (with upweighting) for hard classes
# 3. Combine predictions with confidence-weighted blending
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

set.seed(42)

models_dir <- "models"
features_csv <- file.path(models_dir, "advanced_features.csv")

cat("========================================\n")
cat("Balanced Ensemble Training\n")
cat("========================================\n\n")

# Load features
df <- read.csv(features_csv, check.names = FALSE)
df <- df[df$label != "PSA_1.5", , drop = FALSE]

cat("Dataset:", nrow(df), "samples\n")

class_levels <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                  "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")
hard_classes <- c("PSA_2", "PSA_5", "PSA_8")

# Feature selection
feature_cols <- setdiff(colnames(df), c("path", "label"))
valid_cols <- sapply(feature_cols, function(col) {
  vals <- df[[col]]
  !any(is.na(vals)) && !any(is.infinite(vals)) && sd(vals, na.rm = TRUE) > 1e-10
})
feature_cols <- feature_cols[valid_cols]

var_scores <- sapply(feature_cols, function(col) var(df[[col]], na.rm = TRUE))
top_features <- names(sort(var_scores, decreasing = TRUE))[1:min(500, length(feature_cols))]

cat("Using", length(top_features), "features\n\n")

# =============================================================================
# CROSS-VALIDATION WITH ENSEMBLE
# =============================================================================

df$fold <- sample(rep(1:5, length.out = nrow(df)))
df$label <- factor(df$label, levels = class_levels)

all_preds <- character(0)
all_true <- character(0)
all_probs <- list()
fold_results <- list()

smote_oversample <- function(data, target_col, minority_classes, factor = 2) {
  result <- data
  for (cls in minority_classes) {
    cls_data <- data[data[[target_col]] == cls, , drop = FALSE]
    n <- nrow(cls_data)
    if (n > 1) {
      n_syn <- round(n * (factor - 1))
      syn <- cls_data[sample(1:n, n_syn, replace = TRUE), ]
      # Add noise
      num_cols <- setdiff(colnames(syn), c("path", "label", "fold"))
      for (col in num_cols) {
        if (is.numeric(syn[[col]])) {
          syn[[col]] <- syn[[col]] + rnorm(nrow(syn), 0, sd(cls_data[[col]], na.rm = TRUE) * 0.05)
        }
      }
      result <- rbind(result, syn)
    }
  }
  return(result)
}

cat("Running 5-fold CV with balanced ensemble...\n\n")

for (k in 1:5) {
  cat("Fold", k, "/ 5...\n")
  
  train_idx <- which(df$fold != k)
  test_idx <- which(df$fold == k)
  
  train_df <- df[train_idx, , drop = FALSE]
  test_df <- df[test_idx, , drop = FALSE]
  
  # MODEL 1: Base model (no upweighting)
  base_model <- ranger(
    formula = label ~ .,
    data = train_df[, c(top_features, "label"), drop = FALSE],
    num.trees = 500,
    mtry = floor(sqrt(length(top_features))),
    probability = TRUE,
    seed = 42 + k
  )
  
  # MODEL 2: Specialist model (upweight hard classes)
  train_aug <- smote_oversample(train_df, "label", hard_classes, factor = 1.5)
  
  class_counts <- table(train_df$label)
  hard_boost <- c(PSA_1=1, PSA_2=2, PSA_3=1, PSA_4=1, PSA_5=2.5, 
                  PSA_6=1, PSA_7=1, PSA_8=2, PSA_9=1, PSA_10=1)
  weights <- (max(class_counts) / class_counts[train_aug$label]) * hard_boost[as.character(train_aug$label)]
  
  specialist_model <- ranger(
    formula = label ~ .,
    data = train_aug[, c(top_features, "label"), drop = FALSE],
    num.trees = 500,
    probability = TRUE,
    case.weights = weights,
    seed = 42 + k + 100
  )
  
  # Predict with both models
  base_probs <- predict(base_model, test_df)$predictions
  spec_probs <- predict(specialist_model, test_df)$predictions
  
  # Combine: Use base for most grades, blend in specialist for hard classes
  combined_probs <- base_probs
  
  # For hard classes, blend specialist predictions
  for (cls in hard_classes) {
    cls_idx <- which(colnames(base_probs) == cls)
    
    # If base model is uncertain about this class, trust specialist more
    base_conf <- base_probs[, cls_idx]
    spec_conf <- spec_probs[, cls_idx]
    
    # Blend: more weight to specialist when base is uncertain
    blend_weight <- pmin(0.6, pmax(0.2, 1 - base_conf))
    combined_probs[, cls_idx] <- (1 - blend_weight) * base_conf + blend_weight * spec_conf
  }
  
  # Renormalize
  combined_probs <- combined_probs / rowSums(combined_probs)
  
  pred_labels <- class_levels[apply(combined_probs, 1, which.max)]
  true_labels <- as.character(test_df$label)
  
  all_preds <- c(all_preds, pred_labels)
  all_true <- c(all_true, true_labels)
  
  # Metrics
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

exact_acc <- mean(all_preds == all_true)
pred_nums <- as.numeric(gsub("PSA_", "", all_preds))
true_nums <- as.numeric(gsub("PSA_", "", all_true))
within1_acc <- mean(abs(pred_nums - true_nums) <= 1)
within2_acc <- mean(abs(pred_nums - true_nums) <= 2)

fold_exact <- sapply(fold_results, function(x) x["exact"])

cat(sprintf("Exact Match:  %.1f%% (SD: %.1f%%)\n", exact_acc * 100, sd(fold_exact) * 100))
cat(sprintf("Within 1:     %.1f%%\n", within1_acc * 100))
cat(sprintf("Within 2:     %.1f%%\n", within2_acc * 100))

cat("\nPer-Grade Exact Match:\n")

baseline <- c(PSA_1=70.6, PSA_2=34.5, PSA_3=69.4, PSA_4=74.4, PSA_5=27.0,
              PSA_6=56.9, PSA_7=47.1, PSA_8=28.7, PSA_9=50.2, PSA_10=65.3)

results <- data.frame(grade = class_levels, accuracy = NA, correct = NA, total = NA, change = NA)

for (i in seq_along(class_levels)) {
  g <- class_levels[i]
  mask <- all_true == g
  if (sum(mask) > 0) {
    correct <- sum(all_preds[mask] == g)
    total <- sum(mask)
    acc <- correct / total * 100
    change <- acc - baseline[g]
    results$accuracy[i] <- acc
    results$correct[i] <- correct
    results$total[i] <- total
    results$change[i] <- change
    
    sign <- ifelse(change >= 0, "+", "")
    cat(sprintf("  %s: %.1f%% (%d/%d) [%s%.1f%%]\n", g, acc, correct, total, sign, change))
  }
}

write.csv(results, file.path(models_dir, "balanced_cv_results.csv"), row.names = FALSE)

cat("\n========================================\n")
cat("HARD CLASS CHANGES\n")
cat("========================================\n")
for (cls in hard_classes) {
  r <- results[results$grade == cls, ]
  sign <- ifelse(r$change >= 0, "+", "")
  cat(sprintf("  %s: %.1f%% -> %.1f%% (%s%.1f%%)\n", 
              cls, baseline[cls], r$accuracy, sign, r$change))
}

cat("\nDONE.\n")
