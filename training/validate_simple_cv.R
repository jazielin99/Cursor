#!/usr/bin/env Rscript

# Simple 5-Fold CV (random splits, no grouping)
# This reproduces the ~57% baseline accuracy

suppressPackageStartupMessages({
  library(ranger)
})

set.seed(42)

models_dir <- "models"
features_csv <- file.path(models_dir, "advanced_features.csv")

cat("========================================\n")
cat("Simple 5-Fold Cross-Validation\n")
cat("(Random splits - baseline validation)\n")
cat("========================================\n\n")

# Load features
cat("Loading features:", features_csv, "\n")
df <- read.csv(features_csv, check.names = FALSE)

# Filter out PSA_1.5
df <- df[df$label != "PSA_1.5", , drop = FALSE]

cat("Dataset:", nrow(df), "samples\n\n")

# Get feature columns (exclude path, label)
feature_cols <- setdiff(colnames(df), c("path", "label"))

# Remove columns with NA/Inf
valid_cols <- sapply(feature_cols, function(col) {
  vals <- df[[col]]
  !any(is.na(vals)) && !any(is.infinite(vals)) && sd(vals, na.rm = TRUE) > 1e-10
})
feature_cols <- feature_cols[valid_cols]

cat("Using", length(feature_cols), "features\n\n")

# Create random 5-fold assignment
df$fold <- sample(rep(1:5, length.out = nrow(df)))

# Class levels
class_levels <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                  "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")
df$label <- factor(df$label, levels = class_levels)

# Select top features by variance
var_scores <- sapply(feature_cols, function(col) var(df[[col]], na.rm = TRUE))
top_features <- names(sort(var_scores, decreasing = TRUE))[1:min(500, length(feature_cols))]

cat("Running 5-fold CV...\n\n")

# Store results
all_preds <- character(0)
all_true <- character(0)
fold_results <- list()

for (k in 1:5) {
  cat("Fold", k, "/ 5...\n")
  
  train_idx <- which(df$fold != k)
  test_idx <- which(df$fold == k)
  
  train_data <- df[train_idx, c(top_features, "label"), drop = FALSE]
  test_data <- df[test_idx, c(top_features, "label"), drop = FALSE]
  
  # Train random forest
  model <- ranger(
    formula = label ~ .,
    data = train_data,
    num.trees = 500,
    mtry = floor(sqrt(length(top_features))),
    probability = TRUE,
    seed = 42 + k
  )
  
  # Predict
  preds <- predict(model, test_data)$predictions
  pred_labels <- class_levels[apply(preds, 1, which.max)]
  true_labels <- as.character(test_data$label)
  
  all_preds <- c(all_preds, pred_labels)
  all_true <- c(all_true, true_labels)
  
  # Fold metrics
  exact <- mean(pred_labels == true_labels)
  
  # Within-1 calculation
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
for (grade in class_levels) {
  mask <- all_true == grade
  if (sum(mask) > 0) {
    correct <- sum(all_preds[mask] == grade)
    total <- sum(mask)
    acc <- correct / total * 100
    cat(sprintf("  %s: %.1f%% (%d/%d)\n", grade, acc, correct, total))
  }
}

# Save results
results <- data.frame(
  grade = class_levels,
  exact_match_pct = sapply(class_levels, function(g) {
    mask <- all_true == g
    if (sum(mask) > 0) sum(all_preds[mask] == g) / sum(mask) * 100 else NA
  }),
  correct = sapply(class_levels, function(g) {
    mask <- all_true == g
    sum(all_preds[mask] == g)
  }),
  total = sapply(class_levels, function(g) sum(all_true == g))
)

write.csv(results, file.path(models_dir, "simple_cv_results.csv"), row.names = FALSE)
cat("\nSaved results to: models/simple_cv_results.csv\n")

cat("\nDONE.\n")
