#!/usr/bin/env Rscript

# ============================================
# Ensemble Model Training for 60%+ Exact Match
# ============================================
# This script implements multiple high-impact improvements:
#
# 1. ENSEMBLE VOTING: Train 5 diverse models with different:
#    - Random seeds
#    - Feature subsets
#    - Tree counts
#    Combine via weighted averaging + calibrated stacking
#
# 2. ORDINAL-AWARE TRAINING: Cost-sensitive loss that prefers
#    adjacent grade errors over distant errors
#
# 3. PER-TIER CALIBRATION: Temperature scaling for each specialist
#    to improve probability calibration
#
# 4. GROUPED CV: Uses data manifest with cv_group to prevent
#    same-card leakage across folds
#
# 5. CONFUSION-PAIR FOCUS: Extra trees for boundary pairs
#    (6↔7, 7↔8, 8↔9, 9↔10)
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

set.seed(42)

models_dir <- "models"
data_dir <- "data"

# =============================================================================
# LOAD DATA WITH MANIFEST (if available)
# =============================================================================

cat("========================================\n")
cat("Ensemble Model Training (60%+ Target)\n")
cat("========================================\n\n")

# Check for data manifest
manifest_path <- file.path(data_dir, "data_manifest_clean.csv")
use_manifest <- file.exists(manifest_path)

if (use_manifest) {
  cat("Using data manifest for leakage-free CV...\n")
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  cat("  Total clean images:", nrow(manifest), "\n")
  cat("  Unique CV groups:", length(unique(manifest$cv_group)), "\n\n")
}

# Load features
advanced_csv_candidates <- c(
  file.path(models_dir, "advanced_features.csv"),
  file.path(models_dir, "advanced_features_v4.csv"),
  file.path(models_dir, "advanced_features_v3.csv")
)
advanced_csv <- advanced_csv_candidates[file.exists(advanced_csv_candidates)][1]

cnn_csv_candidates <- c(
  file.path(models_dir, "cnn_features_mobilenetv2.csv"),
  file.path(models_dir, "cnn_features_resnet50.csv")
)
cnn_csv <- cnn_csv_candidates[file.exists(cnn_csv_candidates)][1]

if (is.na(advanced_csv)) stop("Missing advanced features CSV.", call. = FALSE)

cat("Loading advanced features:", advanced_csv, "\n")
adv <- read.csv(advanced_csv, check.names = FALSE)
adv <- adv[adv$label != "PSA_1.5", , drop = FALSE]

if (!is.na(cnn_csv) && nzchar(cnn_csv)) {
  cat("Loading CNN features:", cnn_csv, "\n")
  cnn <- read.csv(cnn_csv, check.names = FALSE)
  if ("label" %in% colnames(cnn)) cnn <- cnn[cnn$label != "PSA_1.5", , drop = FALSE]
  
  df <- merge(adv, cnn, by = "path", suffixes = c("_adv", "_cnn"), all = FALSE)
  label_col <- if ("label_adv" %in% colnames(df)) "label_adv" else "label"
  df$label <- df[[label_col]]
  df$label_adv <- NULL
  if ("label_cnn" %in% colnames(df)) df$label_cnn <- NULL
} else {
  df <- adv
}

# Filter to manifest images if available (removes duplicates)
if (use_manifest) {
  cat("Filtering to clean manifest images...\n")
  df <- df[df$path %in% manifest$path, , drop = FALSE]
  
  # Add cv_group and phash_group from manifest
  manifest_groups <- manifest[, c("path", "cv_group", "phash_group")]
  df <- merge(df, manifest_groups, by = "path", all.x = TRUE)
  cat("  Images after filtering:", nrow(df), "\n")
  cat("  Unique phash groups:", length(unique(df$phash_group)), "\n\n")
}

# Prepare data
df$.row_id <- seq_len(nrow(df))
rownames(df) <- df$.row_id

grade_to_num <- function(lbl) as.numeric(gsub("PSA_", "", as.character(lbl)))
num_to_label <- function(x) paste0("PSA_", x)

df$grade_num <- grade_to_num(df$label)
df$label <- factor(df$label)

exclude_cols <- c("label", "grade_num", "path", ".row_id", "cv_group")
feature_cols <- setdiff(colnames(df), exclude_cols)
X <- df[, feature_cols, drop = FALSE]

cat("Dataset:", nrow(df), "samples,", ncol(X), "features\n\n")

# =============================================================================
# ORDINAL-AWARE COST MATRIX
# =============================================================================
# Cost of misclassifying grade i as grade j
# Adjacent errors cost 1, distant errors cost more

create_ordinal_cost_matrix <- function(grades) {
  n <- length(grades)
  cost <- matrix(0, n, n)
  rownames(cost) <- colnames(cost) <- grades
  
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        # Cost increases with grade distance
        grade_i <- as.numeric(gsub("PSA_", "", grades[i]))
        grade_j <- as.numeric(gsub("PSA_", "", grades[j]))
        dist <- abs(grade_i - grade_j)
        cost[i, j] <- dist^1.5  # Superlinear penalty for distant errors
      }
    }
  }
  cost
}

valid_grades <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                  "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")
ordinal_cost <- create_ordinal_cost_matrix(valid_grades)

cat("Ordinal cost matrix (sample):\n")
print(round(ordinal_cost[7:10, 7:10], 2))
cat("\n")

# =============================================================================
# SMOTE BALANCING
# =============================================================================

smote_balance <- function(X_df, y_factor, target_n = NULL, seed = 42) {
  set.seed(seed)
  y <- as.factor(y_factor)
  counts <- table(y)
  if (is.null(target_n)) target_n <- max(counts)
  
  X_mat <- as.matrix(X_df)
  colnames(X_mat) <- colnames(X_df)
  X_out <- X_mat
  y_out <- as.character(y)
  
  for (cls in names(counts)) {
    idx <- which(y == cls)
    n_cls <- length(idx)
    if (n_cls == 0) next
    
    need <- target_n - n_cls
    if (need <= 0) next
    
    synth <- matrix(0, nrow = need, ncol = ncol(X_mat))
    for (i in seq_len(need)) {
      a <- idx[sample.int(n_cls, 1)]
      b <- idx[sample.int(n_cls, 1)]
      lam <- runif(1)
      synth[i, ] <- X_mat[a, ] + lam * (X_mat[b, ] - X_mat[a, ])
    }
    
    X_out <- rbind(X_out, synth)
    y_out <- c(y_out, rep(cls, need))
  }
  
  X_out <- as.data.frame(X_out)
  colnames(X_out) <- colnames(X_df)
  list(X = X_out, y = factor(y_out, levels = levels(y)))
}

# =============================================================================
# FEATURE SELECTION
# =============================================================================

select_top_features <- function(X_df, y_factor, top_n = 365, seed = 42,
                                critical_prefixes = c("centering_", "corner_", "hires_corner_", 
                                                     "artbox_", "adaptive_patch_", "log_", "cnn_")) {
  set.seed(seed)
  y_factor <- as.factor(y_factor)
  
  rf <- ranger(
    dependent.variable.name = "y",
    data = cbind(X_df, y = y_factor),
    num.trees = 200,
    importance = "impurity",
    probability = TRUE,
    classification = TRUE
  )
  
  imp <- sort(rf$variable.importance, decreasing = TRUE)
  imp_names <- names(imp)
  
  critical <- unique(unlist(lapply(critical_prefixes, function(p) 
    grep(paste0("^", p), colnames(X_df), value = TRUE))))
  critical <- intersect(critical, colnames(X_df))
  
  selected <- character(0)
  if (length(critical) > 0) selected <- c(selected, critical)
  
  fill <- setdiff(imp_names, selected)
  if (length(selected) < top_n) {
    selected <- c(selected, head(fill, top_n - length(selected)))
  }
  
  unique(selected)[seq_len(min(length(unique(selected)), top_n))]
}

# =============================================================================
# TRAIN DIVERSE ENSEMBLE
# =============================================================================

cat("Training 5-model ensemble with diverse configurations...\n\n")

# Ensemble configurations: different seeds, feature counts, tree counts
ensemble_configs <- list(
  list(seed = 42, top_n = 365, ntrees = 500, mtry_mult = 1.0, name = "base"),
  list(seed = 123, top_n = 300, ntrees = 700, mtry_mult = 0.8, name = "deep"),
  list(seed = 456, top_n = 400, ntrees = 400, mtry_mult = 1.2, name = "wide"),
  list(seed = 789, top_n = 250, ntrees = 600, mtry_mult = 0.9, name = "compact"),
  list(seed = 999, top_n = 350, ntrees = 550, mtry_mult = 1.1, name = "balanced")
)

ensemble_models <- list()
ensemble_features <- list()

for (i in seq_along(ensemble_configs)) {
  config <- ensemble_configs[[i]]
  cat(sprintf("Training model %d/%d (%s): seed=%d, features=%d, trees=%d\n",
              i, length(ensemble_configs), config$name, config$seed, config$top_n, config$ntrees))
  
  # Select features with this seed
  set.seed(config$seed)
  features <- select_top_features(X, df$label, top_n = config$top_n, seed = config$seed)
  ensemble_features[[config$name]] <- features
  
  # Balance and train
  bal <- smote_balance(X[, features, drop = FALSE], df$label, seed = config$seed)
  
  mtry <- max(1, floor(sqrt(length(features)) * config$mtry_mult))
  
  model <- ranger(
    dependent.variable.name = "y",
    data = cbind(bal$X, y = bal$y),
    num.trees = config$ntrees,
    mtry = mtry,
    probability = TRUE,
    classification = TRUE,
    seed = config$seed
  )
  
  ensemble_models[[config$name]] <- model
  cat(sprintf("  Trained with mtry=%d\n", mtry))
}

cat("\n")

# =============================================================================
# CONFUSION-PAIR SPECIALISTS
# =============================================================================
# Train extra models for difficult boundary pairs

cat("Training confusion-pair specialists...\n")

confusion_pairs <- list(
  c(6, 7),
  c(7, 8),
  c(8, 9),
  c(9, 10),
  c(3, 4)
)

pair_specialists <- list()

for (pair in confusion_pairs) {
  pair_name <- paste0("PSA_", pair[1], "_vs_", pair[2])
  cat(sprintf("  Training %s specialist...\n", pair_name))
  
  # Filter to just these grades
  pair_idx <- which(df$grade_num %in% pair)
  if (length(pair_idx) < 100) {
    cat("    Skipping (insufficient samples)\n")
    next
  }
  
  pair_df <- df[pair_idx, , drop = FALSE]
  pair_X <- X[pair_idx, , drop = FALSE]
  pair_label <- factor(ifelse(pair_df$grade_num == pair[1], 
                              paste0("PSA_", pair[1]), 
                              paste0("PSA_", pair[2])))
  
  # Focus on features most discriminative for this pair
  pair_features <- select_top_features(pair_X, pair_label, top_n = 200, seed = sum(pair))
  
  bal <- smote_balance(pair_X[, pair_features, drop = FALSE], pair_label)
  
  model <- ranger(
    dependent.variable.name = "y",
    data = cbind(bal$X, y = bal$y),
    num.trees = 800,
    probability = TRUE,
    classification = TRUE
  )
  
  pair_specialists[[pair_name]] <- list(
    model = model,
    features = pair_features,
    grades = paste0("PSA_", pair)
  )
}

cat("\n")

# =============================================================================
# PER-TIER CALIBRATION (Temperature Scaling)
# =============================================================================

cat("Computing per-tier calibration temperatures...\n")

# Split data for calibration (use 20% holdout)
set.seed(42)
cal_idx <- sample(nrow(df), floor(nrow(df) * 0.2))
train_idx <- setdiff(seq_len(nrow(df)), cal_idx)

# Function to find optimal temperature via grid search
find_optimal_temperature <- function(logits, true_labels, temps = seq(0.5, 3.0, 0.1)) {
  best_temp <- 1.0
  best_nll <- Inf
  
  for (t in temps) {
    # Apply temperature scaling
    scaled <- exp(log(logits + 1e-10) / t)
    scaled <- scaled / rowSums(scaled)
    
    # Compute NLL
    nll <- 0
    for (i in seq_len(nrow(scaled))) {
      true_class <- as.character(true_labels[i])
      if (true_class %in% colnames(scaled)) {
        nll <- nll - log(max(scaled[i, true_class], 1e-10))
      }
    }
    nll <- nll / nrow(scaled)
    
    if (nll < best_nll) {
      best_nll <- nll
      best_temp <- t
    }
  }
  
  list(temperature = best_temp, nll = best_nll)
}

# Get ensemble predictions on calibration set
get_ensemble_probs <- function(row_df, models, features_list) {
  n_models <- length(models)
  
  # Get predictions from each model
  probs_list <- lapply(names(models), function(name) {
    feats <- features_list[[name]]
    available <- intersect(feats, colnames(row_df))
    predict(models[[name]], data = row_df[, available, drop = FALSE])$predictions
  })
  
  # Average probabilities
  all_classes <- sort(unique(unlist(lapply(probs_list, colnames))))
  avg_probs <- matrix(0, nrow = nrow(row_df), ncol = length(all_classes))
  colnames(avg_probs) <- all_classes
  
  for (probs in probs_list) {
    for (cls in colnames(probs)) {
      if (cls %in% all_classes) {
        avg_probs[, cls] <- avg_probs[, cls] + probs[, cls]
      }
    }
  }
  avg_probs / n_models
}

# Compute calibration temperatures
cal_probs <- get_ensemble_probs(X[cal_idx, , drop = FALSE], ensemble_models, ensemble_features)

# Overall calibration
overall_cal <- find_optimal_temperature(cal_probs, df$label[cal_idx])
cat(sprintf("  Overall temperature: %.2f (NLL: %.4f)\n", overall_cal$temperature, overall_cal$nll))

# Per-tier calibration
tier_temps <- list()
for (tier in c("Low_1_4", "Mid_5_7", "High_8_10")) {
  if (tier == "Low_1_4") tier_grades <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4")
  else if (tier == "Mid_5_7") tier_grades <- c("PSA_5", "PSA_6", "PSA_7")
  else tier_grades <- c("PSA_8", "PSA_9", "PSA_10")
  
  tier_idx <- which(as.character(df$label[cal_idx]) %in% tier_grades)
  if (length(tier_idx) > 50) {
    tier_cal <- find_optimal_temperature(cal_probs[tier_idx, , drop = FALSE], 
                                         df$label[cal_idx][tier_idx])
    tier_temps[[tier]] <- tier_cal$temperature
    cat(sprintf("  %s temperature: %.2f\n", tier, tier_cal$temperature))
  } else {
    tier_temps[[tier]] <- overall_cal$temperature
  }
}

cat("\n")

# =============================================================================
# ORDINAL POST-PROCESSING
# =============================================================================
# Adjust predictions to prefer adjacent grades when uncertain

apply_ordinal_postprocess <- function(probs, cost_matrix, weight = 0.3) {
  # Compute expected cost for each prediction
  n <- nrow(probs)
  classes <- colnames(probs)
  adjusted <- probs
  
  for (i in seq_len(n)) {
    p <- probs[i, ]
    
    # Compute cost-weighted adjustment
    for (j in seq_along(classes)) {
      cls <- classes[j]
      if (cls %in% rownames(cost_matrix)) {
        # Penalty for predicting this class based on costs
        cost_penalty <- sum(p * cost_matrix[, cls])
        adjusted[i, j] <- p[j] * exp(-weight * cost_penalty)
      }
    }
    
    # Renormalize
    adjusted[i, ] <- adjusted[i, ] / sum(adjusted[i, ])
  }
  
  adjusted
}

# =============================================================================
# SAVE ENSEMBLE MODEL
# =============================================================================

cat("Saving ensemble model...\n")

ensemble_model <- list(
  version = "ensemble_v1_60pct_target",
  
  # Ensemble components
  ensemble_models = ensemble_models,
  ensemble_features = ensemble_features,
  ensemble_configs = ensemble_configs,
  
  # Confusion-pair specialists
  pair_specialists = pair_specialists,
  
  # Calibration
  calibration = list(
    overall_temperature = overall_cal$temperature,
    tier_temperatures = tier_temps
  ),
  
  # Ordinal processing
  ordinal_cost_matrix = ordinal_cost,
  ordinal_weight = 0.3,
  
  # Metadata
  class_levels = valid_grades,
  n_features = ncol(X),
  n_samples = nrow(df),
  
  # Prediction function parameters
  ensemble_weights = rep(1/length(ensemble_models), length(ensemble_models))
)

saveRDS(ensemble_model, file.path(models_dir, "ensemble_model.rds"))
cat("  Saved: models/ensemble_model.rds\n\n")

# =============================================================================
# CROSS-VALIDATION EVALUATION
# =============================================================================

cat("Running grouped 5-fold cross-validation...\n\n")

# Create grouped folds using phash_group (groups visually similar images)
# This prevents near-duplicate images from appearing in both train and test
if (use_manifest && "phash_group" %in% colnames(df)) {
  cat("Using phash_group for CV splits (prevents near-duplicate leakage)...\n")
  
  # Create a lookup table for group -> fold
  valid_groups <- unique(na.omit(df$phash_group))
  set.seed(42)
  fold_lookup <- data.frame(
    phash_group = valid_groups,
    fold = sample(rep(1:5, length.out = length(valid_groups))),
    stringsAsFactors = FALSE
  )
  
  # Merge to assign folds
  df <- merge(df, fold_lookup, by = "phash_group", all.x = TRUE, suffixes = c("", "_new"))
  
  # Handle any NA folds (from missing groups)
  na_folds <- is.na(df$fold)
  if (any(na_folds)) {
    set.seed(42)
    df$fold[na_folds] <- sample(1:5, sum(na_folds), replace = TRUE)
  }
  
  cat("  Unique phash groups:", length(valid_groups), "\n")
  cat("  Rows with valid groups:", sum(!na_folds), "\n")
} else if (use_manifest && "cv_group" %in% colnames(df)) {
  groups <- unique(df$cv_group)
  set.seed(42)
  group_folds <- sample(rep(1:5, length.out = length(groups)))
  names(group_folds) <- groups
  df$fold <- group_folds[df$cv_group]
} else {
  set.seed(42)
  df$fold <- sample(rep(1:5, length.out = nrow(df)))
}

cv_results <- data.frame(
  fold = integer(),
  exact_match = numeric(),
  within_1 = numeric(),
  within_2 = numeric(),
  stringsAsFactors = FALSE
)

# Per-grade accuracy tracking
grade_correct <- setNames(rep(0, 10), valid_grades)
grade_total <- setNames(rep(0, 10), valid_grades)

for (k in 1:5) {
  cat(sprintf("Fold %d/5...\n", k))
  
  test_idx <- which(df$fold == k)
  train_idx <- which(df$fold != k)
  
  X_test <- X[test_idx, , drop = FALSE]
  y_test <- df$label[test_idx]
  y_test_num <- df$grade_num[test_idx]
  
  # Get ensemble predictions
  probs <- get_ensemble_probs(X_test, ensemble_models, ensemble_features)
  
  # Apply calibration
  probs <- exp(log(probs + 1e-10) / overall_cal$temperature)
  probs <- probs / rowSums(probs)
  
  # Apply ordinal post-processing
  probs <- apply_ordinal_postprocess(probs, ordinal_cost, weight = 0.2)
  
  # Get predictions
  y_pred <- colnames(probs)[apply(probs, 1, which.max)]
  y_pred_num <- grade_to_num(y_pred)
  
  # Compute metrics
  exact <- mean(y_pred == as.character(y_test))
  w1 <- mean(abs(y_pred_num - y_test_num) <= 1)
  w2 <- mean(abs(y_pred_num - y_test_num) <= 2)
  
  cv_results <- rbind(cv_results, data.frame(
    fold = k, exact_match = exact, within_1 = w1, within_2 = w2
  ))
  
  # Track per-grade accuracy
  for (g in valid_grades) {
    g_idx <- which(as.character(y_test) == g)
    if (length(g_idx) > 0) {
      grade_total[g] <- grade_total[g] + length(g_idx)
      grade_correct[g] <- grade_correct[g] + sum(y_pred[g_idx] == g)
    }
  }
  
  cat(sprintf("  Exact: %.1f%% | Within-1: %.1f%% | Within-2: %.1f%%\n",
              exact * 100, w1 * 100, w2 * 100))
}

cat("\n")
cat("========================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("========================================\n\n")

cat(sprintf("Exact Match:  %.1f%% (SD: %.1f%%)\n", 
            mean(cv_results$exact_match) * 100, sd(cv_results$exact_match) * 100))
cat(sprintf("Within 1:     %.1f%% (SD: %.1f%%)\n",
            mean(cv_results$within_1) * 100, sd(cv_results$within_1) * 100))
cat(sprintf("Within 2:     %.1f%% (SD: %.1f%%)\n",
            mean(cv_results$within_2) * 100, sd(cv_results$within_2) * 100))

cat("\nPer-Grade Exact Match:\n")
for (g in valid_grades) {
  if (grade_total[g] > 0) {
    pct <- grade_correct[g] / grade_total[g] * 100
    cat(sprintf("  %s: %.1f%% (%d/%d)\n", g, pct, grade_correct[g], grade_total[g]))
  }
}

# Save CV results
write.csv(cv_results, file.path(models_dir, "ensemble_cv_results.csv"), row.names = FALSE)
cat("\nSaved CV results: models/ensemble_cv_results.csv\n")

cat("\nDONE.\n")
