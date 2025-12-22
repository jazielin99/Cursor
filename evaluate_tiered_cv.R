#!/usr/bin/env Rscript

# Quick CV evaluation for the tiered system.
# This is intentionally lighter-weight than the full training script:
# - Uses the already-generated feature CSVs in models/
# - Performs K-fold CV, re-training models per fold
# - Reports Tier1 accuracy, overall exact/within-1/within-2, and 9vs10 accuracy

suppressPackageStartupMessages({
  library(ranger)
})

args <- commandArgs(trailingOnly = TRUE)
K <- if (length(args) >= 1) as.integer(args[[1]]) else 5
if (is.na(K) || K < 2) stop("Usage: Rscript evaluate_tiered_cv.R [K>=2]", call. = FALSE)

models_dir <- "models"
advanced_csv <- if (file.exists(file.path(models_dir, "advanced_features_v3.csv"))) {
  file.path(models_dir, "advanced_features_v3.csv")
} else {
  file.path(models_dir, "advanced_features_v2.csv")
}
cnn_csv <- file.path(models_dir, "cnn_features_mobilenetv2.csv")

if (!file.exists(advanced_csv)) stop("Missing advanced features CSV (run extractor).", call. = FALSE)
if (!file.exists(cnn_csv)) stop("Missing models/cnn_features_mobilenetv2.csv (run extractor).", call. = FALSE)

cat("Loading features...\n")
adv <- read.csv(advanced_csv, check.names = FALSE)
cnn <- read.csv(cnn_csv, check.names = FALSE)

df <- merge(adv, cnn, by = "path", suffixes = c("_adv", "_cnn"), all = FALSE)
df$label <- factor(df$label_adv)
df$label_adv <- NULL
if ("label_cnn" %in% colnames(df)) df$label_cnn <- NULL

grade_to_num <- function(lbl) as.numeric(gsub("PSA_", "", as.character(lbl)))
map_tier <- function(n) {
  if (n <= 4) return("Low_1_4")
  if (n <= 7) return("Mid_5_7")
  "High_8_10"
}

df$grade_num <- grade_to_num(df$label)
df$tier <- factor(vapply(df$grade_num, map_tier, character(1)), levels = c("Low_1_4", "Mid_5_7", "High_8_10"))

feature_cols <- setdiff(colnames(df), c("label", "grade_num", "tier", "path"))
X <- df[, feature_cols, drop = FALSE]

smote_balance <- function(X_df, y_factor, k = 5, seed = 42) {
  set.seed(seed)
  y <- as.factor(y_factor)
  # SMOTE-style interpolation within class to match majority size
  X_mat <- data.matrix(X_df)
  colnames(X_mat) <- colnames(X_df)
  counts <- table(y)
  target_n <- max(counts)

  X_out <- X_mat
  y_out <- as.character(y)

  for (cls in names(counts)) {
    idx <- which(y == cls)
    n_cls <- length(idx)
    need <- target_n - n_cls
    if (need <= 0 || n_cls == 0) next

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

select_top_features <- function(X_df, y_factor, top_n = 365,
                                critical_prefixes = c("centering_", "corner_", "hires_corner_", "corner_circularity_", "log_")) {
  y_factor <- as.factor(y_factor)
  rf <- ranger(
    dependent.variable.name = "y",
    data = cbind(X_df, y = y_factor),
    num.trees = 120,
    importance = "impurity",
    probability = TRUE,
    classification = TRUE
  )
  imp <- sort(rf$variable.importance, decreasing = TRUE)
  imp_names <- names(imp)

  critical <- unique(unlist(lapply(critical_prefixes, function(p) grep(paste0("^", p), colnames(X_df), value = TRUE))))
  critical <- intersect(critical, colnames(X_df))

  selected <- character(0)
  if (length(critical) > 0) selected <- c(selected, critical)
  fill <- setdiff(imp_names, selected)
  if (length(selected) < top_n) selected <- c(selected, head(fill, top_n - length(selected)))
  selected <- unique(selected)
  selected <- selected[seq_len(min(length(selected), top_n))]
  selected
}

predict_probs <- function(model, row_df) {
  p <- predict(model, data = row_df, type = "response")$predictions
  probs <- as.numeric(p[1, ])
  names(probs) <- colnames(p)
  probs
}

as_full_prob <- function(probs, class_levels) {
  out <- rep(0, length(class_levels))
  names(out) <- class_levels
  common <- intersect(names(probs), class_levels)
  out[common] <- probs[common]
  out
}

consensus_probs <- function(tier_probs, probs_low, probs_mid, probs_high, probs_9v10, class_levels, tier_weight_gamma = 2) {
  # tier_probs: named Low_1_4/Mid_5_7/High_8_10
  tier_probs <- tier_probs^tier_weight_gamma
  tier_probs <- tier_probs / (sum(tier_probs) + 1e-12)

  full <- rep(0, length(class_levels))
  names(full) <- class_levels

  full <- full + tier_probs["Low_1_4"] * as_full_prob(probs_low, class_levels)
  full <- full + tier_probs["Mid_5_7"] * as_full_prob(probs_mid, class_levels)
  full <- full + tier_probs["High_8_10"] * as_full_prob(probs_high, class_levels)

  # Apply 9v10 redistribution
  if (all(c("PSA_9", "PSA_10") %in% class_levels) && all(c("PSA_9", "PSA_10") %in% names(probs_9v10))) {
    mass <- full["PSA_9"] + full["PSA_10"]
    if (mass > 0) {
      bsum <- probs_9v10["PSA_9"] + probs_9v10["PSA_10"]
      if (bsum > 0) {
        full["PSA_9"] <- mass * (probs_9v10["PSA_9"] / bsum)
        full["PSA_10"] <- mass * (probs_9v10["PSA_10"] / bsum)
      }
    }
  }

  full <- full / (sum(full) + 1e-12)
  full
}

set.seed(42)
n <- nrow(df)
folds <- sample(rep(1:K, length.out = n))

tier1_acc <- numeric(K)
exact <- numeric(K)
within1 <- numeric(K)
within2 <- numeric(K)
acc_9v10 <- numeric(K)

for (k in 1:K) {
  cat("\n=== Fold ", k, "/", K, " ===\n", sep = "")
  train_idx <- which(folds != k)
  test_idx <- which(folds == k)

  X_train <- X[train_idx, , drop = FALSE]
  y_train <- df$label[train_idx]
  tier_train <- df$tier[train_idx]

  X_test <- X[test_idx, , drop = FALSE]
  y_test <- df$label[test_idx]
  tier_test <- df$tier[test_idx]

  # Feature selection on training fold only
  selected <- select_top_features(X_train, y_train, top_n = 365)

  # Tier 1
  bal1 <- smote_balance(X_train[, selected, drop = FALSE], tier_train)
  m_tier1 <- ranger(
    dependent.variable.name = "y",
    data = cbind(bal1$X, y = bal1$y),
    num.trees = 150,
    probability = TRUE,
    classification = TRUE
  )

  # Specialists
  train_specialist <- function(idx) {
    Xs <- X_train[idx, selected, drop = FALSE]
    ys <- y_train[idx]
    bal <- smote_balance(Xs, ys)
    ranger(
      dependent.variable.name = "y",
      data = cbind(bal$X, y = bal$y),
      num.trees = 200,
      probability = TRUE,
      classification = TRUE
    )
  }

  idx_low <- which(grade_to_num(y_train) <= 4)
  idx_mid <- which(grade_to_num(y_train) >= 5 & grade_to_num(y_train) <= 7)
  idx_high <- which(grade_to_num(y_train) >= 8 & grade_to_num(y_train) <= 10)

  m_low <- train_specialist(idx_low)
  m_mid <- train_specialist(idx_mid)
  m_high <- train_specialist(idx_high)

  # 9 vs 10
  idx_9v10 <- which(grade_to_num(y_train) %in% c(9, 10))
  y_bin <- factor(ifelse(grade_to_num(y_train[idx_9v10]) == 10, "PSA_10", "PSA_9"), levels = c("PSA_9", "PSA_10"))
  bal_bin <- smote_balance(X_train[idx_9v10, selected, drop = FALSE], y_bin)
  m_9v10 <- ranger(
    dependent.variable.name = "y",
    data = cbind(bal_bin$X, y = bal_bin$y),
    num.trees = 200,
    probability = TRUE,
    classification = TRUE
  )

  # Predict test fold (consensus soft voting)
  y_pred <- character(length(test_idx))
  tier_pred <- character(length(test_idx))

  class_levels <- levels(df$label)

  for (i in seq_along(test_idx)) {
    row <- X_test[i, selected, drop = FALSE]
    tp <- predict_probs(m_tier1, row)
    tname <- names(tp)[which.max(tp)]
    tier_pred[i] <- tname

    gp_low <- predict_probs(m_low, row)
    gp_mid <- predict_probs(m_mid, row)
    gp_high <- predict_probs(m_high, row)
    bp <- predict_probs(m_9v10, row)

    # Ensure we have all three tier names present (missing ones get 0)
    tier_full <- c(Low_1_4 = 0, Mid_5_7 = 0, High_8_10 = 0)
    tier_full[names(tp)] <- tp

    full <- consensus_probs(tier_full, gp_low, gp_mid, gp_high, bp, class_levels, tier_weight_gamma = 2)
    gname <- names(full)[which.max(full)]

    y_pred[i] <- gname
  }

  tier1_acc[k] <- mean(tier_pred == as.character(tier_test))

  y_test_num <- grade_to_num(y_test)
  y_pred_num <- grade_to_num(y_pred)

  exact[k] <- mean(y_pred == as.character(y_test))
  within1[k] <- mean(abs(y_pred_num - y_test_num) <= 1)
  within2[k] <- mean(abs(y_pred_num - y_test_num) <= 2)

  idx_test_9v10 <- which(y_test_num %in% c(9, 10))
  if (length(idx_test_9v10) > 0) {
    acc_9v10[k] <- mean(y_pred[idx_test_9v10] == as.character(y_test[idx_test_9v10]))
  } else {
    acc_9v10[k] <- NA_real_
  }

  cat(sprintf("Tier1: %.1f%% | Exact: %.1f%% | Within1: %.1f%% | Within2: %.1f%% | 9v10: %s\n",
              tier1_acc[k] * 100,
              exact[k] * 100,
              within1[k] * 100,
              within2[k] * 100,
              ifelse(is.na(acc_9v10[k]), "NA", sprintf("%.1f%%", acc_9v10[k] * 100))))
}

cat("\n============================\n")
cat("CV SUMMARY\n")
cat("============================\n")
cat(sprintf("Tier1 accuracy: %.1f%% (SD %.1f)\n", mean(tier1_acc) * 100, sd(tier1_acc) * 100))
cat(sprintf("Exact match:    %.1f%% (SD %.1f)\n", mean(exact) * 100, sd(exact) * 100))
cat(sprintf("Within 1 grade: %.1f%% (SD %.1f)\n", mean(within1) * 100, sd(within1) * 100))
cat(sprintf("Within 2 grades:%.1f%% (SD %.1f)\n", mean(within2) * 100, sd(within2) * 100))
cat(sprintf("9 vs 10 acc:    %.1f%% (SD %.1f)\n",
            mean(acc_9v10, na.rm = TRUE) * 100, sd(acc_9v10, na.rm = TRUE) * 100))

