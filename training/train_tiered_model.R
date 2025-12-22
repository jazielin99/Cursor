#!/usr/bin/env Rscript

# ============================================
# Tiered PSA Grade Model (No PCA Trap)
# - Feature importance selection (top 365)
# - Critical feature prefixes preserved
# - Tier 1: Low/Mid/High
# - Tier 2: Specialist models (Low, Mid, High 8/9/10)
# - Binary: PSA 9 vs PSA 10 "glass ceiling" breaker
# - Class balancing: SMOTE (with safe fallback)
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

models_dir <- "models"
# Prefer v3 if present; fallback to v2
advanced_csv_candidates <- c(
  file.path(models_dir, "advanced_features_v3.csv"),
  file.path(models_dir, "advanced_features_v2.csv")
)
advanced_csv <- advanced_csv_candidates[file.exists(advanced_csv_candidates)][1]

cnn_csv_candidates <- c(
  file.path(models_dir, "cnn_features_mobilenetv2.csv"),
  file.path(models_dir, "cnn_features_resnet50.csv") # legacy name
)
cnn_csv <- cnn_csv_candidates[file.exists(cnn_csv_candidates)][1]

stop_if_missing <- function(path, hint) {
  if (!file.exists(path)) stop(paste0("Missing file: ", path, "\n\n", hint), call. = FALSE)
}

if (is.na(advanced_csv) || !nzchar(advanced_csv)) {
  stop(
    paste0(
      "Missing advanced features CSV.\n\n",
      "Run one of:\n",
      "  python3 extract_advanced_features_v3.py\n",
      "  python3 extract_advanced_features_v2.py\n"
    ),
    call. = FALSE
  )
}

cat("========================================\n")
cat("Tiered Model Training (Feature Importance)\n")
cat("========================================\n\n")

cat("Loading advanced features:", advanced_csv, "\n")
adv <- read.csv(advanced_csv, check.names = FALSE)

if (!"label" %in% colnames(adv)) stop("advanced_features_v2.csv must include a 'label' column.", call. = FALSE)
if (!"path" %in% colnames(adv)) stop("advanced_features_v2.csv must include a 'path' column.", call. = FALSE)

if (!is.na(cnn_csv) && nzchar(cnn_csv)) {
  cat("Loading CNN features:", cnn_csv, "\n")
  cnn <- read.csv(cnn_csv, check.names = FALSE)
  if (!"path" %in% colnames(cnn)) stop("CNN CSV must include a 'path' column.", call. = FALSE)

  # Merge by path (inner join)
  merged <- merge(
    adv,
    cnn,
    by = "path",
    suffixes = c("_adv", "_cnn"),
    all = FALSE
  )

  # Prefer label from advanced features (label_adv)
  label_col <- if ("label_adv" %in% colnames(merged)) "label_adv" else if ("label" %in% colnames(merged)) "label" else NA
  if (is.na(label_col)) stop("Could not locate label after merging advanced + CNN features.", call. = FALSE)
  merged$label <- merged[[label_col]]
  merged$label_adv <- NULL
  if ("label_cnn" %in% colnames(merged)) merged$label_cnn <- NULL

  df <- merged
} else {
  cat("CNN features not found; training will use advanced features only.\n")
  df <- adv
}

# Ensure rownames are stable for X slicing
df$.row_id <- seq_len(nrow(df))
rownames(df) <- df$.row_id

# --- Label utilities ---
grade_to_num <- function(lbl) {
  as.numeric(gsub("PSA_", "", as.character(lbl)))
}

num_to_label <- function(x) {
  paste0("PSA_", x)
}

map_tier <- function(lbl_num) {
  if (lbl_num <= 4) return("Low_1_4")
  if (lbl_num <= 7) return("Mid_5_7")
  "High_8_10"
}

# --- Prepare matrix ---
df$grade_num <- grade_to_num(df$label)
df$tier <- factor(vapply(df$grade_num, map_tier, character(1)), levels = c("Low_1_4", "Mid_5_7", "High_8_10"))
df$label <- factor(df$label)

feature_cols <- setdiff(colnames(df), c("label", "grade_num", "tier", "path", ".row_id"))
X <- df[, feature_cols, drop = FALSE]

cat("\nDataset:", nrow(df), "samples\n")
cat("Features:", ncol(X), "\n")
cat("Tiers:", paste(levels(df$tier), collapse = ", "), "\n\n")

# --- Class balancing (SMOTE with fallback) ---
smote_balance <- function(X_df, y_factor, target_n = NULL, k = 5, seed = 42) {
  set.seed(seed)
  y <- as.factor(y_factor)
  counts <- table(y)
  if (is.null(target_n)) target_n <- max(counts)

  # SMOTE-style interpolation within class (no KNN dependency; robust and deterministic).
  # This produces synthetic points by linear interpolation between two samples
  # from the same minority class until each class reaches target_n.

  X_mat <- as.matrix(X_df)
  colnames(X_mat) <- colnames(X_df)
  X_out <- X_mat
  # IMPORTANT: avoid factor -> integer coercion when concatenating
  y_out <- as.character(y)

  for (cls in names(counts)) {
    idx <- which(y == cls)
    n_cls <- length(idx)
    if (n_cls == 0) next

    need <- target_n - n_cls
    if (need <= 0) next

    # interpolate between random pairs from the same class
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
  return(list(X = X_out, y = factor(y_out, levels = levels(y))))
}

# --- Feature importance selection (top 365) ---
select_top_features <- function(
  X_df,
  y_factor,
  top_n = 365,
  critical_prefixes = c(
    "centering_",
    "corner_",
    "hires_corner_",
    "corner_circularity_",
    "log_",
    "lab_",
    "patch_"
  )
) {
  y_factor <- as.factor(y_factor)
  # Train a RF for importance on the full dataset.
  # (If you want stricter validation, compute importance inside each CV fold.)
  rf <- ranger(
    dependent.variable.name = "y",
    data = cbind(X_df, y = y_factor),
    num.trees = 300,
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
  if (length(selected) < top_n) {
    selected <- c(selected, head(fill, top_n - length(selected)))
  } else if (length(selected) > top_n) {
    # Keep the most important criticals first if critical set exceeds top_n
    selected <- head(selected[order(match(selected, imp_names))], top_n)
  }

  selected <- unique(selected)
  selected <- selected[seq_len(min(length(selected), top_n))]
  return(list(selected = selected, importance = imp))
}

# --- Tier-specific feature selection (dynamic hierarchy) ---
cat("Selecting tier-specific feature sets...\n")

# Tier 1 selection (tier labels)
sel_tier1 <- select_top_features(
  X,
  df$tier,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hires_corner_", "corner_circularity_", "log_", "lab_", "patch_")
)
features_tier1 <- sel_tier1$selected

low_df <- df[df$grade_num <= 4, , drop = FALSE]
mid_df <- df[df$grade_num >= 5 & df$grade_num <= 7, , drop = FALSE]
high_df <- df[df$grade_num >= 8 & df$grade_num <= 10, , drop = FALSE]

X_low <- X[rownames(low_df), , drop = FALSE]
X_mid <- X[rownames(mid_df), , drop = FALSE]
X_high <- X[rownames(high_df), , drop = FALSE]

sel_low <- select_top_features(
  X_low,
  low_df$label,
  top_n = 365,
  critical_prefixes = c("log_", "texture_", "hog_", "lbp_", "patch_", "lab_")
)
features_low <- sel_low$selected

sel_mid <- select_top_features(
  X_mid,
  mid_df$label,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hog_", "log_", "lab_", "patch_")
)
features_mid <- sel_mid$selected

sel_high <- select_top_features(
  X_high,
  high_df$label,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hires_corner_", "corner_circularity_", "patch_", "lab_", "log_kurtosis_", "log_")
)
features_high <- sel_high$selected

cat("Tier 1 features:", length(features_tier1), "\n")
cat("Low specialist features:", length(features_low), "\n")
cat("Mid specialist features:", length(features_mid), "\n")
cat("High specialist features:", length(features_high), "\n\n")

# --- Train Tier 1 (Low/Mid/High) ---
cat("Training Tier 1 model (Low/Mid/High) with class balancing...\n")
bal1 <- smote_balance(X[, features_tier1, drop = FALSE], df$tier)
tier1_model <- ranger(
  dependent.variable.name = "y",
  data = cbind(bal1$X, y = bal1$y),
  num.trees = 500,
  importance = "impurity",
  probability = TRUE,
  classification = TRUE
)

# --- Train Tier 2 specialists (exact grades per tier) ---
train_specialist <- function(sub_df, feature_names) {
  # sub_df is a slice of df; use rownames to index X
  Xs <- X[rownames(sub_df), feature_names, drop = FALSE]
  ys <- factor(sub_df$label)
  bal <- smote_balance(Xs, ys)
  ranger(
    dependent.variable.name = "y",
    data = cbind(bal$X, y = bal$y),
    num.trees = 700,
    probability = TRUE,
    classification = TRUE
  )
}

cat("Training Low specialist (PSA 1-4)...\n")
low_model <- train_specialist(low_df, features_low)

cat("Training Mid specialist (PSA 5-7)...\n")
mid_model <- train_specialist(mid_df, features_mid)

cat("Training High specialist (PSA 8-10)...\n")
high_model <- train_specialist(high_df, features_high)

# Save a dedicated high-grade specialist artifact (requested)
high_grade_specialist <- list(
  model = high_model,
  selected_features = features_high,
  range = c("PSA_8", "PSA_9", "PSA_10")
)

# --- 9 vs 10 binary classifier ---
cat("Training PSA 9 vs PSA 10 binary classifier...\n")
bin_df <- df[df$grade_num %in% c(9, 10), , drop = FALSE]
bin_df$bin_label <- factor(ifelse(bin_df$grade_num == 10, "PSA_10", "PSA_9"), levels = c("PSA_9", "PSA_10"))

X_bin <- X[rownames(bin_df), , drop = FALSE]
sel_9v10 <- select_top_features(
  X_bin,
  bin_df$bin_label,
  top_n = 250,
  critical_prefixes = c("centering_", "corner_", "hires_corner_", "corner_circularity_", "patch_", "lab_", "log_kurtosis_", "log_")
)
features_9v10 <- sel_9v10$selected

bal_bin <- smote_balance(X_bin[, features_9v10, drop = FALSE], bin_df$bin_label)
psa_9_vs_10 <- ranger(
  dependent.variable.name = "y",
  data = cbind(bal_bin$X, y = bal_bin$y),
  num.trees = 700,
  probability = TRUE,
  classification = TRUE
)

# --- Hierarchical penalty approximation: ordinal regression head (numeric grade) ---
cat("Training ordinal regression head (squared-error penalty for far misses)...\n")
X_reg <- X[, features_tier1, drop = FALSE]
reg_model <- ranger(
  dependent.variable.name = "y",
  data = cbind(X_reg, y = df$grade_num),
  num.trees = 700,
  classification = FALSE
)

# --- Save artifacts ---
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)

tiered_model <- list(
  version = "tiered_v2_expert",
  # feature sets used by each component
  feature_sets = list(
    tier1 = features_tier1,
    low = features_low,
    mid = features_mid,
    high = features_high,
    psa_9_vs_10 = features_9v10,
    reg = features_tier1
  ),
  class_levels = levels(df$label),
  tier_levels = levels(df$tier),
  tier1_model = tier1_model,
  low_model = low_model,
  mid_model = mid_model,
  high_model = high_model,
  reg_model = reg_model,
  reg_blend = list(weight = 0.25, sigma = 0.85),
  feature_sources = list(
    advanced = basename(advanced_csv),
    cnn = if (!is.na(cnn_csv) && nzchar(cnn_csv)) basename(cnn_csv) else NA
  )
)

saveRDS(tiered_model, file.path(models_dir, "tiered_model.rds"))
saveRDS(high_grade_specialist, file.path(models_dir, "high_grade_specialist.rds"))
saveRDS(list(model = psa_9_vs_10, selected_features = features_9v10), file.path(models_dir, "psa_9_vs_10.rds"))

cat("\nSaved models:\n")
cat("  models/tiered_model.rds\n")
cat("  models/high_grade_specialist.rds\n")
cat("  models/psa_9_vs_10.rds\n\n")

cat("DONE.\n")

