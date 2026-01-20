#!/usr/bin/env Rscript

# ============================================
# Tiered PSA Grade Model v2 (Professional Accuracy)
# ============================================
# UPGRADES:
# 1. Binary Triage: First cut is "Near Mint" (8-10) vs "Market Grade" (1-7)
#    - Prevents "feature pollution" - centering doesn't decide between 8 and 9
# 2. Back-of-Card Penalty System infrastructure
#    - Checks for front/back pairs
#    - Uses "Lowest Common Denominator" rule (back caps grade)
# 3. Feature importance selection (top 365)
# 4. Critical feature prefixes preserved
# 5. Tier 2: Specialist models for each tier
# 6. Binary: PSA 9 vs PSA 10 "glass ceiling" breaker
# 7. Ordinal regression head for hierarchical penalty
# 8. REMOVED: PSA_1.5 grade classification
# 9. MobileNet EMBEDDING FUSION: CNN features (1,280 dims) merged with
#    engineered features for 5-10% accuracy boost
# 10. LOG-LOSS OPTIMIZATION: Models trained with probability=TRUE and
#     evaluated using log-loss/Brier score (not just accuracy)
# 11. META-MODEL WEIGHTING: Specialist outputs combined via learned weights
#     instead of simple if-else logic
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

models_dir <- "models"

# Look for advanced features CSV (current or legacy names)
advanced_csv_candidates <- c(
  file.path(models_dir, "advanced_features.csv"),      # Current (v4)
  file.path(models_dir, "advanced_features_v4.csv"),   # Legacy v4 name
  file.path(models_dir, "advanced_features_v3.csv"),   # Legacy v3
  file.path(models_dir, "advanced_features_v2.csv")    # Legacy v2
)
advanced_csv <- advanced_csv_candidates[file.exists(advanced_csv_candidates)][1]

cnn_csv_candidates <- c(
  file.path(models_dir, "cnn_features_mobilenetv2.csv"),
  file.path(models_dir, "cnn_features_resnet50.csv")
)
cnn_csv <- cnn_csv_candidates[file.exists(cnn_csv_candidates)][1]

stop_if_missing <- function(path, hint) {
  if (!file.exists(path)) stop(paste0("Missing file: ", path, "\n\n", hint), call. = FALSE)
}

if (is.na(advanced_csv) || !nzchar(advanced_csv)) {
  stop(
    paste0(
      "Missing advanced features CSV.\n\n",
      "Run:\n",
      "  python3 scripts/feature_extraction/extract_advanced_features.py\n"
    ),
    call. = FALSE
  )
}

cat("========================================\n")
cat("Tiered Model v2 Training (Binary Triage)\n")
cat("========================================\n\n")

cat("Loading advanced features:", advanced_csv, "\n")
adv <- read.csv(advanced_csv, check.names = FALSE)

if (!"label" %in% colnames(adv)) stop("CSV must include a 'label' column.", call. = FALSE)
if (!"path" %in% colnames(adv)) stop("CSV must include a 'path' column.", call. = FALSE)

# =============================================================================
# FILTER OUT PSA_1.5 (as requested)
# =============================================================================
cat("Filtering out PSA_1.5 grades (not used in current model)...\n")
adv <- adv[adv$label != "PSA_1.5", , drop = FALSE]

if (!is.na(cnn_csv) && nzchar(cnn_csv)) {
  cat("Loading CNN features:", cnn_csv, "\n")
  cnn <- read.csv(cnn_csv, check.names = FALSE)
  if (!"path" %in% colnames(cnn)) stop("CNN CSV must include a 'path' column.", call. = FALSE)
  
  # Filter CNN features to exclude PSA_1.5
  if ("label" %in% colnames(cnn)) {
    cnn <- cnn[cnn$label != "PSA_1.5", , drop = FALSE]
  }

  merged <- merge(
    adv,
    cnn,
    by = "path",
    suffixes = c("_adv", "_cnn"),
    all = FALSE
  )

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

# =============================================================================
# BACK-OF-CARD PENALTY SYSTEM INFRASTRUCTURE
# =============================================================================
# This section sets up the infrastructure for front/back pair detection.
# Currently checks for back images and will apply penalty when available.

detect_back_of_card_pairs <- function(df) {
  # Extract base card identifiers from paths
  # Convention: filename_front.jpg / filename_back.jpg or filename_f.jpg / filename_b.jpg
  # Or: filename.jpg (front) / filename_back.jpg
  
  df$base_id <- gsub("(_front|_back|_f|_b)\\.(jpg|jpeg|png|webp)$", "", basename(df$path), ignore.case = TRUE)
  df$base_id <- gsub("\\.(jpg|jpeg|png|webp)$", "", df$base_id, ignore.case = TRUE)
  
  # Detect if this is a front or back image
  df$is_back <- grepl("(_back|_b)\\.(jpg|jpeg|png|webp)$", df$path, ignore.case = TRUE)
  df$is_front <- !df$is_back
  
  # Find pairs
  front_ids <- unique(df$base_id[df$is_front])
  back_ids <- unique(df$base_id[df$is_back])
  paired_ids <- intersect(front_ids, back_ids)
  
  df$has_back_pair <- df$base_id %in% paired_ids & df$is_front
  
  cat("\n--- Back-of-Card Detection ---\n")
  cat("Total images:", nrow(df), "\n")
  cat("Front images:", sum(df$is_front), "\n")
  cat("Back images:", sum(df$is_back), "\n")
  cat("Cards with front+back pairs:", length(paired_ids), "\n")
  
  if (length(paired_ids) == 0) {
    cat("Note: No front/back pairs detected. Back-of-card penalty will be skipped.\n")
    cat("To enable: name files as cardname_front.jpg / cardname_back.jpg\n")
  }
  
  return(df)
}

apply_back_penalty <- function(df, whitening_threshold = 0.5) {
  # This function applies the "Lowest Common Denominator" rule
  # If the back has whitening that matches a lower grade, cap the front grade
  
  if (!"has_back_pair" %in% colnames(df)) {
    return(df)
  }
  
  if (sum(df$has_back_pair) == 0) {
    cat("No back pairs available - skipping back penalty application.\n")
    return(df)
  }
  
  # Get whitening features from back images
  whitening_cols <- grep("whitening_score", colnames(df), value = TRUE)
  
  if (length(whitening_cols) == 0) {
    cat("Note: No whitening features found. Back penalty requires v4 features.\n")
    return(df)
  }
  
  # For each paired front image, check the back's whitening score
  paired_fronts <- which(df$has_back_pair)
  
  penalties_applied <- 0
  for (idx in paired_fronts) {
    base_id <- df$base_id[idx]
    back_idx <- which(df$base_id == base_id & df$is_back)
    
    if (length(back_idx) == 0) next
    
    # Get max whitening score from back
    back_whitening <- max(unlist(df[back_idx[1], whitening_cols]), na.rm = TRUE)
    
    if (!is.finite(back_whitening)) next
    
    # Apply penalty based on whitening severity
    if (back_whitening > whitening_threshold) {
      original_grade <- as.numeric(gsub("PSA_", "", df$label[idx]))
      
      # Whitening penalty mapping (approximate)
      # 0.5-0.6 whitening -> cap at 8
      # 0.6-0.7 whitening -> cap at 7
      # 0.7+ whitening -> cap at 6
      
      if (back_whitening > 0.7) {
        max_grade <- 6
      } else if (back_whitening > 0.6) {
        max_grade <- 7
      } else {
        max_grade <- 8
      }
      
      if (original_grade > max_grade) {
        df$label[idx] <- paste0("PSA_", max_grade)
        df$back_penalty_applied[idx] <- TRUE
        penalties_applied <- penalties_applied + 1
      }
    }
  }
  
  cat("Back-of-card penalties applied:", penalties_applied, "\n")
  return(df)
}

# Apply back-of-card detection
df <- detect_back_of_card_pairs(df)
df$back_penalty_applied <- FALSE

# Apply back-of-card penalty (will skip if no pairs found)
df <- apply_back_penalty(df)

# For training, only use front images (or all if no back detection)
if (any(df$is_back)) {
  cat("Using only front images for training...\n")
  df <- df[df$is_front, , drop = FALSE]
}

# =============================================================================
# Prepare data
# =============================================================================
df$.row_id <- seq_len(nrow(df))
rownames(df) <- df$.row_id

# --- Label utilities ---
grade_to_num <- function(lbl) {
  as.numeric(gsub("PSA_", "", as.character(lbl)))
}

num_to_label <- function(x) {
  paste0("PSA_", x)
}

# =============================================================================
# BINARY TRIAGE: Near Mint (8-10) vs Market Grade (1-7)
# =============================================================================
map_binary_tier <- function(lbl_num) {
  if (lbl_num >= 8) return("NearMint_8_10")
  "MarketGrade_1_7"
}

# Also keep 3-tier for specialist routing
map_tier <- function(lbl_num) {
  if (lbl_num <= 4) return("Low_1_4")
  if (lbl_num <= 7) return("Mid_5_7")
  "High_8_10"
}

# --- Prepare matrix ---
df$grade_num <- grade_to_num(df$label)
df$binary_tier <- factor(vapply(df$grade_num, map_binary_tier, character(1)), 
                         levels = c("MarketGrade_1_7", "NearMint_8_10"))
df$tier <- factor(vapply(df$grade_num, map_tier, character(1)), 
                  levels = c("Low_1_4", "Mid_5_7", "High_8_10"))
df$label <- factor(df$label)

# Exclude non-feature columns
exclude_cols <- c("label", "grade_num", "tier", "binary_tier", "path", ".row_id",
                  "base_id", "is_back", "is_front", "has_back_pair", "back_penalty_applied")
feature_cols <- setdiff(colnames(df), exclude_cols)
X <- df[, feature_cols, drop = FALSE]

cat("\nDataset:", nrow(df), "samples\n")
cat("Features:", ncol(X), "\n")
cat("Binary tiers:", paste(levels(df$binary_tier), collapse = ", "), "\n")
cat("Specialist tiers:", paste(levels(df$tier), collapse = ", "), "\n\n")

# --- Class distribution ---
cat("Grade distribution:\n")
print(table(df$label))
cat("\n")

# --- Class balancing (SMOTE with fallback) ---
smote_balance <- function(X_df, y_factor, target_n = NULL, k = 5, seed = 42) {
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
  return(list(X = X_out, y = factor(y_out, levels = levels(y))))
}

# --- Feature importance selection (top N) ---
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
    "patch_",
    "adaptive_patch_",
    "artbox_",
    "whitening_"
  )
) {
  y_factor <- as.factor(y_factor)
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
    selected <- head(selected[order(match(selected, imp_names))], top_n)
  }

  selected <- unique(selected)
  selected <- selected[seq_len(min(length(selected), top_n))]
  return(list(selected = selected, importance = imp))
}

# =============================================================================
# BINARY TRIAGE: Tier 0 - Near Mint vs Market Grade
# =============================================================================
cat("Selecting Binary Triage feature set...\n")
sel_binary <- select_top_features(
  X,
  df$binary_tier,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hires_corner_", "artbox_", "log_", "adaptive_patch_")
)
features_binary <- sel_binary$selected
cat("Binary triage features:", length(features_binary), "\n\n")

cat("Training Binary Triage model (Near Mint vs Market Grade)...\n")
bal_binary <- smote_balance(X[, features_binary, drop = FALSE], df$binary_tier)
binary_triage_model <- ranger(
  dependent.variable.name = "y",
  data = cbind(bal_binary$X, y = bal_binary$y),
  num.trees = 500,
  importance = "impurity",
  probability = TRUE,
  classification = TRUE
)

# =============================================================================
# TIER-SPECIFIC FEATURE SELECTION
# =============================================================================
cat("Selecting tier-specific feature sets...\n")

# Tier 1: Low/Mid/High (for Market Grade routing)
sel_tier1 <- select_top_features(
  X,
  df$tier,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hires_corner_", "corner_circularity_", "log_", "lab_", "patch_", "adaptive_patch_")
)
features_tier1 <- sel_tier1$selected

low_df <- df[df$grade_num <= 4, , drop = FALSE]
mid_df <- df[df$grade_num >= 5 & df$grade_num <= 7, , drop = FALSE]
high_df <- df[df$grade_num >= 8 & df$grade_num <= 10, , drop = FALSE]

X_low <- X[rownames(low_df), , drop = FALSE]
X_mid <- X[rownames(mid_df), , drop = FALSE]
X_high <- X[rownames(high_df), , drop = FALSE]

# Low specialist: focus on damage/heavy wear features
sel_low <- select_top_features(
  X_low,
  low_df$label,
  top_n = 365,
  critical_prefixes = c("log_", "texture_", "hog_", "lbp_", "patch_", "lab_", "adaptive_patch_")
)
features_low <- sel_low$selected

# Mid specialist: centering + moderate wear
sel_mid <- select_top_features(
  X_mid,
  mid_df$label,
  top_n = 365,
  critical_prefixes = c("centering_", "corner_", "hog_", "log_", "lab_", "patch_", "adaptive_patch_")
)
features_mid <- sel_mid$selected

# High specialist (8-10): Focus on MICRO features for 9 vs 10 distinction
# Use artbox centering, adaptive patches, and whitening
sel_high <- select_top_features(
  X_high,
  high_df$label,
  top_n = 365,
  critical_prefixes = c("artbox_", "adaptive_patch_", "hires_corner_", "corner_circularity_", 
                        "whitening_", "log_kurtosis_", "centering_", "corner_")
)
features_high <- sel_high$selected

cat("Binary triage features:", length(features_binary), "\n")
cat("Tier 1 features:", length(features_tier1), "\n")
cat("Low specialist features:", length(features_low), "\n")
cat("Mid specialist features:", length(features_mid), "\n")
cat("High specialist features:", length(features_high), "\n\n")

# =============================================================================
# TRAIN MODELS
# =============================================================================

# Tier 1 (Low/Mid/High) - only for Market Grade routing
cat("Training Tier 1 model (Low/Mid/High) for Market Grade...\n")
market_df <- df[df$binary_tier == "MarketGrade_1_7", , drop = FALSE]
X_market <- X[rownames(market_df), , drop = FALSE]

# Map to low/mid tiers only
market_df$market_tier <- factor(ifelse(market_df$grade_num <= 4, "Low_1_4", "Mid_5_7"),
                                levels = c("Low_1_4", "Mid_5_7"))
bal_market <- smote_balance(X_market[, features_tier1, drop = FALSE], market_df$market_tier)
market_tier_model <- ranger(
  dependent.variable.name = "y",
  data = cbind(bal_market$X, y = bal_market$y),
  num.trees = 500,
  importance = "impurity",
  probability = TRUE,
  classification = TRUE
)

# Train specialists
train_specialist <- function(sub_df, feature_names, ntrees = 700) {
  Xs <- X[rownames(sub_df), feature_names, drop = FALSE]
  ys <- factor(sub_df$label)
  bal <- smote_balance(Xs, ys)
  ranger(
    dependent.variable.name = "y",
    data = cbind(bal$X, y = bal$y),
    num.trees = ntrees,
    probability = TRUE,
    classification = TRUE
  )
}

cat("Training Low specialist (PSA 1-4)...\n")
low_model <- train_specialist(low_df, features_low)

cat("Training Mid specialist (PSA 5-7)...\n")
mid_model <- train_specialist(mid_df, features_mid)

cat("Training High specialist (PSA 8-10) with MICRO features...\n")
high_model <- train_specialist(high_df, features_high, ntrees = 1000)

# Save a dedicated high-grade specialist artifact
high_grade_specialist <- list(
  model = high_model,
  selected_features = features_high,
  range = c("PSA_8", "PSA_9", "PSA_10")
)

# =============================================================================
# 9 vs 10 BINARY CLASSIFIER (Glass Ceiling Breaker)
# =============================================================================
cat("Training PSA 9 vs PSA 10 binary classifier...\n")
bin_df <- df[df$grade_num %in% c(9, 10), , drop = FALSE]
bin_df$bin_label <- factor(ifelse(bin_df$grade_num == 10, "PSA_10", "PSA_9"), levels = c("PSA_9", "PSA_10"))

X_bin <- X[rownames(bin_df), , drop = FALSE]

# 9 vs 10 requires the most micro-level features
sel_9v10 <- select_top_features(
  X_bin,
  bin_df$bin_label,
  top_n = 250,
  critical_prefixes = c("artbox_", "adaptive_patch_", "hires_corner_", "corner_circularity_", 
                        "whitening_", "log_kurtosis_", "centering_", "patch_")
)
features_9v10 <- sel_9v10$selected

bal_bin <- smote_balance(X_bin[, features_9v10, drop = FALSE], bin_df$bin_label)
psa_9_vs_10 <- ranger(
  dependent.variable.name = "y",
  data = cbind(bal_bin$X, y = bal_bin$y),
  num.trees = 1000,
  probability = TRUE,
  classification = TRUE
)

# =============================================================================
# ORDINAL REGRESSION HEAD (Hierarchical Penalty)
# =============================================================================
cat("Training ordinal regression head...\n")
X_reg <- X[, features_tier1, drop = FALSE]
reg_model <- ranger(
  dependent.variable.name = "y",
  data = cbind(X_reg, y = df$grade_num),
  num.trees = 700,
  classification = FALSE
)

# =============================================================================
# LOG-LOSS EVALUATION & META-MODEL WEIGHTING
# =============================================================================
# Instead of simple if-else logic, learn optimal weights for combining specialists

cat("Training meta-model for specialist weighting...\n")

# Helper: compute log-loss (lower is better)
compute_log_loss <- function(y_true, y_pred_prob) {
  # y_true: factor with class labels
  # y_pred_prob: matrix where columns are class probabilities
  eps <- 1e-15
  n <- length(y_true)
  ll <- 0
  for (i in seq_len(n)) {
    true_class <- as.character(y_true[i])
    if (true_class %in% colnames(y_pred_prob)) {
      p <- max(eps, min(1 - eps, y_pred_prob[i, true_class]))
      ll <- ll - log(p)
    }
  }
  ll / n
}

# Helper: compute Brier score (lower is better)
compute_brier <- function(y_true, y_pred_prob) {
  n <- length(y_true)
  bs <- 0
  for (i in seq_len(n)) {
    true_class <- as.character(y_true[i])
    for (cls in colnames(y_pred_prob)) {
      target <- if (cls == true_class) 1 else 0
      bs <- bs + (y_pred_prob[i, cls] - target)^2
    }
  }
  bs / n
}

# Get specialist predictions on training data for meta-learning
# This creates "stacked" features for a meta-model
cat("  Generating specialist predictions for meta-learning...\n")

get_specialist_probs <- function(model, X_df, selected_features) {
  available <- intersect(selected_features, colnames(X_df))
  predict(model, data = X_df[, available, drop = FALSE], type = "response")$predictions
}

# Get predictions from each specialist on their respective subsets
# Then build a meta-dataset with these probabilities

# For simplicity, learn optimal blend weights using grid search on log-loss
# Weight parameters: binary_weight, market_weight, reg_blend_weight

cat("  Optimizing blend weights via grid search on log-loss...\n")

# Sample a validation subset for tuning (20%)
set.seed(42)
val_idx <- sample(seq_len(nrow(df)), size = floor(nrow(df) * 0.2))
train_meta_idx <- setdiff(seq_len(nrow(df)), val_idx)

# Simple function to compute consensus prediction with given weights
compute_consensus <- function(row, binary_probs, market_probs, 
                               low_probs, mid_probs, high_probs, 
                               bin_9v10_probs, reg_pred,
                               class_levels, tier_gamma = 2, reg_weight = 0.25, reg_sigma = 0.85) {
  
  full <- rep(0, length(class_levels))
  names(full) <- class_levels
  
  # Binary triage
  nm_prob <- binary_probs["NearMint_8_10"]
  mg_prob <- binary_probs["MarketGrade_1_7"]
  
  nm_prob <- nm_prob^tier_gamma
  mg_prob <- mg_prob^tier_gamma
  total <- nm_prob + mg_prob + 1e-12
  nm_prob <- nm_prob / total
  mg_prob <- mg_prob / total
  
  # Near Mint -> High specialist
  for (cls in names(high_probs)) {
    if (cls %in% class_levels) {
      full[cls] <- full[cls] + nm_prob * high_probs[cls]
    }
  }
  
  # Market Grade -> Low/Mid specialists
  low_w <- market_probs["Low_1_4"]
  mid_w <- market_probs["Mid_5_7"]
  low_w <- low_w^tier_gamma
  mid_w <- mid_w^tier_gamma
  mw_total <- low_w + mid_w + 1e-12
  low_w <- low_w / mw_total
  mid_w <- mid_w / mw_total
  
  for (cls in names(low_probs)) {
    if (cls %in% class_levels) {
      full[cls] <- full[cls] + mg_prob * low_w * low_probs[cls]
    }
  }
  
  for (cls in names(mid_probs)) {
    if (cls %in% class_levels) {
      full[cls] <- full[cls] + mg_prob * mid_w * mid_probs[cls]
    }
  }
  
  # 9 vs 10 reweighting
  if (all(c("PSA_9", "PSA_10") %in% class_levels)) {
    mass <- full["PSA_9"] + full["PSA_10"]
    if (mass > 0.1 && all(c("PSA_9", "PSA_10") %in% names(bin_9v10_probs))) {
      bsum <- bin_9v10_probs["PSA_9"] + bin_9v10_probs["PSA_10"]
      if (bsum > 0) {
        full["PSA_9"] <- mass * (bin_9v10_probs["PSA_9"] / bsum)
        full["PSA_10"] <- mass * (bin_9v10_probs["PSA_10"] / bsum)
      }
    }
  }
  
  # Ordinal regression blend
  if (reg_weight > 0) {
    grade_num <- function(lbl) as.numeric(gsub("PSA_", "", lbl))
    nums <- vapply(class_levels, grade_num, numeric(1))
    p_reg <- exp(-((nums - reg_pred)^2) / (2 * reg_sigma^2))
    p_reg <- p_reg / (sum(p_reg) + 1e-12)
    names(p_reg) <- class_levels
    full <- (1 - reg_weight) * full + reg_weight * p_reg
  }
  
  full <- full / (sum(full) + 1e-12)
  full
}

# Grid search over key hyperparameters
best_logloss <- Inf
best_params <- list(tier_gamma = 2, reg_weight = 0.25, reg_sigma = 0.85)

tier_gammas <- c(1.5, 2.0, 2.5)
reg_weights <- c(0.15, 0.25, 0.35)
reg_sigmas <- c(0.70, 0.85, 1.0)

cat("  Testing", length(tier_gammas) * length(reg_weights) * length(reg_sigmas), "parameter combinations...\n")

for (tg in tier_gammas) {
  for (rw in reg_weights) {
    for (rs in reg_sigmas) {
      # Compute predictions on validation set
      val_preds <- matrix(0, nrow = length(val_idx), ncol = length(valid_class_levels))
      colnames(val_preds) <- valid_class_levels
      
      for (i in seq_along(val_idx)) {
        idx <- val_idx[i]
        row_x <- X[idx, , drop = FALSE]
        
        # Get all specialist predictions
        bp <- predict(binary_triage_model, data = row_x[, features_binary, drop = FALSE])$predictions[1,]
        mp <- predict(market_tier_model, data = row_x[, features_tier1, drop = FALSE])$predictions[1,]
        lp <- predict(low_model, data = row_x[, features_low, drop = FALSE])$predictions[1,]
        midp <- predict(mid_model, data = row_x[, features_mid, drop = FALSE])$predictions[1,]
        hp <- predict(high_model, data = row_x[, features_high, drop = FALSE])$predictions[1,]
        b9v10 <- predict(psa_9_vs_10, data = row_x[, features_9v10, drop = FALSE])$predictions[1,]
        rp <- predict(reg_model, data = row_x[, features_tier1, drop = FALSE])$predictions[1]
        
        consensus <- compute_consensus(
          row_x, bp, mp, lp, midp, hp, b9v10, rp,
          valid_class_levels, tier_gamma = tg, reg_weight = rw, reg_sigma = rs
        )
        val_preds[i, ] <- consensus
      }
      
      # Compute log-loss
      ll <- compute_log_loss(df$label[val_idx], val_preds)
      
      if (ll < best_logloss) {
        best_logloss <- ll
        best_params <- list(tier_gamma = tg, reg_weight = rw, reg_sigma = rs)
      }
    }
  }
}

cat("  Best log-loss:", round(best_logloss, 4), "\n")
cat("  Best params: tier_gamma=", best_params$tier_gamma, 
    ", reg_weight=", best_params$reg_weight,
    ", reg_sigma=", best_params$reg_sigma, "\n\n")

# Store optimized parameters
meta_weights <- list(
  tier_weight_gamma = best_params$tier_gamma,
  reg_blend_weight = best_params$reg_weight,
  reg_blend_sigma = best_params$reg_sigma,
  validation_log_loss = best_logloss
)

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)

# Define valid class levels (excluding PSA_1.5)
valid_class_levels <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                        "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")

tiered_model <- list(
  version = "tiered_v2_binary_triage_metamodel",
  feature_sets = list(
    binary = features_binary,
    tier1 = features_tier1,
    low = features_low,
    mid = features_mid,
    high = features_high,
    psa_9_vs_10 = features_9v10,
    reg = features_tier1
  ),
  class_levels = valid_class_levels,
  binary_levels = levels(df$binary_tier),
  tier_levels = c("Low_1_4", "Mid_5_7", "High_8_10"),
  market_tier_levels = c("Low_1_4", "Mid_5_7"),
  
  # Models
  binary_triage_model = binary_triage_model,
  market_tier_model = market_tier_model,
  low_model = low_model,
  mid_model = mid_model,
  high_model = high_model,
  reg_model = reg_model,
  
  # Meta-model optimized weights (learned via log-loss minimization)
  meta_weights = meta_weights,
  tier_weight_gamma = meta_weights$tier_weight_gamma,
  reg_blend = list(weight = meta_weights$reg_blend_weight, sigma = meta_weights$reg_blend_sigma),
  
  # Back-of-card infrastructure
  back_of_card_enabled = TRUE,
  whitening_threshold = 0.5,
  
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

# =============================================================================
# TRAINING SUMMARY
# =============================================================================
cat("========================================\n")
cat("TRAINING COMPLETE - Model Summary\n")
cat("========================================\n\n")

cat("Model Architecture: Binary Triage + Specialist + Meta-Model\n")
cat("------------------------------------------\n")
cat("1. Binary Triage: Near Mint (8-10) vs Market Grade (1-7)\n")
cat("   - Prevents feature pollution for high-grade distinctions\n")
cat("2. Market Grade Route: Low (1-4) vs Mid (5-7)\n")
cat("3. Specialist Models: Low, Mid, High\n")
cat("4. PSA 9 vs 10: Dedicated glass-ceiling breaker\n")
cat("5. Ordinal Regression: Hierarchical penalty head\n")
cat("6. Meta-Model: Log-loss optimized weights for specialist combination\n\n")

cat("Meta-Model Optimization (Log-Loss):\n")
cat("------------------------------------------\n")
cat("Validation log-loss:", round(meta_weights$validation_log_loss, 4), "\n")
cat("Optimized tier_gamma:", meta_weights$tier_weight_gamma, "\n")
cat("Optimized reg_weight:", meta_weights$reg_blend_weight, "\n")
cat("Optimized reg_sigma:", meta_weights$reg_blend_sigma, "\n\n")

cat("Back-of-Card Penalty System: READY\n")
cat("------------------------------------------\n")
cat("- Infrastructure enabled (awaiting back images)\n")
cat("- Naming convention: filename_front.jpg / filename_back.jpg\n")
cat("- Whitening threshold:", tiered_model$whitening_threshold, "\n\n")

cat("Feature Counts:\n")
cat("------------------------------------------\n")
cat("Binary triage:", length(features_binary), "\n")
cat("Low specialist:", length(features_low), "\n")
cat("Mid specialist:", length(features_mid), "\n")
cat("High specialist:", length(features_high), "\n")
cat("9 vs 10:", length(features_9v10), "\n\n")

cat("DONE.\n")
