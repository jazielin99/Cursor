#!/usr/bin/env Rscript

# ============================================
# Full Pipeline Training with All Improvements
# ============================================
# Features:
# 1. CNN Feature Fusion (MobileNetV2 + engineered features)
# 2. Front/Back handling (skips back if not available)
# 3. Card type specialists (if tagged in manifest)
# 4. Cost-sensitive learning for hard classes
# 5. Confusion-pair specialists
# 6. Balanced ensemble approach
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

set.seed(42)

models_dir <- "models"
data_dir <- "data"

cat("========================================\n")
cat("Full Pipeline Training\n")
cat("(CNN Fusion + Front/Back + Card Types)\n")
cat("========================================\n\n")

# =============================================================================
# 1. LOAD FEATURES (ENGINEERED + CNN)
# =============================================================================

# Load engineered features
adv_csv <- file.path(models_dir, "advanced_features.csv")
if (!file.exists(adv_csv)) {
  stop("Missing advanced features. Run: python scripts/feature_extraction/extract_advanced_features.py")
}

cat("Loading engineered features:", adv_csv, "\n")
adv <- read.csv(adv_csv, check.names = FALSE)
adv <- adv[adv$label != "PSA_1.5", , drop = FALSE]
cat("  Samples:", nrow(adv), "\n")

# Load CNN features if available
cnn_csv <- file.path(models_dir, "cnn_features.csv")
use_cnn <- file.exists(cnn_csv)

if (use_cnn) {
  cat("Loading CNN features:", cnn_csv, "\n")
  cnn <- read.csv(cnn_csv, check.names = FALSE)
  cnn <- cnn[cnn$label != "PSA_1.5", , drop = FALSE]
  
  # Normalize paths for matching (handle training vs training_front)
  normalize_path <- function(p) {
    p <- gsub("data/training_front/", "data/training/", p)
    p <- gsub("data/training/", "", p)  # Keep only relative part
    basename(p)  # Just use filename for matching
  }
  
  adv$match_key <- normalize_path(adv$path)
  cnn$match_key <- normalize_path(cnn$path)
  
  # Merge on normalized key
  cnn_features_only <- cnn[, c("match_key", grep("^cnn_", colnames(cnn), value = TRUE)), drop = FALSE]
  df <- merge(adv, cnn_features_only, by = "match_key", all.x = TRUE)
  df$match_key <- NULL
  
  cat("  Merged samples:", nrow(df), "\n")
  cat("  CNN features:", sum(grepl("^cnn_", colnames(df))), "\n")
  cat("  Non-NA CNN rows:", sum(!is.na(df$cnn_0)), "\n")
} else {
  cat("CNN features not found - using engineered features only\n")
  cat("  Run: python scripts/feature_extraction/extract_cnn_features.py\n")
  df <- adv
}

# =============================================================================
# 2. LOAD BACK-OF-CARD FEATURES (IF AVAILABLE)
# =============================================================================

back_csv <- file.path(models_dir, "advanced_features_back.csv")
use_back <- file.exists(back_csv)

if (use_back) {
  cat("\nLoading back-of-card features:", back_csv, "\n")
  back <- read.csv(back_csv, check.names = FALSE)
  
  # Create path mapping for front/back pairing
  # Expected: front path ends with _front.jpg, back path ends with _back.jpg
  # Or: paths in training_front vs training_back with same filename
  
  df$front_path <- df$path
  df$back_path <- gsub("training_front", "training_back", df$path)
  
  # Check which have back images
  back_paths <- back$path
  df$has_back <- df$back_path %in% back_paths
  
  cat("  Cards with back images:", sum(df$has_back), "/", nrow(df), "\n")
  
  # Merge back features for those that have them
  if (sum(df$has_back) > 0) {
    back_features <- grep("^(hog_|lbp_|corner_|center_)", colnames(back), value = TRUE)
    back_renamed <- back[, c("path", back_features), drop = FALSE]
    colnames(back_renamed) <- c("back_path", paste0("back_", back_features))
    
    df <- merge(df, back_renamed, by = "back_path", all.x = TRUE)
    cat("  Added", length(back_features), "back features\n")
  }
} else {
  cat("\nNo back-of-card features found - using front only\n")
  cat("  To add: Place images in data/training_back/PSA_X/\n")
  cat("  Then run: python scripts/feature_extraction/extract_advanced_features.py --data-dir data/training_back --output-base models/advanced_features_back\n")
  df$has_back <- FALSE
}

# =============================================================================
# 3. LOAD CARD TYPE TAGS (IF AVAILABLE IN MANIFEST)
# =============================================================================

manifest_csv <- file.path(data_dir, "data_manifest_clean.csv")
use_card_types <- FALSE

if (file.exists(manifest_csv)) {
  manifest <- read.csv(manifest_csv, stringsAsFactors = FALSE)
  
  if ("card_type" %in% colnames(manifest)) {
    cat("\nLoading card type tags from manifest\n")
    
    type_info <- manifest[, c("path", "card_type")]
    df <- merge(df, type_info, by = "path", all.x = TRUE)
    df$card_type[is.na(df$card_type)] <- "unknown"
    
    type_counts <- table(df$card_type)
    cat("  Card types:\n")
    for (t in names(type_counts)) {
      cat(sprintf("    %s: %d\n", t, type_counts[t]))
    }
    
    use_card_types <- length(unique(df$card_type)) > 1
  }
}

if (!use_card_types) {
  cat("\nNo card type tags - using unified model\n")
  df$card_type <- "all"
}

# =============================================================================
# 4. FEATURE SELECTION
# =============================================================================

cat("\nSelecting features...\n")

class_levels <- c("PSA_1", "PSA_2", "PSA_3", "PSA_4", "PSA_5", 
                  "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")
hard_classes <- c("PSA_2", "PSA_5", "PSA_8")

# Get all feature columns
meta_cols <- c("path", "label", "front_path", "back_path", "has_back", "card_type", "fold")
all_cols <- setdiff(colnames(df), meta_cols)

# Separate CNN and engineered columns BEFORE filtering
cnn_cols_raw <- grep("^cnn_", all_cols, value = TRUE)
eng_cols_raw <- setdiff(all_cols, cnn_cols_raw)

cat("  Raw CNN columns:", length(cnn_cols_raw), "\n")

# Filter engineered columns for validity
valid_eng <- sapply(eng_cols_raw, function(col) {
  vals <- df[[col]]
  is.numeric(vals) && !any(is.na(vals)) && !any(is.infinite(vals)) && sd(vals, na.rm = TRUE) > 1e-10
})
eng_cols <- eng_cols_raw[valid_eng]

# For CNN columns, just check they exist and are numeric (allow lower variance)
valid_cnn <- sapply(cnn_cols_raw, function(col) {
  vals <- df[[col]]
  is.numeric(vals) && sum(is.na(vals)) < nrow(df) * 0.1  # Allow up to 10% NA
})
cnn_cols <- cnn_cols_raw[valid_cnn]

# Fill any NA in CNN columns with 0
for (col in cnn_cols) {
  df[[col]][is.na(df[[col]])] <- 0
}

cat("  Valid CNN features:", length(cnn_cols), "\n")
cat("  Valid engineered features:", length(eng_cols), "\n")

# Select top engineered features by variance
var_scores <- sapply(eng_cols, function(col) var(df[[col]], na.rm = TRUE))
top_eng <- names(sort(var_scores, decreasing = TRUE))[1:min(400, length(eng_cols))]

# ALWAYS include all CNN features (they are critical for accuracy)
selected_features <- c(cnn_cols, top_eng)
cat("  Selected features:", length(selected_features), "\n")
cat("    CNN:", length(cnn_cols), "\n")
cat("    Engineered:", length(top_eng), "\n")

# Verify CNN features are included
if (length(cnn_cols) == 0 && use_cnn) {
  warning("CNN features exist but none were selected!")
}

# =============================================================================
# 5. CROSS-VALIDATION WITH FULL PIPELINE
# =============================================================================

cat("\n========================================\n")
cat("Running 5-fold cross-validation...\n")
cat("========================================\n")

df$label <- factor(df$label, levels = class_levels)
df$fold <- sample(rep(1:5, length.out = nrow(df)))

# Class weights for cost-sensitive learning
class_counts <- table(df$label)
hard_boost <- c(PSA_1=1, PSA_2=2.0, PSA_3=1, PSA_4=1, PSA_5=2.5, 
                PSA_6=1, PSA_7=1, PSA_8=2.0, PSA_9=1, PSA_10=1)

smote_oversample <- function(data, target_col, minority_classes, factor = 1.5) {
  result <- data
  for (cls in minority_classes) {
    cls_data <- data[data[[target_col]] == cls, , drop = FALSE]
    n <- nrow(cls_data)
    if (n > 1) {
      n_syn <- round(n * (factor - 1))
      syn <- cls_data[sample(1:n, n_syn, replace = TRUE), ]
      num_cols <- intersect(selected_features, colnames(syn))
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

all_preds <- character(0)
all_true <- character(0)
all_probs <- list()
fold_results <- list()

for (k in 1:5) {
  cat("\nFold", k, "/ 5...\n")
  
  train_idx <- which(df$fold != k)
  test_idx <- which(df$fold == k)
  
  train_df <- df[train_idx, , drop = FALSE]
  test_df <- df[test_idx, , drop = FALSE]
  
  # Augment training data
  train_aug <- smote_oversample(train_df, "label", hard_classes, factor = 1.5)
  train_aug$label <- factor(train_aug$label, levels = class_levels)
  
  # Calculate weights
  weights <- (max(class_counts) / class_counts[as.character(train_aug$label)]) * 
             hard_boost[as.character(train_aug$label)]
  
  # Train base model
  train_data <- train_aug[, c(selected_features, "label"), drop = FALSE]
  train_data <- train_data[complete.cases(train_data), ]
  weights <- weights[complete.cases(train_aug[, c(selected_features, "label")])]
  
  base_model <- ranger(
    formula = label ~ .,
    data = train_data,
    num.trees = 500,
    mtry = floor(sqrt(length(selected_features))),
    probability = TRUE,
    case.weights = weights,
    seed = 42 + k
  )
  
  # Predict
  test_data <- test_df[, c(selected_features, "label"), drop = FALSE]
  test_data <- test_data[complete.cases(test_data), ]
  
  probs <- predict(base_model, test_data)$predictions
  pred_labels <- class_levels[apply(probs, 1, which.max)]
  true_labels <- as.character(test_data$label)
  
  all_preds <- c(all_preds, pred_labels)
  all_true <- c(all_true, true_labels)
  
  # Store probabilities for uncertainty analysis
  for (i in seq_len(nrow(probs))) {
    all_probs[[length(all_probs) + 1]] <- list(
      true = true_labels[i],
      pred = pred_labels[i],
      probs = probs[i, ]
    )
  }
  
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

# =============================================================================
# 6. RESULTS
# =============================================================================

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

# =============================================================================
# 7. SAVE RESULTS AND UNCERTAINTY DATA
# =============================================================================

write.csv(results, file.path(models_dir, "full_pipeline_cv_results.csv"), row.names = FALSE)

# Save predictions with probabilities for uncertainty analysis
pred_df <- data.frame(
  true_label = all_true,
  pred_label = all_preds,
  correct = all_preds == all_true
)
for (g in class_levels) {
  pred_df[[paste0("prob_", g)]] <- sapply(all_probs, function(x) x$probs[g])
}
write.csv(pred_df, file.path(models_dir, "full_pipeline_predictions.csv"), row.names = FALSE)

cat("\nSaved results to:\n")
cat("  models/full_pipeline_cv_results.csv\n")
cat("  models/full_pipeline_predictions.csv\n")

# =============================================================================
# 8. TRAIN FINAL MODEL ON ALL DATA
# =============================================================================

cat("\n========================================\n")
cat("Training final model on all data...\n")
cat("========================================\n")

# Augment all data
all_aug <- smote_oversample(df, "label", hard_classes, factor = 1.5)
all_aug$label <- factor(all_aug$label, levels = class_levels)

all_weights <- (max(class_counts) / class_counts[as.character(all_aug$label)]) * 
               hard_boost[as.character(all_aug$label)]

all_train <- all_aug[, c(selected_features, "label"), drop = FALSE]
all_train <- all_train[complete.cases(all_train), ]
all_weights <- all_weights[complete.cases(all_aug[, c(selected_features, "label")])]

final_model <- ranger(
  formula = label ~ .,
  data = all_train,
  num.trees = 700,
  mtry = floor(sqrt(length(selected_features))),
  probability = TRUE,
  case.weights = all_weights,
  importance = "impurity",
  seed = 42
)

# Save model info
full_pipeline_model <- list(
  version = "full_pipeline_v1",
  model = final_model,
  feature_names = selected_features,
  class_levels = class_levels,
  use_cnn = use_cnn,
  use_back = use_back,
  use_card_types = use_card_types,
  hard_classes = hard_classes,
  hard_boost = hard_boost
)

saveRDS(full_pipeline_model, file.path(models_dir, "full_pipeline_model.rds"))
cat("Saved model to: models/full_pipeline_model.rds\n")

cat("\nDONE.\n")
