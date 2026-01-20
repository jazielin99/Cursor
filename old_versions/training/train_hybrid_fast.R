# ============================================
# Fast Hybrid CNN-RF with PCA + Ordinal
# Uses PCA to reduce features for speed
# ============================================

library(ranger)

cat("========================================\n")
cat("Fast Hybrid CNN-RF (PCA + Ordinal)\n")
cat("========================================\n\n")

# --- Load CNN Features ---
cat("Loading CNN features...\n")
cnn_file <- "models/cnn_features_resnet50.csv"
cnn_data <- read.csv(cnn_file)
cnn_labels <- cnn_data$label
cnn_features <- as.matrix(cnn_data[, -ncol(cnn_data)])
cat("CNN features:", nrow(cnn_features), "samples,", ncol(cnn_features), "features\n")

# --- Load Advanced Features ---
cat("Loading advanced features...\n")
adv_file <- "models/advanced_features.csv"
adv_data <- read.csv(adv_file)
adv_features <- as.matrix(adv_data[, -ncol(adv_data)])
cat("Advanced features:", nrow(adv_features), "samples,", ncol(adv_features), "features\n")

# Handle any NA/Inf
adv_features[is.na(adv_features)] <- 0
adv_features[is.infinite(adv_features)] <- 0

# --- Apply PCA to Advanced Features ---
cat("\nApplying PCA to advanced features...\n")

# Remove zero-variance columns
col_vars <- apply(adv_features, 2, var, na.rm = TRUE)
keep_cols <- which(col_vars > 1e-10)
adv_filtered <- adv_features[, keep_cols]
cat("Removed", ncol(adv_features) - length(keep_cols), "zero-variance columns\n")
cat("Remaining features:", ncol(adv_filtered), "\n")

# Scale features
adv_scaled <- scale(adv_filtered)
adv_scaled[is.na(adv_scaled)] <- 0

# PCA - keep components explaining 95% variance
pca_result <- prcomp(adv_scaled, center = FALSE, scale. = FALSE)
var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
n_components <- min(which(var_explained >= 0.95), 200)  # Max 200 components
cat("PCA: keeping", n_components, "components (95% variance)\n")

adv_pca <- pca_result$x[, 1:n_components]

# --- Combine CNN + PCA Features ---
X <- cbind(cnn_features, adv_pca)
y <- cnn_labels

cat("Combined features:", nrow(X), "samples,", ncol(X), "features\n")

# Convert to numeric grades
grade_to_num <- function(g) {
  g <- gsub("PSA_", "", as.character(g))
  as.numeric(g)
}
y_num <- sapply(as.character(y), grade_to_num)

grade_order <- c("PSA_1", "PSA_1.5", "PSA_2", "PSA_3", "PSA_4", 
                 "PSA_5", "PSA_6", "PSA_7", "PSA_8", "PSA_9", "PSA_10")
grade_order <- grade_order[grade_order %in% unique(y)]

cat("\nGrade distribution:\n")
print(table(y))

# --- 5-Fold Cross-Validation ---
cat("\n========================================\n")
cat("5-Fold Cross-Validation with Ranger\n")
cat("========================================\n")

set.seed(42)
n <- nrow(X)
K <- 5
folds <- sample(rep(1:K, length.out = n))

all_preds <- rep(NA, n)

cv_results <- data.frame(
  Fold = 1:K,
  Accuracy = numeric(K),
  Within1 = numeric(K),
  Within2 = numeric(K)
)

for (k in 1:K) {
  cat("\n=== Fold", k, "/", K, "===\n")
  
  train_idx <- which(folds != k)
  test_idx <- which(folds == k)
  
  X_train <- X[train_idx, ]
  y_train_num <- y_num[train_idx]
  X_test <- X[test_idx, ]
  y_test_num <- y_num[test_idx]
  
  # Prepare data for ranger
  train_df <- data.frame(X_train, grade = y_train_num)
  colnames(train_df) <- c(paste0("V", 1:ncol(X_train)), "grade")
  
  test_df <- data.frame(X_test)
  colnames(test_df) <- paste0("V", 1:ncol(X_test))
  
  # Train Ranger (fast RF with ordinal regression)
  cat("  Training Ranger (500 trees)...\n")
  
  model <- ranger(
    grade ~ .,
    data = train_df,
    num.trees = 500,
    mtry = floor(sqrt(ncol(X_train))),
    min.node.size = 5,
    seed = 42,
    num.threads = 4
  )
  
  # Predict
  preds <- predict(model, test_df)$predictions
  preds_rounded <- pmax(1, pmin(10, round(preds)))
  all_preds[test_idx] <- preds_rounded
  
  # Metrics
  acc <- mean(preds_rounded == y_test_num)
  within1 <- mean(abs(preds_rounded - y_test_num) <= 1)
  within2 <- mean(abs(preds_rounded - y_test_num) <= 2)
  
  cv_results$Accuracy[k] <- acc
  cv_results$Within1[k] <- within1
  cv_results$Within2[k] <- within2
  
  cat(sprintf("  Accuracy: %.1f%%, Within 1: %.1f%%, Within 2: %.1f%%\n",
              acc * 100, within1 * 100, within2 * 100))
  
  gc(verbose = FALSE)
}

# --- Overall Results ---
cat("\n========================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("========================================\n\n")

cat("Per-Fold Results:\n")
print(round(cv_results[, -1] * 100, 1))

cat(sprintf("\n--- Summary ---\n"))
cat(sprintf("Accuracy:      %.1f%% (+/- %.1f%%)\n",
            mean(cv_results$Accuracy) * 100, sd(cv_results$Accuracy) * 100))
cat(sprintf("Within 1:      %.1f%% (+/- %.1f%%)\n",
            mean(cv_results$Within1) * 100, sd(cv_results$Within1) * 100))
cat(sprintf("Within 2:      %.1f%% (+/- %.1f%%)\n",
            mean(cv_results$Within2) * 100, sd(cv_results$Within2) * 100))

# Overall metrics
cat("\n--- Overall (All Folds Combined) ---\n")
cat(sprintf("Exact match:      %.1f%%\n", mean(all_preds == y_num, na.rm = TRUE) * 100))
cat(sprintf("Within 1 grade:   %.1f%%\n", mean(abs(all_preds - y_num) <= 1, na.rm = TRUE) * 100))
cat(sprintf("Within 2 grades:  %.1f%%\n", mean(abs(all_preds - y_num) <= 2, na.rm = TRUE) * 100))

# Per-class accuracy
cat("\n--- Per-Class Accuracy ---\n")
for (lvl in grade_order) {
  grade_num <- grade_to_num(lvl)
  idx <- which(y_num == grade_num)
  if (length(idx) > 0) {
    acc <- mean(all_preds[idx] == y_num[idx], na.rm = TRUE)
    cat(sprintf("  %s: %.1f%% (%d samples)\n", lvl, acc * 100, length(idx)))
  }
}

# --- Train Final Model ---
cat("\n========================================\n")
cat("Training Final Model on All Data\n")
cat("========================================\n")

train_df_all <- data.frame(X, grade = y_num)
colnames(train_df_all) <- c(paste0("V", 1:ncol(X)), "grade")

cat("Training Ranger (1000 trees)...\n")
final_model <- ranger(
  grade ~ .,
  data = train_df_all,
  num.trees = 1000,
  importance = "impurity",
  seed = 42,
  num.threads = 4
)

# Save model and PCA info
cat("\nSaving model...\n")
model_output <- list(
  model = final_model,
  pca = pca_result,
  n_pca_components = n_components,
  cv_results = cv_results,
  grade_order = grade_order,
  cnn_feature_cols = ncol(cnn_features),
  pca_scaling = attr(adv_scaled, "scaled:center")
)
saveRDS(model_output, "models/hybrid_pca_model.rds")
cat("Model saved: models/hybrid_pca_model.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
