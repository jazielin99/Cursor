# ============================================
# Advanced Ensemble with Ordinal Regression
# Methods: RF + XGBoost + Ordinal + Stacking
# ============================================

library(randomForest)
library(magick)

dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

library(xgboost)

cat("========================================\n")
cat("Advanced Ensemble Training\n")
cat("========================================\n\n")

# Feature extraction (280 features)
extract_features <- function(img_path) {
  tryCatch({
    img <- image_read(img_path)
    img <- image_resize(img, "224x224!")
    img_data <- image_data(img, channels = "rgb")
    img_array <- as.integer(img_data)
    img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1)) / 255
    
    r <- img_array[,,1]; g <- img_array[,,2]; b <- img_array[,,3]
    gray <- 0.299*r + 0.587*g + 0.114*b
    h <- nrow(gray); w <- ncol(gray)
    
    features <- c(mean(r), sd(r), mean(g), sd(g), mean(b), sd(b), mean(gray), sd(gray))
    features <- c(features, as.vector(hist(r, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(g, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(b, breaks=seq(0,1,0.1), plot=FALSE)$counts))
    features <- c(features, mean(abs(diff(gray))), mean(abs(apply(gray,2,diff))), 
                  sd(abs(diff(gray))), sd(abs(apply(gray,2,diff))))
    
    mid_h <- floor(h/2); mid_w <- floor(w/2)
    quadrants <- list(gray[1:mid_h,1:mid_w], gray[1:mid_h,(mid_w+1):w], 
                      gray[(mid_h+1):h,1:mid_w], gray[(mid_h+1):h,(mid_w+1):w])
    for (q in quadrants) {
      features <- c(features, mean(q), sd(q), min(q), max(q), max(q)-min(q))
      gh <- abs(q[-1,]-q[-nrow(q),]); gv <- abs(q[,-1]-q[,-ncol(q)])
      features <- c(features, mean(gh), sd(gh), mean(gv), sd(gv))
      features <- c(features, mean(q>0.9), mean(q<0.1), mean(q>0.8), mean(q<0.2))
      qh <- hist(q, breaks=16, plot=FALSE)$counts; qp <- qh/sum(qh); qp <- qp[qp>0]
      features <- c(features, -sum(qp*log2(qp)))
    }
    q_means <- sapply(quadrants, mean); q_sds <- sapply(quadrants, sd)
    features <- c(features, sd(q_means), max(q_means)-min(q_means), 
                  sd(q_sds), max(q_sds)-min(q_sds), 
                  which.max(q_means), which.min(q_means), 
                  which.max(q_sds), which.min(q_sds))
    
    third_h <- floor(h/3); third_w <- floor(w/3); zones <- list()
    for (row in 1:3) for (col in 1:3) { 
      r1 <- (row-1)*third_h+1; r2 <- min(row*third_h,h)
      c1 <- (col-1)*third_w+1; c2 <- min(col*third_w,w)
      zones <- c(zones, list(gray[r1:r2,c1:c2])) 
    }
    zone_means <- sapply(zones, mean); zone_sds <- sapply(zones, sd)
    features <- c(features, zone_means, zone_sds, 
                  sd(zone_means), max(zone_means)-min(zone_means), 
                  sd(zone_sds), max(zone_sds)-min(zone_sds))
    features <- c(features, mean(zones[[5]])-mean(sapply(zones[c(1:4,6:9)], mean)), 
                  sd(c(zones[[5]]))-mean(sapply(zones[c(1:4,6:9)], sd)))
    
    for (pct in c(0.10,0.15,0.20)) {
      cs <- floor(min(h,w)*pct)
      corners <- list(gray[1:cs,1:cs], gray[1:cs,(w-cs+1):w], 
                      gray[(h-cs+1):h,1:cs], gray[(h-cs+1):h,(w-cs+1):w])
      cm <- sapply(corners,mean); csd <- sapply(corners,sd)
      cmin <- sapply(corners,min); cmax <- sapply(corners,max)
      features <- c(features, cm, csd, cmin, cmax, 
                    sd(cm), max(cm)-min(cm), sd(csd), max(csd)-min(csd))
      cw <- sapply(corners, function(c) mean(c>0.8))
      features <- c(features, cw, max(cw), sd(cw))
      cd <- sapply(corners, function(c) mean(c<0.2))
      features <- c(features, cd, max(cd))
    }
    
    cs <- floor(min(h,w)*0.15)
    compute_sharp <- function(c) { 
      if(nrow(c)<3||ncol(c)<3) return(c(0,0,0,0))
      gh <- abs(c[-1,]-c[-nrow(c),]); gv <- abs(c[,-1]-c[,-ncol(c)])
      c(mean(gh),mean(gv),sd(gh),sd(gv)) 
    }
    corners <- list(gray[1:cs,1:cs], gray[1:cs,(w-cs+1):w], 
                    gray[(h-cs+1):h,1:cs], gray[(h-cs+1):h,(w-cs+1):w])
    for (c in corners) features <- c(features, compute_sharp(c))
    all_sharp <- sapply(corners, function(c) mean(compute_sharp(c)[1:2]))
    features <- c(features, all_sharp, sd(all_sharp), min(all_sharp), max(all_sharp)-min(all_sharp))
    
    row_means <- rowMeans(gray); col_means <- colMeans(gray)
    row_diffs <- abs(diff(row_means)); col_diffs <- abs(diff(col_means))
    features <- c(features, max(row_diffs), mean(row_diffs), sd(row_diffs), 
                  max(col_diffs), mean(col_diffs), sd(col_diffs), 
                  sum(row_diffs>0.1), sum(col_diffs>0.1))
    
    bs <- 5
    features <- c(features, mean(gray[1:bs,]), sd(gray[1:bs,]), 
                  mean(gray[(h-bs+1):h,]), sd(gray[(h-bs+1):h,]), 
                  mean(gray[,1:bs]), sd(gray[,1:bs]), 
                  mean(gray[,(w-bs+1):w]), sd(gray[,(w-bs+1):w]))
    bm <- c(mean(gray[1:bs,]), mean(gray[(h-bs+1):h,]), 
            mean(gray[,1:bs]), mean(gray[,(w-bs+1):w]))
    features <- c(features, sd(bm), max(bm)-min(bm))
    
    left <- gray[,1:floor(w/3)]; right <- gray[,(floor(2*w/3)):w]
    top <- gray[1:floor(h/3),]; bottom <- gray[(floor(2*h/3)):h,]
    center <- gray[floor(h/4):floor(3*h/4), floor(w/4):floor(3*w/4)]
    features <- c(features, mean(left), mean(right), abs(mean(left)-mean(right)), 
                  mean(top), mean(bottom), abs(mean(top)-mean(bottom)), 
                  mean(center), sd(center), sd(rowMeans(gray)), sd(colMeans(gray)))
    
    lap <- abs(gray[-c(1,h),-c(1,w)]*4 - gray[-c(1,h),-c(w-1,w)] - 
               gray[-c(1,h),-c(1,2)] - gray[-c(h-1,h),-c(1,w)] - gray[-c(1,2),-c(1,w)])
    features <- c(features, mean(lap), sd(lap))
    
    gh <- hist(gray, breaks=32, plot=FALSE)$counts
    gp <- gh/sum(gh); gp <- gp[gp>0]
    features <- c(features, -sum(gp*log2(gp)))
    for (ch in list(r,g,b)) { 
      ch_h <- hist(ch, breaks=16, plot=FALSE)$counts
      ch_p <- ch_h/sum(ch_h); ch_p <- ch_p[ch_p>0]
      features <- c(features, -sum(ch_p*log2(ch_p))) 
    }
    
    rm(img, img_data, img_array); gc(verbose=FALSE)
    return(features)
  }, error = function(e) return(NULL))
}

# --- Load cached features if available ---
cache_file <- "models/features_cache.rds"
if (file.exists(cache_file)) {
  cat("Loading cached features...\n")
  cache <- readRDS(cache_file)
  X <- cache$X
  y <- cache$y
  class_levels <- cache$class_levels
} else {
  # Load and extract features
  training_dir <- "data/training"
  class_dirs <- list.dirs(training_dir, recursive = FALSE)
  all_paths <- c(); all_labels <- c()
  
  for (class_dir in class_dirs) {
    class_name <- basename(class_dir)
    if (class_name == "NO_GRADE") next
    files <- list.files(class_dir, pattern = "\\.(jpg|jpeg|png)$", 
                        ignore.case = TRUE, full.names = TRUE)
    files <- files[!grepl("originals_backup", files)]
    if (length(files) > 0) { 
      all_paths <- c(all_paths, files)
      all_labels <- c(all_labels, rep(class_name, length(files))) 
    }
  }
  
  cat("Total images:", length(all_paths), "\n")
  cat("Extracting features...\n")
  
  features_list <- lapply(seq_along(all_paths), function(i) { 
    if(i %% 500 == 0) cat("  ", i, "/", length(all_paths), "\n")
    extract_features(all_paths[i]) 
  })
  
  valid <- !sapply(features_list, is.null)
  X <- do.call(rbind, features_list[valid])
  y <- factor(all_labels[valid])
  class_levels <- levels(y)
  
  # Cache for future runs
  saveRDS(list(X = X, y = y, class_levels = class_levels), cache_file)
  cat("Features cached.\n")
}

cat("Dataset:", nrow(X), "images,", ncol(X), "features\n")

# --- Upsampling ---
cat("\n========================================\n")
cat("Upsampling minority classes\n")
cat("========================================\n")

class_counts <- table(y)
max_count <- max(class_counts)
target_count <- floor(max_count * 0.8)

set.seed(42)
X_up <- X
y_up <- as.character(y)

for (class_name in names(class_counts)) {
  current_count <- class_counts[class_name]
  if (current_count < target_count) {
    need <- target_count - current_count
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

# Convert to numeric grades for ordinal methods
grade_to_num <- function(g) {
  g <- gsub("PSA_", "", as.character(g))
  as.numeric(g)
}
y_num <- sapply(y_up, grade_to_num)

# --- 5-Fold CV with Multiple Models ---
cat("\n========================================\n")
cat("5-Fold CV with Stacking Ensemble\n")
cat("========================================\n")

set.seed(42)
n <- nrow(X_up)
K <- 5
folds <- sample(rep(1:K, length.out = n))

# Storage for out-of-fold predictions (for stacking)
oof_rf_prob <- matrix(0, n, length(class_levels))
oof_xgb_prob <- matrix(0, n, length(class_levels))
oof_rf_reg <- numeric(n)  # Regression predictions

# Results
results <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  Within1 = numeric(),
  Within2 = numeric(),
  MAE = numeric()
)

num_classes <- length(class_levels)

for (k in 1:K) {
  cat("Fold", k, "...\n")
  
  train_idx <- folds != k
  test_idx <- folds == k
  
  X_train <- X_up[train_idx, ]
  y_train <- y_up[train_idx]
  y_train_num <- y_num[train_idx]
  X_test <- X_up[test_idx, ]
  y_test <- y_up[test_idx]
  y_test_num <- y_num[test_idx]
  
  # === Model 1: Random Forest Classification ===
  rf <- randomForest(x = X_train, y = y_train, ntree = 300)
  rf_prob <- predict(rf, X_test, type = "prob")
  oof_rf_prob[test_idx, ] <- rf_prob
  
  # === Model 2: Random Forest Regression (Ordinal) ===
  rf_reg <- randomForest(x = X_train, y = y_train_num, ntree = 300)
  rf_reg_pred <- predict(rf_reg, X_test)
  oof_rf_reg[test_idx] <- rf_reg_pred
  
  # === Model 3: XGBoost with better params ===
  y_train_xgb <- as.integer(y_train) - 1
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train_xgb)
  dtest <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = num_classes,
    eta = 0.05,
    max_depth = 8,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    eval_metric = "mlogloss"
  )
  
  xgb <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
  xgb_prob <- predict(xgb, dtest)
  xgb_prob_matrix <- matrix(xgb_prob, ncol = num_classes, byrow = TRUE)
  oof_xgb_prob[test_idx, ] <- xgb_prob_matrix
  
  gc(verbose = FALSE)
}

cat("\nEvaluating models...\n")

# --- Evaluate Each Model ---
evaluate_model <- function(pred_num, true_num, name) {
  acc <- mean(round(pred_num) == true_num, na.rm = TRUE)
  within1 <- mean(abs(pred_num - true_num) <= 1, na.rm = TRUE)
  within2 <- mean(abs(pred_num - true_num) <= 2, na.rm = TRUE)
  mae <- mean(abs(pred_num - true_num), na.rm = TRUE)
  data.frame(Model = name, Accuracy = acc, Within1 = within1, Within2 = within2, MAE = mae)
}

# RF Classification
rf_pred_class <- max.col(oof_rf_prob)
rf_pred_num <- sapply(class_levels[rf_pred_class], grade_to_num)
results <- rbind(results, evaluate_model(rf_pred_num, y_num, "RF Classification"))

# RF Regression (rounded to nearest grade)
rf_reg_rounded <- pmax(1, pmin(10, round(oof_rf_reg)))
results <- rbind(results, evaluate_model(rf_reg_rounded, y_num, "RF Regression"))

# XGBoost
xgb_pred_class <- max.col(oof_xgb_prob)
xgb_pred_num <- sapply(class_levels[xgb_pred_class], grade_to_num)
results <- rbind(results, evaluate_model(xgb_pred_num, y_num, "XGBoost"))

# === Ensemble Methods ===

# 1. Simple Average of Probabilities
avg_prob <- (oof_rf_prob + oof_xgb_prob) / 2
avg_pred_class <- max.col(avg_prob)
avg_pred_num <- sapply(class_levels[avg_pred_class], grade_to_num)
results <- rbind(results, evaluate_model(avg_pred_num, y_num, "Avg Ensemble"))

# 2. Weighted Average (RF 70%, XGB 30%)
weighted_prob <- 0.7 * oof_rf_prob + 0.3 * oof_xgb_prob
weighted_pred_class <- max.col(weighted_prob)
weighted_pred_num <- sapply(class_levels[weighted_pred_class], grade_to_num)
results <- rbind(results, evaluate_model(weighted_pred_num, y_num, "Weighted Ensemble"))

# 3. Hybrid: Average of Classification + Regression
hybrid_pred <- (avg_pred_num + rf_reg_rounded) / 2
hybrid_rounded <- pmax(1, pmin(10, round(hybrid_pred)))
results <- rbind(results, evaluate_model(hybrid_rounded, y_num, "Hybrid (Class+Reg)"))

# 4. Stacking: Use base model predictions as features
cat("\nTraining stacking meta-learner...\n")

# Create meta-features from OOF predictions
meta_features <- cbind(
  rf_class = rf_pred_num,
  rf_reg = oof_rf_reg,
  xgb_class = xgb_pred_num,
  rf_conf = apply(oof_rf_prob, 1, max),
  xgb_conf = apply(oof_xgb_prob, 1, max)
)

# Train a simple RF as meta-learner using internal CV
set.seed(123)
meta_folds <- sample(rep(1:5, length.out = n))
stack_pred <- numeric(n)

for (mk in 1:5) {
  meta_train_idx <- meta_folds != mk
  meta_test_idx <- meta_folds == mk
  
  meta_rf <- randomForest(
    x = meta_features[meta_train_idx, ],
    y = y_num[meta_train_idx],
    ntree = 100
  )
  stack_pred[meta_test_idx] <- predict(meta_rf, meta_features[meta_test_idx, ])
}

stack_rounded <- pmax(1, pmin(10, round(stack_pred)))
results <- rbind(results, evaluate_model(stack_rounded, y_num, "Stacking Ensemble"))

# --- Print Results ---
cat("\n========================================\n")
cat("MODEL COMPARISON\n")
cat("========================================\n\n")

results$Accuracy <- round(results$Accuracy * 100, 1)
results$Within1 <- round(results$Within1 * 100, 1)
results$Within2 <- round(results$Within2 * 100, 1)
results$MAE <- round(results$MAE, 2)

print(results)

# Find best model
best_idx <- which.max(results$Accuracy)
cat("\n>>> Best Model:", results$Model[best_idx], "with", results$Accuracy[best_idx], "% accuracy\n")

# --- Train Final Ensemble ---
cat("\n========================================\n")
cat("Training Final Models\n")
cat("========================================\n")

# Train on all data
cat("Training RF Classification...\n")
rf_final <- randomForest(x = X_up, y = y_up, ntree = 500, importance = TRUE)

cat("Training RF Regression...\n")
rf_reg_final <- randomForest(x = X_up, y = y_num, ntree = 500)

cat("Training XGBoost...\n")
y_up_xgb <- as.integer(y_up) - 1
dall <- xgb.DMatrix(data = X_up, label = y_up_xgb)
params <- list(
  objective = "multi:softprob",
  num_class = num_classes,
  eta = 0.05,
  max_depth = 8,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3
)
xgb_final <- xgb.train(params = params, data = dall, nrounds = 200, verbose = 0)

cat("Training Stacking Meta-learner...\n")
# Get predictions on all data for meta-learner
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

meta_final <- randomForest(x = meta_all, y = y_num, ntree = 200)

# Save all models
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

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
