# ============================================
# Advanced Model Training with Upsampling
# Models: Random Forest, XGBoost, Ensemble
# ============================================

library(randomForest)
library(magick)

dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

library(xgboost)

cat("========================================\n")
cat("Advanced Model Training\n")
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

# --- Load Data ---
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
cat("\nOriginal class distribution:\n")
print(table(all_labels))

# --- Extract Features ---
cat("\nExtracting features...\n")
features_list <- lapply(seq_along(all_paths), function(i) { 
  if(i %% 500 == 0) cat("  ", i, "/", length(all_paths), "\n")
  extract_features(all_paths[i]) 
})

valid <- !sapply(features_list, is.null)
X <- do.call(rbind, features_list[valid])
y <- factor(all_labels[valid])
cat("Features extracted:", nrow(X), "x", ncol(X), "\n")

# --- Upsampling ---
cat("\n========================================\n")
cat("Upsampling minority classes\n")
cat("========================================\n")

class_counts <- table(y)
max_count <- max(class_counts)
target_count <- floor(max_count * 0.8)

cat("Target samples per class:", target_count, "\n")

set.seed(42)
X_up <- X
y_up <- y

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
    y_up <- c(as.character(y_up), rep(class_name, need))
  }
}

y_up <- factor(y_up)
cat("\nUpsampled distribution:\n")
print(table(y_up))

# --- 5-Fold Cross Validation ---
cat("\n========================================\n")
cat("5-Fold Cross Validation\n")
cat("========================================\n")

set.seed(42)
n <- nrow(X_up)
K <- 5
folds <- sample(rep(1:K, length.out = n))

rf_accs <- numeric(K)
xgb_accs <- numeric(K)
ens_accs <- numeric(K)

all_preds_rf <- rep(NA, n)
all_preds_xgb <- rep(NA, n)
all_preds_ens <- rep(NA, n)
all_true <- as.integer(y_up)

class_levels <- levels(y_up)
num_classes <- length(class_levels)

for (k in 1:K) {
  cat("Fold", k, "...")
  
  train_idx <- folds != k
  test_idx <- folds == k
  
  X_train <- X_up[train_idx, ]
  y_train <- y_up[train_idx]
  X_test <- X_up[test_idx, ]
  y_test <- y_up[test_idx]
  
  # Random Forest
  rf <- randomForest(x = X_train, y = y_train, ntree = 300)
  rf_pred <- predict(rf, X_test)
  rf_prob <- predict(rf, X_test, type = "prob")
  rf_accs[k] <- mean(rf_pred == y_test)
  all_preds_rf[test_idx] <- as.integer(rf_pred)
  
  # XGBoost
  y_train_num <- as.integer(y_train) - 1
  y_test_num <- as.integer(y_test) - 1
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train_num)
  dtest <- xgb.DMatrix(data = X_test, label = y_test_num)
  
  params <- list(
    objective = "multi:softprob",
    num_class = num_classes,
    eta = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "mlogloss"
  )
  
  xgb <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
  
  xgb_prob <- predict(xgb, dtest)
  xgb_prob_matrix <- matrix(xgb_prob, ncol = num_classes, byrow = TRUE)
  xgb_pred_num <- max.col(xgb_prob_matrix) - 1
  xgb_accs[k] <- mean(xgb_pred_num == y_test_num)
  all_preds_xgb[test_idx] <- xgb_pred_num + 1
  
  # Ensemble (average probabilities)
  ensemble_prob <- (rf_prob + xgb_prob_matrix) / 2
  ensemble_pred <- max.col(ensemble_prob)
  ens_accs[k] <- mean(ensemble_pred == as.integer(y_test))
  all_preds_ens[test_idx] <- ensemble_pred
  
  cat(" RF:", round(rf_accs[k]*100,1), "% | XGB:", round(xgb_accs[k]*100,1), 
      "% | Ens:", round(ens_accs[k]*100,1), "%\n")
  
  gc(verbose = FALSE)
}

# --- Results ---
cat("\n========================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("========================================\n")

cat("\nMean Accuracy:\n")
cat(sprintf("  Random Forest: %5.1f%% (+/- %.1f%%)\n", mean(rf_accs)*100, sd(rf_accs)*100))
cat(sprintf("  XGBoost:       %5.1f%% (+/- %.1f%%)\n", mean(xgb_accs)*100, sd(xgb_accs)*100))
cat(sprintf("  Ensemble:      %5.1f%% (+/- %.1f%%)\n", mean(ens_accs)*100, sd(ens_accs)*100))

# Relaxed accuracy
grade_to_num <- function(g) {
  if (is.numeric(g)) return(g)
  g <- gsub("PSA_", "", as.character(g))
  as.numeric(g)
}

pred_num_ens <- sapply(class_levels[all_preds_ens], grade_to_num)
true_num <- sapply(class_levels[all_true], grade_to_num)

within_1 <- mean(abs(pred_num_ens - true_num) <= 1, na.rm = TRUE)
within_2 <- mean(abs(pred_num_ens - true_num) <= 2, na.rm = TRUE)

cat("\nEnsemble Relaxed Accuracy:\n")
cat("  Exact match:     ", round(mean(ens_accs) * 100, 1), "%\n")
cat("  Within 1 grade:  ", round(within_1 * 100, 1), "%\n")
cat("  Within 2 grades: ", round(within_2 * 100, 1), "%\n")

# --- Train Final Models on All Data ---
cat("\n========================================\n")
cat("Training Final Models\n")
cat("========================================\n")

# Random Forest
cat("Training final RF...\n")
rf_final <- randomForest(x = X_up, y = y_up, ntree = 500, importance = TRUE)
saveRDS(rf_final, "models/psa_rf_upsampled.rds")

# XGBoost
cat("Training final XGBoost...\n")
y_up_num <- as.integer(y_up) - 1
dall <- xgb.DMatrix(data = X_up, label = y_up_num)

params <- list(
  objective = "multi:softprob",
  num_class = num_classes,
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_final <- xgb.train(params = params, data = dall, nrounds = 150, verbose = 0)
xgb.save(xgb_final, "models/psa_xgb.model")

# Save model info
model_info <- list(class_levels = class_levels)
saveRDS(model_info, "models/model_info.rds")

cat("\nModels saved:\n")
cat("  - models/psa_rf_upsampled.rds\n")
cat("  - models/psa_xgb.model\n")
cat("  - models/model_info.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
