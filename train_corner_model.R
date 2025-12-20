library(randomForest)
library(magick)

# Enhanced feature extraction with detailed corner analysis
extract_enhanced_features <- function(img_path) {
  tryCatch({
    img <- image_read(img_path)
    img <- image_resize(img, "224x224!")
    img_data <- image_data(img, channels = "rgb")
    img_array <- as.integer(img_data)
    img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1)) / 255
    
    r <- img_array[,,1]; g <- img_array[,,2]; b <- img_array[,,3]
    gray <- 0.299*r + 0.587*g + 0.114*b
    h <- nrow(gray); w <- ncol(gray)
    
    # === BASIC FEATURES ===
    features <- c(mean(r), sd(r), mean(g), sd(g), mean(b), sd(b), mean(gray), sd(gray))
    
    # Color histograms
    features <- c(features,
      as.vector(hist(r, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(g, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(b, breaks=seq(0,1,0.1), plot=FALSE)$counts))
    
    # Edge features
    features <- c(features,
      mean(abs(diff(gray))), mean(abs(apply(gray, 2, diff))),
      sd(abs(diff(gray))), sd(abs(apply(gray, 2, diff))))
    
    # === ENHANCED CORNER ANALYSIS ===
    for (corner_pct in c(0.10, 0.15, 0.20)) {
      cs <- floor(min(h, w) * corner_pct)
      
      tl <- gray[1:cs, 1:cs]
      tr <- gray[1:cs, (w-cs+1):w]
      bl <- gray[(h-cs+1):h, 1:cs]
      br <- gray[(h-cs+1):h, (w-cs+1):w]
      
      corners <- list(tl, tr, bl, br)
      corner_means <- sapply(corners, mean)
      corner_sds <- sapply(corners, sd)
      corner_mins <- sapply(corners, min)
      corner_maxs <- sapply(corners, max)
      
      features <- c(features, corner_means, corner_sds)
      features <- c(features, corner_mins, corner_maxs)
      
      # Corner consistency
      features <- c(features, sd(corner_means), max(corner_means) - min(corner_means))
      features <- c(features, sd(corner_sds), max(corner_sds) - min(corner_sds))
      
      # Whiteness detection (wear shows as white)
      corner_whiteness <- sapply(corners, function(c) mean(c > 0.8))
      features <- c(features, corner_whiteness, max(corner_whiteness), sd(corner_whiteness))
      
      # Darkness detection
      corner_darkness <- sapply(corners, function(c) mean(c < 0.2))
      features <- c(features, corner_darkness, max(corner_darkness))
    }
    
    # === CORNER SHARPNESS ===
    cs <- floor(min(h, w) * 0.15)
    
    compute_corner_sharpness <- function(corner) {
      if (nrow(corner) < 3 || ncol(corner) < 3) return(c(0, 0, 0, 0))
      grad_h <- abs(corner[-1,] - corner[-nrow(corner),])
      grad_v <- abs(corner[,-1] - corner[,-ncol(corner)])
      return(c(mean(grad_h), mean(grad_v), sd(grad_h), sd(grad_v)))
    }
    
    tl <- gray[1:cs, 1:cs]
    tr <- gray[1:cs, (w-cs+1):w]
    bl <- gray[(h-cs+1):h, 1:cs]
    br <- gray[(h-cs+1):h, (w-cs+1):w]
    
    tl_sharp <- compute_corner_sharpness(tl)
    tr_sharp <- compute_corner_sharpness(tr)
    bl_sharp <- compute_corner_sharpness(bl)
    br_sharp <- compute_corner_sharpness(br)
    
    features <- c(features, tl_sharp, tr_sharp, bl_sharp, br_sharp)
    
    all_sharp <- c(mean(tl_sharp[1:2]), mean(tr_sharp[1:2]), mean(bl_sharp[1:2]), mean(br_sharp[1:2]))
    features <- c(features, all_sharp, sd(all_sharp), min(all_sharp), max(all_sharp) - min(all_sharp))
    
    # === EDGE BORDER ANALYSIS ===
    border_size <- 5
    top_border <- gray[1:border_size, ]
    bottom_border <- gray[(h-border_size+1):h, ]
    left_border <- gray[, 1:border_size]
    right_border <- gray[, (w-border_size+1):w]
    
    features <- c(features, 
      mean(top_border), sd(top_border),
      mean(bottom_border), sd(bottom_border),
      mean(left_border), sd(left_border),
      mean(right_border), sd(right_border))
    
    border_means <- c(mean(top_border), mean(bottom_border), mean(left_border), mean(right_border))
    features <- c(features, sd(border_means), max(border_means) - min(border_means))
    
    # === STANDARD FEATURES ===
    left <- gray[, 1:floor(w/3)]; right <- gray[, (floor(2*w/3)):w]
    top <- gray[1:floor(h/3), ]; bottom <- gray[(floor(2*h/3)):h, ]
    center <- gray[floor(h/4):floor(3*h/4), floor(w/4):floor(3*w/4)]
    features <- c(features, mean(left), mean(right), abs(mean(left)-mean(right)))
    features <- c(features, mean(top), mean(bottom), abs(mean(top)-mean(bottom)))
    features <- c(features, mean(center), sd(center))
    
    features <- c(features, sd(rowMeans(gray)), sd(colMeans(gray)))
    
    lap <- abs(gray[-c(1,h),-c(1,w)]*4 - gray[-c(1,h),-c(w-1,w)] - gray[-c(1,h),-c(1,2)] - 
               gray[-c(h-1,h),-c(1,w)] - gray[-c(1,2),-c(1,w)])
    features <- c(features, mean(lap), sd(lap))
    
    gh <- hist(gray, breaks=32, plot=FALSE)$counts
    gp <- gh/sum(gh); gp <- gp[gp>0]
    features <- c(features, -sum(gp * log2(gp)))
    
    for (ch in list(r, g, b)) {
      ch_h <- hist(ch, breaks=16, plot=FALSE)$counts
      ch_p <- ch_h/sum(ch_h); ch_p <- ch_p[ch_p>0]
      features <- c(features, -sum(ch_p * log2(ch_p)))
    }
    
    rm(img, img_data, img_array); gc(verbose=FALSE)
    return(features)
  }, error = function(e) return(NULL))
}

cat("========================================\n")
cat("Training with Enhanced Corner Analysis\n")
cat("========================================\n\n")

training_dir <- "data/training"
class_dirs <- list.dirs(training_dir, recursive = FALSE)
all_paths <- c(); all_labels <- c()
for (class_dir in class_dirs) {
  class_name <- basename(class_dir)
  if (class_name == "NO_GRADE") next
  files <- list.files(class_dir, pattern = "\\.(jpg|jpeg|png)$", ignore.case = TRUE, full.names = TRUE)
  files <- files[!grepl("originals_backup", files)]
  if (length(files) > 0) {
    all_paths <- c(all_paths, files)
    all_labels <- c(all_labels, rep(class_name, length(files)))
  }
}
cat("Total images:", length(all_paths), "\n")

cat("Extracting enhanced features...\n")
features_list <- vector("list", length(all_paths))
for (i in seq_along(all_paths)) {
  if (i %% 500 == 0) cat("  ", i, "/", length(all_paths), "\n")
  features_list[[i]] <- extract_enhanced_features(all_paths[i])
  if (i %% 1000 == 0) gc(verbose=FALSE)
}

valid <- !sapply(features_list, is.null)
X <- do.call(rbind, features_list[valid])
y <- factor(all_labels[valid])
cat("Features per image:", ncol(X), "\n")

set.seed(42); n <- nrow(X); K <- 5
folds <- sample(rep(1:K, length.out = n))
accuracies <- numeric(K)
all_preds <- rep(NA, n); all_true <- as.integer(y)

cat("\n5-Fold CV:\n")
for (k in 1:K) {
  cat("Fold", k, "...")
  rf <- randomForest(x = X[folds != k,], y = y[folds != k], ntree = 300)
  preds <- predict(rf, X[folds == k,])
  all_preds[folds == k] <- as.integer(preds)
  accuracies[k] <- mean(preds == y[folds == k])
  cat(" ", round(accuracies[k]*100, 1), "%\n")
  gc(verbose=FALSE)
}

cat("\n========================================\n")
cat("RESULTS (Enhanced Corner Analysis)\n")
cat("========================================\n")
cat("Mean Accuracy:", round(mean(accuracies)*100, 1), "% (+/-", round(sd(accuracies)*100, 1), "%)\n")
cat("\nExact match:     ", round(mean(all_preds==all_true, na.rm=T)*100, 1), "%\n")
cat("Within 1 grade:  ", round(mean(abs(all_preds-all_true)<=1, na.rm=T)*100, 1), "%\n")
cat("Within 2 grades: ", round(mean(abs(all_preds-all_true)<=2, na.rm=T)*100, 1), "%\n")

cat("\nPer-Class:\n")
for (lvl in levels(y)) {
  idx <- which(y == lvl)
  cat(sprintf("%-8s: %5.1f%% (%d/%d)\n", lvl, mean(all_preds[idx]==all_true[idx], na.rm=T)*100, 
              sum(all_preds[idx]==all_true[idx], na.rm=T), length(idx)))
}

cat("\nTraining final model...\n")
final_rf <- randomForest(x = X, y = y, ntree = 300, importance = TRUE)
saveRDS(final_rf, "models/psa_rf_corners.rds")
cat("Model saved: models/psa_rf_corners.rds\n")

map_to_3class <- function(label) {
  num <- as.numeric(gsub("PSA_", "", as.character(label)))
  if (num <= 4) return("Low_1_4") else if (num <= 7) return("Mid_5_7") else return("High_8_10")
}
y_3class <- factor(sapply(all_labels[valid], map_to_3class), levels = c("Low_1_4", "Mid_5_7", "High_8_10"))
final_rf_3class <- randomForest(x = X, y = y_3class, ntree = 300)
saveRDS(final_rf_3class, "models/psa_rf_3class_corners.rds")
cat("3-class model saved: models/psa_rf_3class_corners.rds\n")
