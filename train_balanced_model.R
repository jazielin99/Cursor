# ============================================
# Train BALANCED Random Forest Model
# Uses balanced sampling to address class imbalance
# ============================================

library(randomForest)
library(magick)

cat("========================================\n")
cat("Training BALANCED Model\n")
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
    
    # Basic features
    features <- c(mean(r), sd(r), mean(g), sd(g), mean(b), sd(b), mean(gray), sd(gray))
    features <- c(features, as.vector(hist(r, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(g, breaks=seq(0,1,0.1), plot=FALSE)$counts),
      as.vector(hist(b, breaks=seq(0,1,0.1), plot=FALSE)$counts))
    features <- c(features, mean(abs(diff(gray))), mean(abs(apply(gray,2,diff))), 
                  sd(abs(diff(gray))), sd(abs(apply(gray,2,diff))))
    
    # Quadrant analysis
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
    
    # 9-zone grid
    third_h <- floor(h/3); third_w <- floor(w/3); zones <- list()
    for (row in 1:3) {
      for (col in 1:3) { 
        r1 <- (row-1)*third_h+1; r2 <- min(row*third_h, h)
        c1 <- (col-1)*third_w+1; c2 <- min(col*third_w, w)
        zones <- c(zones, list(gray[r1:r2, c1:c2])) 
      }
    }
    zone_means <- sapply(zones, mean); zone_sds <- sapply(zones, sd)
    features <- c(features, zone_means, zone_sds, 
                  sd(zone_means), max(zone_means)-min(zone_means), 
                  sd(zone_sds), max(zone_sds)-min(zone_sds))
    features <- c(features, mean(zones[[5]])-mean(sapply(zones[c(1:4,6:9)], mean)), 
                  sd(c(zones[[5]]))-mean(sapply(zones[c(1:4,6:9)], sd)))
    
    # Multi-scale corner analysis
    for (pct in c(0.10, 0.15, 0.20)) {
      cs <- floor(min(h,w)*pct)
      corners <- list(gray[1:cs,1:cs], gray[1:cs,(w-cs+1):w], 
                      gray[(h-cs+1):h,1:cs], gray[(h-cs+1):h,(w-cs+1):w])
      cm <- sapply(corners, mean); csd <- sapply(corners, sd)
      cmin <- sapply(corners, min); cmax <- sapply(corners, max)
      features <- c(features, cm, csd, cmin, cmax, 
                    sd(cm), max(cm)-min(cm), sd(csd), max(csd)-min(csd))
      cw <- sapply(corners, function(c) mean(c>0.8))
      features <- c(features, cw, max(cw), sd(cw))
      cd <- sapply(corners, function(c) mean(c<0.2))
      features <- c(features, cd, max(cd))
    }
    
    # Corner sharpness
    cs <- floor(min(h,w)*0.15)
    compute_sharp <- function(c) { 
      if(nrow(c)<3 || ncol(c)<3) return(c(0,0,0,0))
      gh <- abs(c[-1,]-c[-nrow(c),]); gv <- abs(c[,-1]-c[,-ncol(c)])
      c(mean(gh), mean(gv), sd(gh), sd(gv)) 
    }
    corners <- list(gray[1:cs,1:cs], gray[1:cs,(w-cs+1):w], 
                    gray[(h-cs+1):h,1:cs], gray[(h-cs+1):h,(w-cs+1):w])
    for (c in corners) features <- c(features, compute_sharp(c))
    all_sharp <- sapply(corners, function(c) mean(compute_sharp(c)[1:2]))
    features <- c(features, all_sharp, sd(all_sharp), min(all_sharp), max(all_sharp)-min(all_sharp))
    
    # Crease detection
    row_means <- rowMeans(gray); col_means <- colMeans(gray)
    row_diffs <- abs(diff(row_means)); col_diffs <- abs(diff(col_means))
    features <- c(features, max(row_diffs), mean(row_diffs), sd(row_diffs), 
                  max(col_diffs), mean(col_diffs), sd(col_diffs), 
                  sum(row_diffs>0.1), sum(col_diffs>0.1))
    
    # Border analysis
    bs <- 5
    features <- c(features, mean(gray[1:bs,]), sd(gray[1:bs,]), 
                  mean(gray[(h-bs+1):h,]), sd(gray[(h-bs+1):h,]), 
                  mean(gray[,1:bs]), sd(gray[,1:bs]), 
                  mean(gray[,(w-bs+1):w]), sd(gray[,(w-bs+1):w]))
    bm <- c(mean(gray[1:bs,]), mean(gray[(h-bs+1):h,]), 
            mean(gray[,1:bs]), mean(gray[,(w-bs+1):w]))
    features <- c(features, sd(bm), max(bm)-min(bm))
    
    # Centering
    left <- gray[, 1:floor(w/3)]; right <- gray[, (floor(2*w/3)):w]
    top <- gray[1:floor(h/3), ]; bottom <- gray[(floor(2*h/3)):h, ]
    center <- gray[floor(h/4):floor(3*h/4), floor(w/4):floor(3*w/4)]
    features <- c(features, mean(left), mean(right), abs(mean(left)-mean(right)), 
                  mean(top), mean(bottom), abs(mean(top)-mean(bottom)), 
                  mean(center), sd(center), sd(rowMeans(gray)), sd(colMeans(gray)))
    
    # Sharpness
    lap <- abs(gray[-c(1,h),-c(1,w)]*4 - gray[-c(1,h),-c(w-1,w)] - 
               gray[-c(1,h),-c(1,2)] - gray[-c(h-1,h),-c(1,w)] - gray[-c(1,2),-c(1,w)])
    features <- c(features, mean(lap), sd(lap))
    
    # Entropy
    gh <- hist(gray, breaks=32, plot=FALSE)$counts
    gp <- gh/sum(gh); gp <- gp[gp>0]
    features <- c(features, -sum(gp*log2(gp)))
    for (ch in list(r, g, b)) { 
      ch_h <- hist(ch, breaks=16, plot=FALSE)$counts
      ch_p <- ch_h/sum(ch_h); ch_p <- ch_p[ch_p>0]
      features <- c(features, -sum(ch_p*log2(ch_p))) 
    }
    
    rm(img, img_data, img_array); gc(verbose=FALSE)
    return(features)
  }, error = function(e) return(NULL))
}

# Load training data
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
cat("\nClass distribution:\n")
print(table(all_labels))

# Extract features
cat("\nExtracting features...\n")
features_list <- lapply(seq_along(all_paths), function(i) { 
  if(i %% 500 == 0) cat("  ", i, "/", length(all_paths), "\n")
  extract_features(all_paths[i]) 
})

valid <- !sapply(features_list, is.null)
X <- do.call(rbind, features_list[valid])
y <- factor(all_labels[valid])
cat("Features extracted:", nrow(X), "images,", ncol(X), "features\n")

# Calculate balanced sampling sizes
class_counts <- table(y)
min_class_size <- min(class_counts)
sample_sizes <- rep(min_class_size, length(class_counts))
names(sample_sizes) <- names(class_counts)

cat("\nClass counts:\n")
print(class_counts)
cat("\nUsing balanced sampling with", min_class_size, "samples per class\n")

# Train balanced model
set.seed(42)
cat("\nTraining balanced RF model...\n")
rf_balanced <- randomForest(x = X, y = y, ntree = 500, 
                            sampsize = sample_sizes, 
                            importance = TRUE)

cat("\n========================================\n")
cat("BALANCED MODEL RESULTS\n")
cat("========================================\n")
cat("OOB Error:", round(rf_balanced$err.rate[500, "OOB"] * 100, 1), "%\n")
cat("OOB Accuracy:", round((1 - rf_balanced$err.rate[500, "OOB"]) * 100, 1), "%\n")

cat("\nConfusion Matrix:\n")
print(rf_balanced$confusion)

# Save model
saveRDS(rf_balanced, "models/psa_rf_balanced.rds")
cat("\nModel saved: models/psa_rf_balanced.rds\n")

# Train 3-class balanced model
y_3class <- ifelse(y %in% c("PSA_1", "PSA_1.5", "PSA_2", "PSA_3", "PSA_4"), "Low_1_4",
             ifelse(y %in% c("PSA_5", "PSA_6", "PSA_7"), "Mid_5_7", "High_8_10"))
y_3class <- factor(y_3class)
class_counts_3 <- table(y_3class)
min_3 <- min(class_counts_3)
sample_sizes_3 <- rep(min_3, 3)
names(sample_sizes_3) <- names(class_counts_3)

cat("\nTraining balanced 3-class model...\n")
rf_3class_balanced <- randomForest(x = X, y = y_3class, ntree = 500, 
                                   sampsize = sample_sizes_3, importance = TRUE)
saveRDS(rf_3class_balanced, "models/psa_rf_3class_balanced.rds")
cat("3-class model saved: models/psa_rf_3class_balanced.rds\n")

cat("\n========================================\n")
cat("TRAINING COMPLETE\n")
cat("========================================\n")
