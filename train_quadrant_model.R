library(randomForest)
library(magick)

# Enhanced feature extraction with QUADRANT analysis
extract_quadrant_features <- function(img_path) {
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
    
    # === QUADRANT ANALYSIS (4 sections) ===
    mid_h <- floor(h/2); mid_w <- floor(w/2)
    
    q_tl <- gray[1:mid_h, 1:mid_w]           # Top-left
    q_tr <- gray[1:mid_h, (mid_w+1):w]       # Top-right
    q_bl <- gray[(mid_h+1):h, 1:mid_w]       # Bottom-left
    q_br <- gray[(mid_h+1):h, (mid_w+1):w]   # Bottom-right
    
    quadrants <- list(q_tl, q_tr, q_bl, q_br)
    
    # For each quadrant: basic stats
    for (q in quadrants) {
      features <- c(features, mean(q), sd(q), min(q), max(q), max(q) - min(q))
      
      # Texture within quadrant
      grad_h <- abs(q[-1,] - q[-nrow(q),])
      grad_v <- abs(q[,-1] - q[,-ncol(q)])
      features <- c(features, mean(grad_h), sd(grad_h), mean(grad_v), sd(grad_v))
      
      # Defect indicators
      features <- c(features, 
        mean(q > 0.9),      # Very bright spots (surface damage)
        mean(q < 0.1),      # Very dark spots (stains/dirt)
        mean(q > 0.8),      # Light areas
        mean(q < 0.2))      # Dark areas
      
      # Histogram entropy (uniformity)
      qh <- hist(q, breaks=16, plot=FALSE)$counts
      qp <- qh/sum(qh); qp <- qp[qp > 0]
      features <- c(features, -sum(qp * log2(qp)))
    }
    
    # Quadrant comparison (detect localized damage)
    q_means <- sapply(quadrants, mean)
    q_sds <- sapply(quadrants, sd)
    features <- c(features, 
      sd(q_means), max(q_means) - min(q_means),  # Brightness consistency
      sd(q_sds), max(q_sds) - min(q_sds),        # Texture consistency
      which.max(q_means), which.min(q_means),     # Which quadrant is brightest/darkest
      which.max(q_sds), which.min(q_sds))         # Which has most/least texture
    
    # === 9-ZONE GRID ANALYSIS (3x3) ===
    third_h <- floor(h/3); third_w <- floor(w/3)
    zones <- list()
    zone_idx <- 1
    for (row in 1:3) {
      for (col in 1:3) {
        r1 <- (row-1)*third_h + 1; r2 <- min(row*third_h, h)
        c1 <- (col-1)*third_w + 1; c2 <- min(col*third_w, w)
        zones[[zone_idx]] <- gray[r1:r2, c1:c2]
        zone_idx <- zone_idx + 1
      }
    }
    
    # Zone statistics
    zone_means <- sapply(zones, mean)
    zone_sds <- sapply(zones, sd)
    features <- c(features, zone_means, zone_sds)
    
    # Zone consistency
    features <- c(features, 
      sd(zone_means), max(zone_means) - min(zone_means),
      sd(zone_sds), max(zone_sds) - min(zone_sds))
    
    # Center zone vs edge zones
    center_zone <- zones[[5]]  # Middle zone
    edge_zones <- zones[c(1,2,3,4,6,7,8,9)]
    edge_mean <- mean(sapply(edge_zones, mean))
    features <- c(features, mean(center_zone) - edge_mean, sd(c(center_zone)) - mean(sapply(edge_zones, sd)))
    
    # === ENHANCED CORNER ANALYSIS ===
    for (corner_pct in c(0.10, 0.15, 0.20)) {
      cs <- floor(min(h, w) * corner_pct)
      tl <- gray[1:cs, 1:cs]; tr <- gray[1:cs, (w-cs+1):w]
      bl <- gray[(h-cs+1):h, 1:cs]; br <- gray[(h-cs+1):h, (w-cs+1):w]
      corners <- list(tl, tr, bl, br)
      
      corner_means <- sapply(corners, mean); corner_sds <- sapply(corners, sd)
      corner_mins <- sapply(corners, min); corner_maxs <- sapply(corners, max)
      features <- c(features, corner_means, corner_sds, corner_mins, corner_maxs)
      features <- c(features, sd(corner_means), max(corner_means) - min(corner_means))
      features <- c(features, sd(corner_sds), max(corner_sds) - min(corner_sds))
      
      corner_whiteness <- sapply(corners, function(c) mean(c > 0.8))
      features <- c(features, corner_whiteness, max(corner_whiteness), sd(corner_whiteness))
      corner_darkness <- sapply(corners, function(c) mean(c < 0.2))
      features <- c(features, corner_darkness, max(corner_darkness))
    }
    
    # === CORNER SHARPNESS ===
    cs <- floor(min(h, w) * 0.15)
    compute_sharp <- function(corner) {
      if (nrow(corner) < 3 || ncol(corner) < 3) return(c(0,0,0,0))
      gh <- abs(corner[-1,] - corner[-nrow(corner),])
      gv <- abs(corner[,-1] - corner[,-ncol(corner)])
      return(c(mean(gh), mean(gv), sd(gh), sd(gv)))
    }
    tl <- gray[1:cs, 1:cs]; tr <- gray[1:cs, (w-cs+1):w]
    bl <- gray[(h-cs+1):h, 1:cs]; br <- gray[(h-cs+1):h, (w-cs+1):w]
    features <- c(features, compute_sharp(tl), compute_sharp(tr), compute_sharp(bl), compute_sharp(br))
    all_sharp <- c(mean(compute_sharp(tl)[1:2]), mean(compute_sharp(tr)[1:2]), 
                   mean(compute_sharp(bl)[1:2]), mean(compute_sharp(br)[1:2]))
    features <- c(features, all_sharp, sd(all_sharp), min(all_sharp), max(all_sharp) - min(all_sharp))
    
    # === CREASE/LINE DETECTION ===
    # Look for horizontal and vertical lines (creases show as brightness changes)
    row_means <- rowMeans(gray)
    col_means <- colMeans(gray)
    row_diffs <- abs(diff(row_means))
    col_diffs <- abs(diff(col_means))
    
    features <- c(features,
      max(row_diffs), mean(row_diffs), sd(row_diffs),  # Horizontal crease indicators
      max(col_diffs), mean(col_diffs), sd(col_diffs),  # Vertical crease indicators
      sum(row_diffs > 0.1), sum(col_diffs > 0.1))      # Count of potential creases
    
    # === BORDER ANALYSIS ===
    border_size <- 5
    features <- c(features, 
      mean(gray[1:border_size,]), sd(gray[1:border_size,]),
      mean(gray[(h-border_size+1):h,]), sd(gray[(h-border_size+1):h,]),
      mean(gray[,1:border_size]), sd(gray[,1:border_size]),
      mean(gray[,(w-border_size+1):w]), sd(gray[,(w-border_size+1):w]))
    border_means <- c(mean(gray[1:border_size,]), mean(gray[(h-border_size+1):h,]), 
                      mean(gray[,1:border_size]), mean(gray[,(w-border_size+1):w]))
    features <- c(features, sd(border_means), max(border_means) - min(border_means))
    
    # === CENTERING ===
    left <- gray[, 1:floor(w/3)]; right <- gray[, (floor(2*w/3)):w]
    top <- gray[1:floor(h/3), ]; bottom <- gray[(floor(2*h/3)):h, ]
    center <- gray[floor(h/4):floor(3*h/4), floor(w/4):floor(3*w/4)]
    features <- c(features, mean(left), mean(right), abs(mean(left)-mean(right)))
    features <- c(features, mean(top), mean(bottom), abs(mean(top)-mean(bottom)), mean(center), sd(center))
    features <- c(features, sd(rowMeans(gray)), sd(colMeans(gray)))
    
    # Sharpness (Laplacian)
    lap <- abs(gray[-c(1,h),-c(1,w)]*4 - gray[-c(1,h),-c(w-1,w)] - gray[-c(1,h),-c(1,2)] - 
               gray[-c(h-1,h),-c(1,w)] - gray[-c(1,2),-c(1,w)])
    features <- c(features, mean(lap), sd(lap))
    
    # Entropy
    gh <- hist(gray, breaks=32, plot=FALSE)$counts; gp <- gh/sum(gh); gp <- gp[gp>0]
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
cat("Training with Quadrant + Zone Analysis\n")
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

cat("Extracting quadrant features...\n")
features_list <- vector("list", length(all_paths))
for (i in seq_along(all_paths)) {
  if (i %% 500 == 0) cat("  ", i, "/", length(all_paths), "\n")
  features_list[[i]] <- extract_quadrant_features(all_paths[i])
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
cat("RESULTS (Quadrant + Zone Analysis)\n")
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
saveRDS(final_rf, "models/psa_rf_quadrant.rds")
cat("Model saved: models/psa_rf_quadrant.rds\n")

map_to_3class <- function(label) {
  num <- as.numeric(gsub("PSA_", "", as.character(label)))
  if (num <= 4) return("Low_1_4") else if (num <= 7) return("Mid_5_7") else return("High_8_10")
}
y_3class <- factor(sapply(all_labels[valid], map_to_3class), levels = c("Low_1_4", "Mid_5_7", "High_8_10"))
final_rf_3class <- randomForest(x = X, y = y_3class, ntree = 300)
saveRDS(final_rf_3class, "models/psa_rf_3class_quadrant.rds")
cat("3-class model saved: models/psa_rf_3class_quadrant.rds\n")
