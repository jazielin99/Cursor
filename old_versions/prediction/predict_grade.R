# ============================================
# PSA Card Grade Prediction Script
# Enhanced with Quadrant + Zone + Corner Analysis
# ============================================

library(randomForest)
library(magick)

# --- Crop Card from Slab ---
crop_card_from_slab <- function(img) {
  info <- image_info(img)
  w <- info$width; h <- info$height
  aspect <- w / h
  if (aspect > 1.2) {
    card_w <- floor(h * 0.65); card_h <- floor(h * 0.85)
    x_off <- floor((w - card_w) / 2); y_off <- floor(h * 0.10)
  } else if (aspect < 0.8) {
    card_w <- floor(w * 0.85); card_h <- floor(h * 0.60)
    x_off <- floor((w - card_w) / 2); y_off <- floor(h * 0.25)
  } else {
    card_w <- floor(w * 0.70); card_h <- floor(h * 0.70)
    x_off <- floor((w - card_w) / 2); y_off <- floor((h - card_h) / 2)
  }
  return(image_crop(img, sprintf("%dx%d+%d+%d", card_w, card_h, x_off, y_off)))
}

# --- Enhanced Feature Extraction ---
extract_features <- function(img_path, crop_slab = TRUE) {
  img <- image_read(img_path)
  if (crop_slab) img <- crop_card_from_slab(img)
  img <- image_resize(img, "224x224!")
  img_data <- image_data(img, channels = "rgb")
  img_array <- as.integer(img_data)
  img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1)) / 255
  
  r <- img_array[,,1]; g <- img_array[,,2]; b <- img_array[,,3]
  gray <- 0.299*r + 0.587*g + 0.114*b
  h <- nrow(gray); w <- ncol(gray)
  
  # Basic color stats
  features <- c(mean(r), sd(r), mean(g), sd(g), mean(b), sd(b), mean(gray), sd(gray))
  
  # Color histograms
  features <- c(features, as.vector(hist(r, breaks=seq(0,1,0.1), plot=FALSE)$counts),
    as.vector(hist(g, breaks=seq(0,1,0.1), plot=FALSE)$counts),
    as.vector(hist(b, breaks=seq(0,1,0.1), plot=FALSE)$counts))
  
  # Edge features
  features <- c(features, mean(abs(diff(gray))), mean(abs(apply(gray, 2, diff))),
    sd(abs(diff(gray))), sd(abs(apply(gray, 2, diff))))
  
  # === QUADRANT ANALYSIS ===
  mid_h <- floor(h/2); mid_w <- floor(w/2)
  quadrants <- list(gray[1:mid_h, 1:mid_w], gray[1:mid_h, (mid_w+1):w],
                    gray[(mid_h+1):h, 1:mid_w], gray[(mid_h+1):h, (mid_w+1):w])
  
  for (q in quadrants) {
    features <- c(features, mean(q), sd(q), min(q), max(q), max(q) - min(q))
    grad_h <- abs(q[-1,] - q[-nrow(q),]); grad_v <- abs(q[,-1] - q[,-ncol(q)])
    features <- c(features, mean(grad_h), sd(grad_h), mean(grad_v), sd(grad_v))
    features <- c(features, mean(q > 0.9), mean(q < 0.1), mean(q > 0.8), mean(q < 0.2))
    qh <- hist(q, breaks=16, plot=FALSE)$counts; qp <- qh/sum(qh); qp <- qp[qp > 0]
    features <- c(features, -sum(qp * log2(qp)))
  }
  
  q_means <- sapply(quadrants, mean); q_sds <- sapply(quadrants, sd)
  features <- c(features, sd(q_means), max(q_means) - min(q_means),
    sd(q_sds), max(q_sds) - min(q_sds),
    which.max(q_means), which.min(q_means), which.max(q_sds), which.min(q_sds))
  
  # === 9-ZONE GRID ===
  third_h <- floor(h/3); third_w <- floor(w/3)
  zones <- list()
  for (row in 1:3) {
    for (col in 1:3) {
      r1 <- (row-1)*third_h + 1; r2 <- min(row*third_h, h)
      c1 <- (col-1)*third_w + 1; c2 <- min(col*third_w, w)
      zones <- c(zones, list(gray[r1:r2, c1:c2]))
    }
  }
  zone_means <- sapply(zones, mean); zone_sds <- sapply(zones, sd)
  features <- c(features, zone_means, zone_sds,
    sd(zone_means), max(zone_means) - min(zone_means),
    sd(zone_sds), max(zone_sds) - min(zone_sds))
  features <- c(features, mean(zones[[5]]) - mean(sapply(zones[c(1:4,6:9)], mean)),
    sd(c(zones[[5]])) - mean(sapply(zones[c(1:4,6:9)], sd)))
  
  # === CORNER ANALYSIS ===
  for (corner_pct in c(0.10, 0.15, 0.20)) {
    cs <- floor(min(h, w) * corner_pct)
    corners <- list(gray[1:cs, 1:cs], gray[1:cs, (w-cs+1):w],
                    gray[(h-cs+1):h, 1:cs], gray[(h-cs+1):h, (w-cs+1):w])
    
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
  
  # Corner sharpness
  cs <- floor(min(h, w) * 0.15)
  compute_sharp <- function(corner) {
    if (nrow(corner) < 3 || ncol(corner) < 3) return(c(0, 0, 0, 0))
    gh <- abs(corner[-1,] - corner[-nrow(corner),])
    gv <- abs(corner[,-1] - corner[,-ncol(corner)])
    return(c(mean(gh), mean(gv), sd(gh), sd(gv)))
  }
  corners <- list(gray[1:cs, 1:cs], gray[1:cs, (w-cs+1):w],
                  gray[(h-cs+1):h, 1:cs], gray[(h-cs+1):h, (w-cs+1):w])
  for (c in corners) features <- c(features, compute_sharp(c))
  all_sharp <- sapply(corners, function(c) mean(compute_sharp(c)[1:2]))
  features <- c(features, all_sharp, sd(all_sharp), min(all_sharp), max(all_sharp) - min(all_sharp))
  
  # === CREASE DETECTION ===
  row_means <- rowMeans(gray); col_means <- colMeans(gray)
  row_diffs <- abs(diff(row_means)); col_diffs <- abs(diff(col_means))
  features <- c(features, max(row_diffs), mean(row_diffs), sd(row_diffs),
    max(col_diffs), mean(col_diffs), sd(col_diffs),
    sum(row_diffs > 0.1), sum(col_diffs > 0.1))
  
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
  features <- c(features, mean(left), mean(right), abs(mean(left) - mean(right)))
  features <- c(features, mean(top), mean(bottom), abs(mean(top) - mean(bottom)))
  features <- c(features, mean(center), sd(center))
  features <- c(features, sd(rowMeans(gray)), sd(colMeans(gray)))
  
  # Sharpness
  lap <- abs(gray[-c(1,h),-c(1,w)]*4 - gray[-c(1,h),-c(w-1,w)] - gray[-c(1,h),-c(1,2)] - 
             gray[-c(h-1,h),-c(1,w)] - gray[-c(1,2),-c(1,w)])
  features <- c(features, mean(lap), sd(lap))
  
  # Entropy
  gh <- hist(gray, breaks=32, plot=FALSE)$counts; gp <- gh/sum(gh); gp <- gp[gp > 0]
  features <- c(features, -sum(gp * log2(gp)))
  for (ch in list(r, g, b)) {
    ch_h <- hist(ch, breaks=16, plot=FALSE)$counts
    ch_p <- ch_h/sum(ch_h); ch_p <- ch_p[ch_p > 0]
    features <- c(features, -sum(ch_p * log2(ch_p)))
  }
  
  return(matrix(features, nrow = 1))
}

# --- Load Models ---
model_11class <- readRDS("models/psa_rf_quadrant.rds")
model_3class <- readRDS("models/psa_rf_3class_quadrant.rds")

# --- Predict Function ---
predict_grade <- function(image_path, crop_slab = TRUE) {
  features <- extract_features(image_path, crop_slab = crop_slab)
  
  pred_11 <- predict(model_11class, features)
  prob_11 <- predict(model_11class, features, type = "prob")
  pred_3 <- predict(model_3class, features)
  prob_3 <- predict(model_3class, features, type = "prob")
  
  cat("\n========================================\n")
  cat("PSA Grade Prediction\n")
  cat("Image:", image_path, "\n")
  cat("Crop slab:", crop_slab, "\n")
  cat("========================================\n\n")
  
  cat("11-Class Prediction:\n")
  cat("  Predicted Grade:", as.character(pred_11), "\n")
  cat("  Confidence:", round(max(prob_11) * 100, 1), "%\n")
  top3 <- sort(prob_11[1,], decreasing = TRUE)[1:3]
  cat("  Top 3:\n")
  for (i in 1:3) {
    cat("    ", names(top3)[i], ":", round(top3[i] * 100, 1), "%\n")
  }
  
  cat("\n3-Class Prediction:\n")
  cat("  Category:", as.character(pred_3), "\n")
  cat("  Confidence:", round(max(prob_3) * 100, 1), "%\n")
  
  return(list(
    grade_11 = as.character(pred_11),
    grade_3 = as.character(pred_3),
    probabilities_11 = prob_11,
    probabilities_3 = prob_3
  ))
}

# ============================================
# USAGE
# ============================================
# For slabbed cards (will crop out the card):
#   result <- predict_grade("path/to/slabbed_card.jpg", crop_slab = TRUE)
#
# For raw cards (no cropping needed):
#   result <- predict_grade("path/to/raw_card.jpg", crop_slab = FALSE)
