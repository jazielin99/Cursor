# ============================================
# PSA Card Grade Prediction Script
# ============================================

library(randomForest)
library(magick)

# --- Feature Extraction Function ---
extract_features <- function(img_path) {
  img <- image_read(img_path)
  img <- image_resize(img, "224x224!")
  img_data <- image_data(img, channels = "rgb")
  img_array <- as.integer(img_data)
  img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1)) / 255
  
  r <- img_array[,,1]; g <- img_array[,,2]; b <- img_array[,,3]
  gray <- 0.299*r + 0.587*g + 0.114*b
  h <- nrow(gray); w <- ncol(gray)
  
  features <- c(mean(r), sd(r), mean(g), sd(g), mean(b), sd(b), mean(gray), sd(gray))
  features <- c(features,
    as.vector(hist(r, breaks=seq(0,1,0.1), plot=FALSE)$counts),
    as.vector(hist(g, breaks=seq(0,1,0.1), plot=FALSE)$counts),
    as.vector(hist(b, breaks=seq(0,1,0.1), plot=FALSE)$counts))
  features <- c(features,
    mean(abs(diff(gray))), mean(abs(apply(gray, 2, diff))),
    sd(abs(diff(gray))), sd(abs(apply(gray, 2, diff))))
  cs <- floor(h/4)
  tl <- gray[1:cs, 1:cs]; tr <- gray[1:cs, (w-cs+1):w]
  bl <- gray[(h-cs+1):h, 1:cs]; br <- gray[(h-cs+1):h, (w-cs+1):w]
  features <- c(features, mean(tl), mean(tr), mean(bl), mean(br), sd(tl), sd(tr), sd(bl), sd(br))
  left <- gray[, 1:floor(w/3)]; right <- gray[, (floor(2*w/3)):w]
  top <- gray[1:floor(h/3), ]; bottom <- gray[(floor(2*h/3)):h, ]
  center <- gray[floor(h/4):floor(3*h/4), floor(w/4):floor(3*w/4)]
  features <- c(features, mean(left), mean(right), abs(mean(left)-mean(right)))
  features <- c(features, mean(top), mean(bottom), abs(mean(top)-mean(bottom)), mean(center), sd(center))
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
  return(matrix(features, nrow = 1))
}

# --- Load Models ---
model_11class <- readRDS("models/psa_rf_highres.rds")
model_3class <- readRDS("models/psa_rf_3class_highres.rds")

# --- Predict Function ---
predict_grade <- function(image_path) {
  features <- extract_features(image_path)
  pred_11 <- predict(model_11class, features)
  prob_11 <- predict(model_11class, features, type = "prob")
  pred_3 <- predict(model_3class, features)
  prob_3 <- predict(model_3class, features, type = "prob")
  
  cat("\n========================================\n")
  cat("PSA Grade Prediction\n")
  cat("Image:", image_path, "\n")
  cat("========================================\n\n")
  cat("11-Class Prediction:\n")
  cat("  Predicted Grade:", as.character(pred_11), "\n")
  cat("  Confidence:", round(max(prob_11) * 100, 1), "%\n")
  cat("  Top 3 predictions:\n")
  top3 <- sort(prob_11[1,], decreasing = TRUE)[1:3]
  for (i in 1:3) {
    cat("    ", names(top3)[i], ":", round(top3[i] * 100, 1), "%\n")
  }
  cat("\n3-Class Prediction:\n")
  cat("  Category:", as.character(pred_3), "\n")
  cat("  Confidence:", round(max(prob_3) * 100, 1), "%\n")
  cat("  Probabilities:\n")
  for (i in 1:3) {
    cat("    ", colnames(prob_3)[i], ":", round(prob_3[1,i] * 100, 1), "%\n")
  }
  
  return(list(
    grade_11 = as.character(pred_11),
    grade_3 = as.character(pred_3),
    probabilities_11 = prob_11,
    probabilities_3 = prob_3
  ))
}

# ============================================
# USAGE - Replace with your image path
# ============================================
result <- predict_grade("path/to/your/card_image.jpg")
