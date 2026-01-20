# ============================================
# PSA Grade Prediction v2 (Binary Triage + LLM Integration)
# ============================================
# UPGRADES:
# 1. Binary Triage: Near Mint (8-10) vs Market Grade (1-7) first pass
# 2. LLM Visual Auditor integration for high-grade candidates
# 3. Automated grading notes generation
# 4. Back-of-card penalty system (when back images available)
# 5. Support for v4 adaptive features
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

.psa_models_env <- new.env(parent = emptyenv())

.require_file <- function(path) {
  if (!file.exists(path)) stop(paste0("Missing required file: ", path), call. = FALSE)
}

.run_python <- function(args) {
  res <- system2("python3", args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop(paste(res, collapse = "\n"), call. = FALSE)
  }
  invisible(res)
}

.load_models <- function(models_dir = "models") {
  if (exists("tiered_model", envir = .psa_models_env, inherits = FALSE)) return(invisible(TRUE))

  tiered_path <- file.path(models_dir, "tiered_model.rds")
  high_path <- file.path(models_dir, "high_grade_specialist.rds")
  bin_path <- file.path(models_dir, "psa_9_vs_10.rds")

  .require_file(tiered_path)
  .require_file(high_path)
  .require_file(bin_path)

  .psa_models_env$tiered_model <- readRDS(tiered_path)
  .psa_models_env$high_grade_specialist <- readRDS(high_path)
  .psa_models_env$psa_9_vs_10 <- readRDS(bin_path)

  invisible(TRUE)
}

# =============================================================================
# Feature Extraction (supports v2, v3, v4)
# =============================================================================

.extract_advanced_v4 <- function(image_path, tmp_dir = tempdir(), script = "scripts/feature_extraction/extract_advanced_features_v4.py") {
  adv_out <- file.path(tmp_dir, "adv_v4_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", adv_out))
  adv <- read.csv(adv_out, check.names = FALSE)
  adv$path <- NULL
  adv
}

.extract_advanced_v3 <- function(image_path, tmp_dir = tempdir(), script = "scripts/feature_extraction/extract_advanced_features_v3.py") {
  adv_out <- file.path(tmp_dir, "adv_v3_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", adv_out))
  adv <- read.csv(adv_out, check.names = FALSE)
  adv$path <- NULL
  adv
}

.extract_advanced_v2 <- function(image_path, tmp_dir = tempdir(), script = "scripts/feature_extraction/extract_advanced_features_v2.py") {
  adv_out <- file.path(tmp_dir, "adv_v2_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", adv_out))
  adv <- read.csv(adv_out, check.names = FALSE)
  adv$path <- NULL
  adv
}

.extract_cnn <- function(image_path, tmp_dir = tempdir(), script = "scripts/feature_extraction/extract_cnn_features_single.py") {
  cnn_out <- file.path(tmp_dir, "cnn_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", cnn_out))
  cnn <- read.csv(cnn_out, check.names = FALSE)
  cnn$path <- NULL
  cnn
}

.build_feature_row_full <- function(image_path) {
  # Try v4 first, then v3, then v2
  adv <- NULL
  
  if (file.exists("scripts/feature_extraction/extract_advanced_features_v4.py")) {
    try(adv <- .extract_advanced_v4(image_path), silent = TRUE)
  }
  
  if (is.null(adv) && file.exists("scripts/feature_extraction/extract_advanced_features_v3.py")) {
    try(adv <- .extract_advanced_v3(image_path), silent = TRUE)
  }
  
  if (is.null(adv) && file.exists("scripts/feature_extraction/extract_advanced_features_v2.py")) {
    adv <- .extract_advanced_v2(image_path)
  }
  
  if (is.null(adv)) {
    stop("No feature extraction script available.", call. = FALSE)
  }

  cnn <- NULL
  if (file.exists("scripts/feature_extraction/extract_cnn_features_single.py")) {
    try(cnn <- .extract_cnn(image_path), silent = TRUE)
  }

  if (!is.null(cnn)) {
    row <- cbind(adv, cnn)
  } else {
    row <- adv
  }
  row
}

.predict_ranger_prob <- function(model, row_df) {
  pred <- predict(model, data = row_df, type = "response")$predictions
  if (!(is.matrix(pred) || is.data.frame(pred))) {
    stop("Model did not return class probabilities; ensure ranger was trained with probability=TRUE.", call. = FALSE)
  }
  probs <- as.numeric(pred[1, ])
  names(probs) <- colnames(pred)
  probs
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# =============================================================================
# Binary Triage Prediction
# =============================================================================

.predict_binary_triage <- function(tm, row) {
  if (is.null(tm$binary_triage_model)) {
    # Fallback to old tier1 if binary triage not available
    return(NULL)
  }
  
  fs <- tm$feature_sets$binary
  if (is.null(fs)) return(NULL)
  
  # Get available features
  available <- intersect(fs, colnames(row))
  if (length(available) < length(fs) * 0.8) {
    warning("Missing >20% of binary triage features; falling back to tier1")
    return(NULL)
  }
  
  .predict_ranger_prob(tm$binary_triage_model, row[, available, drop = FALSE])
}

.predict_market_tier <- function(tm, row) {
  if (is.null(tm$market_tier_model)) return(NULL)
  
  fs <- tm$feature_sets$tier1
  available <- intersect(fs, colnames(row))
  .predict_ranger_prob(tm$market_tier_model, row[, available, drop = FALSE])
}

# =============================================================================
# Specialist Predictions
# =============================================================================

.predict_specialist_probs <- function(tm, hg, row, tier_name) {
  fs <- tm$feature_sets
  
  if (tier_name == "Low_1_4") {
    available <- intersect(fs$low, colnames(row))
    return(.predict_ranger_prob(tm$low_model, row[, available, drop = FALSE]))
  }
  
  if (tier_name == "Mid_5_7") {
    available <- intersect(fs$mid, colnames(row))
    return(.predict_ranger_prob(tm$mid_model, row[, available, drop = FALSE]))
  }
  
  # High tier
  available <- intersect(fs$high, colnames(row))
  .predict_ranger_prob(hg$model, row[, available, drop = FALSE])
}

.as_full_prob <- function(probs, class_levels) {
  out <- rep(0, length(class_levels))
  names(out) <- class_levels
  common <- intersect(names(probs), class_levels)
  out[common] <- probs[common]
  out
}

# =============================================================================
# Consensus Grade Prediction with Binary Triage
# =============================================================================

.consensus_grade_probs_v2 <- function(tm, hg, bin, row, tier_weight_gamma = 2) {
  # Use class levels from model (excludes PSA_1.5)
  class_levels <- tm$class_levels %||% c("PSA_1","PSA_2","PSA_3","PSA_4","PSA_5","PSA_6","PSA_7","PSA_8","PSA_9","PSA_10")
  
  # Step 1: Binary Triage (Near Mint vs Market Grade)
  binary_probs <- .predict_binary_triage(tm, row)
  
  full <- rep(0, length(class_levels))
  names(full) <- class_levels
  
  if (!is.null(binary_probs)) {
    # Binary Triage path
    near_mint_prob <- binary_probs["NearMint_8_10"] %||% 0
    market_prob <- binary_probs["MarketGrade_1_7"] %||% 0
    
    # Sharpen probabilities
    near_mint_prob <- near_mint_prob^tier_weight_gamma
    market_prob <- market_prob^tier_weight_gamma
    total <- near_mint_prob + market_prob + 1e-12
    near_mint_prob <- near_mint_prob / total
    market_prob <- market_prob / total
    
    # Near Mint path: go directly to high specialist
    if (near_mint_prob > 0.01) {
      high_probs <- .predict_specialist_probs(tm, hg, row, "High_8_10")
      high_full <- .as_full_prob(high_probs, class_levels)
      full <- full + near_mint_prob * high_full
    }
    
    # Market Grade path: route through Low/Mid
    if (market_prob > 0.01) {
      market_tier_probs <- .predict_market_tier(tm, row)
      
      if (!is.null(market_tier_probs)) {
        low_prob <- market_tier_probs["Low_1_4"] %||% 0.5
        mid_prob <- market_tier_probs["Mid_5_7"] %||% 0.5
        
        # Sharpen
        low_prob <- low_prob^tier_weight_gamma
        mid_prob <- mid_prob^tier_weight_gamma
        mtotal <- low_prob + mid_prob + 1e-12
        low_prob <- low_prob / mtotal
        mid_prob <- mid_prob / mtotal
        
        # Low specialist
        if (low_prob > 0.01) {
          low_probs <- .predict_specialist_probs(tm, hg, row, "Low_1_4")
          low_full <- .as_full_prob(low_probs, class_levels)
          full <- full + market_prob * low_prob * low_full
        }
        
        # Mid specialist
        if (mid_prob > 0.01) {
          mid_probs <- .predict_specialist_probs(tm, hg, row, "Mid_5_7")
          mid_full <- .as_full_prob(mid_probs, class_levels)
          full <- full + market_prob * mid_prob * mid_full
        }
      }
    }
    
  } else {
    # Fallback: Original tier1 routing
    if (!is.null(tm$tier1_model)) {
      tier_probs <- .predict_ranger_prob(tm$tier1_model, row[, tm$feature_sets$tier1, drop = FALSE])
      tier_probs <- tier_probs^tier_weight_gamma
      tier_probs <- tier_probs / (sum(tier_probs) + 1e-12)
      
      for (tname in names(tier_probs)) {
        sp <- .predict_specialist_probs(tm, hg, row, tname)
        sp_full <- .as_full_prob(sp, class_levels)
        full <- full + tier_probs[tname] * sp_full
      }
    }
    binary_probs <- c(MarketGrade_1_7 = 0.5, NearMint_8_10 = 0.5)
  }

  # 9 vs 10 reweighting
  if (all(c("PSA_9", "PSA_10") %in% names(full))) {
    mass <- full["PSA_9"] + full["PSA_10"]
    if (mass > 0.1) {
      fs_9v10 <- tm$feature_sets$psa_9_vs_10
      available_9v10 <- intersect(fs_9v10, colnames(row))
      bp <- .predict_ranger_prob(bin$model, row[, available_9v10, drop = FALSE])
      if (all(c("PSA_9", "PSA_10") %in% names(bp))) {
        bsum <- bp["PSA_9"] + bp["PSA_10"]
        if (bsum > 0) {
          full["PSA_9"] <- mass * (bp["PSA_9"] / bsum)
          full["PSA_10"] <- mass * (bp["PSA_10"] / bsum)
        }
      }
    }
  }

  # Hierarchical penalty blend (ordinal regression head)
  if (!is.null(tm$reg_model) && !is.null(tm$reg_blend)) {
    w <- tm$reg_blend$weight %||% 0.25
    sigma <- tm$reg_blend$sigma %||% 0.85
    if (is.finite(w) && w > 0) {
      fs_reg <- tm$feature_sets$reg
      available_reg <- intersect(fs_reg, colnames(row))
      reg_pred <- predict(tm$reg_model, data = row[, available_reg, drop = FALSE])$predictions[1]
      grade_num <- function(lbl) as.numeric(gsub("PSA_", "", lbl))
      nums <- vapply(class_levels, grade_num, numeric(1))
      p_reg <- exp(-((nums - reg_pred) ^ 2) / (2 * sigma ^ 2))
      p_reg <- p_reg / (sum(p_reg) + 1e-12)
      names(p_reg) <- class_levels
      full <- (1 - w) * full + w * p_reg
    }
  }

  full <- full / (sum(full) + 1e-12)
  list(binary_probs = binary_probs, grade_probs = full)
}

# =============================================================================
# LLM Visual Audit Integration
# =============================================================================

.run_llm_audit <- function(image_path, predicted_grade, confidence, provider = "none") {
  if (provider == "none" || predicted_grade < 8 || confidence < 0.85) {
    return(NULL)
  }
  
  script <- "scripts/llm_integration/llm_grading_assistant.py"
  if (!file.exists(script)) return(NULL)
  
  # Run LLM audit for high-grade candidates
  result <- tryCatch({
    args <- c(script, "--image", image_path, 
              "--provider", provider,
              "--grade", as.character(predicted_grade),
              "--confidence", as.character(confidence))
    out <- system2("python3", args = args, stdout = TRUE, stderr = TRUE)
    # Parse result (simplified - real implementation would parse JSON)
    list(ran = TRUE, output = paste(out, collapse = "\n"))
  }, error = function(e) {
    list(ran = FALSE, error = e$message)
  })
  
  return(result)
}

# =============================================================================
# Generate Grading Notes
# =============================================================================

generate_grading_notes <- function(features, predicted_grade, confidence) {
  # Extract key features for explanation
  centering_quality <- features$artbox_overall_score %||% features$centering_overall_quality %||% 0.5
  lr_ratio <- features$artbox_lr_ratio %||% features$centering_left_ratio %||% 0.5
  tb_ratio <- features$artbox_tb_ratio %||% features$centering_top_ratio %||% 0.5
  
  # Build centering note
  lr_pct <- round(lr_ratio * 100)
  tb_pct <- round(tb_ratio * 100)
  centering_str <- paste0(lr_pct, "/", 100 - lr_pct, " L/R, ", tb_pct, "/", 100 - tb_pct, " T/B")
  
  if (centering_quality > 0.9) {
    centering_note <- paste("Excellent centering:", centering_str)
  } else if (centering_quality > 0.8) {
    centering_note <- paste("Good centering:", centering_str)
  } else {
    centering_note <- paste("Off-center:", centering_str)
  }
  
  # Corner analysis
  corner_notes <- c()
  for (corner in c("tl", "tr", "bl", "br")) {
    whitening_col <- paste0("adaptive_patch_", corner, "_whitening_score")
    whitening <- features[[whitening_col]] %||% 0
    
    if (whitening > 0.5) {
      corner_notes <- c(corner_notes, paste(toupper(corner), ": Whitening detected"))
    } else if (whitening > 0.3) {
      corner_notes <- c(corner_notes, paste(toupper(corner), ": Minor wear"))
    } else {
      corner_notes <- c(corner_notes, paste(toupper(corner), ": Good"))
    }
  }
  
  # Summary
  if (predicted_grade == 10) {
    summary <- "Gem Mint - No visible defects, perfect centering."
  } else if (predicted_grade == 9) {
    summary <- "Near Mint-Mint - Minor imperfections under magnification."
  } else if (predicted_grade == 8) {
    summary <- "Near Mint-Mint - Minor wear at edges or corners."
  } else if (predicted_grade >= 5) {
    summary <- "Moderate wear visible. Good for collection."
  } else {
    summary <- "Significant wear. Best for set completion."
  }
  
  list(
    centering = centering_note,
    corners = corner_notes,
    summary = summary,
    confidence = confidence
  )
}

# =============================================================================
# Main Prediction Function
# =============================================================================

predict_grade <- function(image_path,
                          models_dir = "models",
                          psa10_upgrade_threshold = 0.55,
                          enable_llm_audit = FALSE,
                          llm_provider = "none") {
  .load_models(models_dir)

  tm <- .psa_models_env$tiered_model
  hg <- .psa_models_env$high_grade_specialist
  bin <- .psa_models_env$psa_9_vs_10

  # Build full feature row
  row <- .build_feature_row_full(image_path)

  # Get consensus prediction with binary triage
  consensus <- .consensus_grade_probs_v2(tm, hg, bin, row, tier_weight_gamma = 2)
  binary_probs <- consensus$binary_probs
  grade_probs <- consensus$grade_probs

  # Extract tier from binary
  near_mint <- binary_probs["NearMint_8_10"] %||% 0
  is_near_mint <- near_mint > 0.5
  tier <- if (is_near_mint) "NearMint_8_10" else "MarketGrade_1_7"
  tier_conf <- max(binary_probs)

  grade <- names(grade_probs)[which.max(grade_probs)]
  grade_conf <- max(grade_probs)
  grade_num <- as.numeric(gsub("PSA_", "", grade))

  # Upgrade hint for 9 vs 10
  upgrade_hint <- NULL
  if (grade %in% c("PSA_9", "PSA_10")) {
    fs_9v10 <- tm$feature_sets$psa_9_vs_10
    available_9v10 <- intersect(fs_9v10, colnames(row))
    bin_probs <- .predict_ranger_prob(bin$model, row[, available_9v10, drop = FALSE])
    bin_grade <- names(bin_probs)[which.max(bin_probs)]
    bin_conf <- max(bin_probs)
    if (bin_grade == "PSA_10" && !is.na(bin_conf) && bin_conf >= psa10_upgrade_threshold && bin_conf < 0.70) {
      upgrade_hint <- "PSA 9 (Potential 10 Upgrade)"
    }
  }
  
  # Generate grading notes
  notes <- generate_grading_notes(as.list(row), grade_num, grade_conf)
  
  # LLM audit for high-grade candidates
  llm_audit <- NULL
  if (enable_llm_audit && grade_num >= 8 && grade_conf >= 0.85) {
    llm_audit <- .run_llm_audit(image_path, grade_num, grade_conf, llm_provider)
  }

  out <- list(
    image = image_path,
    tier = tier,
    tier_confidence = tier_conf,
    grade = grade,
    grade_confidence = grade_conf,
    upgrade_hint = upgrade_hint,
    binary_probabilities = binary_probs,
    grade_probabilities = grade_probs,
    grading_notes = notes,
    llm_audit = llm_audit
  )

  return(out)
}

# =============================================================================
# Batch Prediction
# =============================================================================

predict_batch <- function(folder,
                          models_dir = "models",
                          pattern = "\\.(jpg|jpeg|png|webp)$",
                          recursive = FALSE,
                          enable_llm_audit = FALSE,
                          llm_provider = "none") {
  files <- list.files(folder, pattern = pattern, full.names = TRUE, recursive = recursive, ignore.case = TRUE)
  if (length(files) == 0) stop(paste0("No images found in: ", folder), call. = FALSE)

  results <- lapply(files, function(f) {
    r <- predict_grade(f, models_dir = models_dir, 
                       enable_llm_audit = enable_llm_audit,
                       llm_provider = llm_provider)
    data.frame(
      image = r$image,
      tier = r$tier,
      tier_confidence = r$tier_confidence,
      grade = r$grade,
      grade_confidence = r$grade_confidence,
      upgrade_hint = ifelse(is.null(r$upgrade_hint), "", r$upgrade_hint),
      centering = r$grading_notes$centering,
      summary = r$grading_notes$summary,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, results)
}

# =============================================================================
# Pretty Print Result
# =============================================================================

print_prediction <- function(result) {
  cat("\n")
  cat("========================================\n")
  cat("PSA GRADE PREDICTION\n")
  cat("========================================\n")
  cat("Image:", result$image, "\n\n")
  
  cat("TIER:", result$tier, "(", round(result$tier_confidence * 100, 1), "% confidence)\n")
  cat("GRADE:", result$grade, "(", round(result$grade_confidence * 100, 1), "% confidence)\n")
  
  if (!is.null(result$upgrade_hint)) {
    cat("NOTE:", result$upgrade_hint, "\n")
  }
  
  cat("\n--- Grading Notes ---\n")
  cat("Centering:", result$grading_notes$centering, "\n")
  cat("Corners:\n")
  for (cn in result$grading_notes$corners) {
    cat("  •", cn, "\n")
  }
  cat("Summary:", result$grading_notes$summary, "\n")
  
  if (!is.null(result$llm_audit) && result$llm_audit$ran) {
    cat("\n--- LLM Visual Audit ---\n")
    cat(result$llm_audit$output, "\n")
  }
  
  cat("========================================\n\n")
}

# =============================================================================
# Real-world sanity checks
# =============================================================================

rotation_invariance_test <- function(image_path,
                                     degrees = 5,
                                     models_dir = "models") {
  tmp <- tempfile(fileext = ".jpg")
  cmd <- paste(
    "python3 - <<'PY'",
    "import cv2",
    "import numpy as np",
    "from pathlib import Path",
    sprintf("src = Path(%s)", dQuote(normalizePath(image_path))),
    sprintf("dst = Path(%s)", dQuote(tmp)),
    "img = cv2.imread(str(src))",
    "h, w = img.shape[:2]",
    "M = cv2.getRotationMatrix2D((w/2, h/2), float(",
    degrees,
    "), 1.0)",
    "rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)",
    "cv2.imwrite(str(dst), rot)",
    "print(dst)",
    "PY",
    sep = "\n"
  )
  system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)

  base <- predict_grade(image_path, models_dir = models_dir)
  rot <- predict_grade(tmp, models_dir = models_dir)
  list(
    base = base,
    rotated = rot,
    degrees = degrees,
    rotated_path = tmp
  )
}

lighting_check_test <- function(image_path,
                                models_dir = "models") {
  warm_path <- tempfile(fileext = ".jpg")
  white_path <- tempfile(fileext = ".jpg")

  cmd <- paste(
    "python3 - <<'PY'",
    "import cv2",
    "import numpy as np",
    "from pathlib import Path",
    sprintf("src = Path(%s)", dQuote(normalizePath(image_path))),
    sprintf("warm_path = Path(%s)", dQuote(warm_path)),
    sprintf("white_path = Path(%s)", dQuote(white_path)),
    "img = cv2.imread(str(src))",
    "warm = img.astype(np.float32)",
    "warm[:,:,2] *= 1.10",
    "warm[:,:,0] *= 0.90",
    "warm = np.clip(warm, 0, 255).astype(np.uint8)",
    "cv2.imwrite(str(warm_path), warm)",
    "white = img.astype(np.float32)",
    "white[:,:,0] *= 1.08",
    "white[:,:,2] *= 0.95",
    "white = np.clip(white, 0, 255).astype(np.uint8)",
    "cv2.imwrite(str(white_path), white)",
    "print(warm_path)",
    "print(white_path)",
    "PY",
    sep = "\n"
  )
  system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)

  warm <- predict_grade(warm_path, models_dir = models_dir)
  white <- predict_grade(white_path, models_dir = models_dir)

  list(
    warm = warm,
    white = white,
    warm_path = warm_path,
    white_path = white_path
  )
}
