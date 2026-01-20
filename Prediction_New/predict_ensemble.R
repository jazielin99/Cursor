# ============================================
# Ensemble Prediction with TTA and Calibration
# ============================================
# High-accuracy prediction using:
# 1. 5-model ensemble with weighted averaging
# 2. Confusion-pair specialists for boundary cases
# 3. Per-tier temperature calibration
# 4. Ordinal post-processing (prefer adjacent errors)
# 5. Optional TTA (test-time augmentation)
# ============================================

suppressPackageStartupMessages({
  library(ranger)
})

.ensemble_env <- new.env(parent = emptyenv())

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

# =============================================================================
# LOAD MODELS
# =============================================================================

.load_ensemble <- function(models_dir = "models") {
  if (exists("ensemble_model", envir = .ensemble_env, inherits = FALSE)) {
    return(invisible(TRUE))
  }
  
  ensemble_path <- file.path(models_dir, "ensemble_model.rds")
  .require_file(ensemble_path)
  
  .ensemble_env$ensemble_model <- readRDS(ensemble_path)
  
  cat("Loaded ensemble model:\n")
  cat("  - Models:", length(.ensemble_env$ensemble_model$ensemble_models), "\n")
  cat("  - Pair specialists:", length(.ensemble_env$ensemble_model$pair_specialists), "\n")
  cat("  - Calibration temp:", .ensemble_env$ensemble_model$calibration$overall_temperature, "\n")
  
  invisible(TRUE)
}

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

.extract_features <- function(image_path, use_tta = FALSE) {
  tmp_dir <- tempdir()
  
  if (use_tta) {
    # Use TTA extraction
    script <- "scripts/feature_extraction/extract_features_tta.py"
    if (!file.exists(script)) {
      warning("TTA script not found, falling back to standard extraction")
      use_tta <- FALSE
    } else {
      out_csv <- file.path(tmp_dir, "features_tta.csv")
      .run_python(c(script, "--image", image_path, "--output-csv", out_csv))
      feats <- read.csv(out_csv, check.names = FALSE)
      feats$path <- NULL
      feats$tta_count <- NULL
      feats$tta_stability <- NULL
      return(feats)
    }
  }
  
  # Standard extraction
  script <- "scripts/feature_extraction/extract_advanced_features.py"
  if (!file.exists(script)) {
    stop("Feature extraction script not found", call. = FALSE)
  }
  
  out_csv <- file.path(tmp_dir, "features.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", out_csv))
  feats <- read.csv(out_csv, check.names = FALSE)
  feats$path <- NULL
  
  # Also try CNN features
  cnn_script <- "scripts/feature_extraction/extract_cnn_features_single.py"
  if (file.exists(cnn_script)) {
    cnn_csv <- file.path(tmp_dir, "cnn.csv")
    tryCatch({
      .run_python(c(cnn_script, "--image", image_path, "--output-csv", cnn_csv))
      cnn <- read.csv(cnn_csv, check.names = FALSE)
      cnn$path <- NULL
      feats <- cbind(feats, cnn)
    }, error = function(e) {
      warning("CNN extraction failed: ", e$message)
    })
  }
  
  feats
}

# =============================================================================
# ENSEMBLE PREDICTION
# =============================================================================

.predict_ensemble <- function(row_df, em) {
  # Get predictions from each model in ensemble
  all_probs <- list()
  
  for (name in names(em$ensemble_models)) {
    model <- em$ensemble_models[[name]]
    features <- em$ensemble_features[[name]]
    
    available <- intersect(features, colnames(row_df))
    if (length(available) < length(features) * 0.7) {
      warning(sprintf("Model %s: only %d/%d features available", name, length(available), length(features)))
      next
    }
    
    pred <- predict(model, data = row_df[, available, drop = FALSE])$predictions
    all_probs[[name]] <- pred[1, ]
  }
  
  if (length(all_probs) == 0) {
    stop("No ensemble models could make predictions", call. = FALSE)
  }
  
  # Average probabilities
  all_classes <- em$class_levels
  avg_probs <- setNames(rep(0, length(all_classes)), all_classes)
  
  for (probs in all_probs) {
    for (cls in names(probs)) {
      if (cls %in% names(avg_probs)) {
        avg_probs[cls] <- avg_probs[cls] + probs[cls]
      }
    }
  }
  
  avg_probs / length(all_probs)
}

.apply_calibration <- function(probs, temperature) {
  # Temperature scaling
  log_probs <- log(probs + 1e-10)
  scaled <- exp(log_probs / temperature)
  scaled / sum(scaled)
}

.apply_ordinal_postprocess <- function(probs, cost_matrix, weight = 0.3) {
  classes <- names(probs)
  adjusted <- probs
  
  for (cls in classes) {
    if (cls %in% rownames(cost_matrix)) {
      # Penalty for predicting this class based on costs
      cost_penalty <- sum(probs * cost_matrix[, cls])
      adjusted[cls] <- probs[cls] * exp(-weight * cost_penalty)
    }
  }
  
  adjusted / sum(adjusted)
}

.apply_pair_specialists <- function(probs, row_df, em) {
  # If prediction is near a confusion boundary, consult specialist
  top_pred <- names(probs)[which.max(probs)]
  top_prob <- max(probs)
  
  # Only apply if confidence is borderline (30-70%)
  if (top_prob > 0.7 || top_prob < 0.3) {
    return(probs)
  }
  
  grade_num <- as.numeric(gsub("PSA_", "", top_pred))
  
  # Check adjacent grades
  adjacent_pairs <- list(
    c(grade_num - 1, grade_num),
    c(grade_num, grade_num + 1)
  )
  
  for (pair in adjacent_pairs) {
    if (any(pair < 1) || any(pair > 10)) next
    
    pair_name <- paste0("PSA_", pair[1], "_vs_", pair[2])
    
    if (pair_name %in% names(em$pair_specialists)) {
      specialist <- em$pair_specialists[[pair_name]]
      available <- intersect(specialist$features, colnames(row_df))
      
      if (length(available) >= length(specialist$features) * 0.7) {
        sp_pred <- predict(specialist$model, data = row_df[, available, drop = FALSE])$predictions[1, ]
        
        # Blend specialist prediction for these two grades
        g1 <- paste0("PSA_", pair[1])
        g2 <- paste0("PSA_", pair[2])
        
        if (g1 %in% names(sp_pred) && g2 %in% names(sp_pred)) {
          mass <- probs[g1] + probs[g2]
          if (mass > 0.1) {
            sp_sum <- sp_pred[g1] + sp_pred[g2]
            if (sp_sum > 0) {
              # Blend 50/50 with specialist
              probs[g1] <- mass * (0.5 * (probs[g1]/mass) + 0.5 * (sp_pred[g1]/sp_sum))
              probs[g2] <- mass * (0.5 * (probs[g2]/mass) + 0.5 * (sp_pred[g2]/sp_sum))
            }
          }
        }
      }
    }
  }
  
  probs / sum(probs)
}

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

predict_grade_ensemble <- function(
  image_path,
  models_dir = "models",
  use_tta = FALSE,
  apply_calibration = TRUE,
  apply_ordinal = TRUE,
  use_specialists = TRUE
) {
  .load_ensemble(models_dir)
  em <- .ensemble_env$ensemble_model
  
  # Extract features
  row_df <- .extract_features(image_path, use_tta = use_tta)
  
  # Get ensemble prediction
  probs <- .predict_ensemble(row_df, em)
  
  # Apply calibration
  if (apply_calibration && !is.null(em$calibration)) {
    temp <- em$calibration$overall_temperature
    probs <- .apply_calibration(probs, temp)
  }
  
  # Apply pair specialists for boundary cases
  if (use_specialists && !is.null(em$pair_specialists)) {
    probs <- .apply_pair_specialists(probs, row_df, em)
  }
  
  # Apply ordinal post-processing
  if (apply_ordinal && !is.null(em$ordinal_cost_matrix)) {
    probs <- .apply_ordinal_postprocess(probs, em$ordinal_cost_matrix, em$ordinal_weight)
  }
  
  # Get final prediction
  grade <- names(probs)[which.max(probs)]
  confidence <- max(probs)
  
  # Determine tier
  grade_num <- as.numeric(gsub("PSA_", "", grade))
  if (grade_num >= 8) {
    tier <- "NearMint_8_10"
  } else if (grade_num >= 5) {
    tier <- "Mid_5_7"
  } else {
    tier <- "Low_1_4"
  }
  
  # Check for upgrade potential
  upgrade_hint <- NULL
  if (grade == "PSA_9" && probs["PSA_10"] > 0.2) {
    upgrade_hint <- "Potential 10 - borderline case"
  } else if (grade == "PSA_10" && probs["PSA_9"] > 0.3) {
    upgrade_hint <- "Strong 10 candidate, but verify corners"
  }
  
  list(
    image = image_path,
    grade = grade,
    grade_confidence = confidence,
    tier = tier,
    grade_probabilities = probs,
    upgrade_hint = upgrade_hint,
    tta_used = use_tta,
    ensemble_size = length(em$ensemble_models)
  )
}

# =============================================================================
# BATCH PREDICTION
# =============================================================================

predict_batch_ensemble <- function(
  folder,
  models_dir = "models",
  pattern = "\\.(jpg|jpeg|png|webp)$",
  use_tta = FALSE
) {
  files <- list.files(folder, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
  if (length(files) == 0) stop(paste0("No images found in: ", folder), call. = FALSE)
  
  results <- lapply(files, function(f) {
    tryCatch({
      r <- predict_grade_ensemble(f, models_dir = models_dir, use_tta = use_tta)
      data.frame(
        image = r$image,
        grade = r$grade,
        confidence = r$grade_confidence,
        tier = r$tier,
        upgrade_hint = ifelse(is.null(r$upgrade_hint), "", r$upgrade_hint),
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      data.frame(
        image = f,
        grade = NA,
        confidence = NA,
        tier = NA,
        upgrade_hint = paste("Error:", e$message),
        stringsAsFactors = FALSE
      )
    })
  })
  
  do.call(rbind, results)
}

# =============================================================================
# PRINT RESULT
# =============================================================================

print_ensemble_prediction <- function(result) {
  cat("\n")
  cat("========================================\n")
  cat("ENSEMBLE PREDICTION\n")
  cat("========================================\n")
  cat("Image:", result$image, "\n\n")
  
  cat("GRADE:", result$grade, "\n")
  cat("Confidence:", sprintf("%.1f%%", result$grade_confidence * 100), "\n")
  cat("Tier:", result$tier, "\n")
  
  if (!is.null(result$upgrade_hint)) {
    cat("Note:", result$upgrade_hint, "\n")
  }
  
  cat("\nTop Probabilities:\n")
  sorted_probs <- sort(result$grade_probabilities, decreasing = TRUE)
  for (i in 1:min(5, length(sorted_probs))) {
    cat(sprintf("  %s: %.1f%%\n", names(sorted_probs)[i], sorted_probs[i] * 100))
  }
  
  cat("\nSettings:\n")
  cat("  Ensemble size:", result$ensemble_size, "models\n")
  cat("  TTA used:", result$tta_used, "\n")
  cat("========================================\n\n")
}

# =============================================================================
# CLI
# =============================================================================

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript predict_ensemble.R <image_path> [--tta]\n")
    quit(status = 1)
  }
  
  image_path <- args[1]
  use_tta <- "--tta" %in% args
  
  result <- predict_grade_ensemble(image_path, use_tta = use_tta)
  print_ensemble_prediction(result)
}
