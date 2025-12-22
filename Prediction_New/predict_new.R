# ============================================
# PSA Grade Prediction (Tiered System v2)
# - Auto-loads tiered + specialists
# - Uses Python for CNN + advanced feature extraction
# - Batch prediction support
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

.extract_advanced_v2 <- function(image_path, tmp_dir = tempdir(), script = "extract_advanced_features_v2.py") {
  adv_out <- file.path(tmp_dir, "adv_v2_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", adv_out))
  adv <- read.csv(adv_out, check.names = FALSE)
  adv$path <- NULL
  adv
}

.extract_advanced_v3 <- function(image_path, tmp_dir = tempdir(), script = "extract_advanced_features_v3.py") {
  adv_out <- file.path(tmp_dir, "adv_v3_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", adv_out))
  adv <- read.csv(adv_out, check.names = FALSE)
  adv$path <- NULL
  adv
}

.extract_cnn <- function(image_path, tmp_dir = tempdir(), script = "extract_cnn_features_single.py") {
  cnn_out <- file.path(tmp_dir, "cnn_single.csv")
  .run_python(c(script, "--image", image_path, "--output-csv", cnn_out))
  cnn <- read.csv(cnn_out, check.names = FALSE)
  cnn$path <- NULL
  cnn
}

.build_feature_row <- function(image_path, selected_features) {
  adv <- if (file.exists("extract_advanced_features_v3.py")) .extract_advanced_v3(image_path) else .extract_advanced_v2(image_path)

  # Try CNN; allow tiered model to be trained without it
  cnn <- NULL
  if (file.exists("extract_cnn_features_single.py")) {
    try(cnn <- .extract_cnn(image_path), silent = TRUE)
  }

  if (!is.null(cnn)) {
    row <- cbind(adv, cnn)
  } else {
    row <- adv
  }

  missing <- setdiff(selected_features, colnames(row))
  if (length(missing) > 0) {
    stop(paste0(
      "Feature mismatch: missing ", length(missing), " required features.\n",
      "Common cause: tiered model trained with CNN features, but CNN extraction isn't available at prediction time.\n",
      "Missing examples: ", paste(head(missing, 10), collapse = ", ")
    ), call. = FALSE)
  }

  row <- row[, selected_features, drop = FALSE]
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

.predict_specialist_probs <- function(tm, hg, row, tier_name) {
  if (tier_name == "Low_1_4") return(.predict_ranger_prob(tm$low_model, row))
  if (tier_name == "Mid_5_7") return(.predict_ranger_prob(tm$mid_model, row))
  .predict_ranger_prob(hg$model, row)
}

.as_full_prob <- function(probs, class_levels) {
  out <- rep(0, length(class_levels))
  names(out) <- class_levels
  common <- intersect(names(probs), class_levels)
  out[common] <- probs[common]
  out
}

.consensus_grade_probs <- function(tm, hg, bin, row, tier_weight_gamma = 2) {
  class_levels <- tm$class_levels %||% sort(unique(c("PSA_1","PSA_1.5","PSA_2","PSA_3","PSA_4","PSA_5","PSA_6","PSA_7","PSA_8","PSA_9","PSA_10")))

  tier_probs <- .predict_ranger_prob(tm$tier1_model, row)
  if (!is.null(tm$tier_levels)) {
    keep <- intersect(names(tier_probs), tm$tier_levels)
    tier_probs <- tier_probs[keep]
  }
  # Sharpen tier weights to reflect Tier-1 confidence
  tier_probs <- tier_probs^tier_weight_gamma
  tier_probs <- tier_probs / (sum(tier_probs) + 1e-12)

  full <- rep(0, length(class_levels))
  names(full) <- class_levels

  for (tname in names(tier_probs)) {
    sp <- .predict_specialist_probs(tm, hg, row, tname)
    sp_full <- .as_full_prob(sp, class_levels)
    full <- full + tier_probs[tname] * sp_full
  }

  # 9 vs 10 reweighting
  if (all(c("PSA_9", "PSA_10") %in% names(full))) {
    mass <- full["PSA_9"] + full["PSA_10"]
    if (mass > 0) {
      bp <- .predict_ranger_prob(bin$model, row)
      if (all(c("PSA_9", "PSA_10") %in% names(bp))) {
        bsum <- bp["PSA_9"] + bp["PSA_10"]
        if (bsum > 0) {
          full["PSA_9"] <- mass * (bp["PSA_9"] / bsum)
          full["PSA_10"] <- mass * (bp["PSA_10"] / bsum)
        }
      }
    }
  }

  full <- full / (sum(full) + 1e-12)
  list(tier_probs = tier_probs, grade_probs = full)
}

predict_grade <- function(image_path,
                          models_dir = "models",
                          psa10_upgrade_threshold = 0.55) {
  .load_models(models_dir)

  tm <- .psa_models_env$tiered_model
  hg <- .psa_models_env$high_grade_specialist
  bin <- .psa_models_env$psa_9_vs_10

  selected <- tm$selected_features
  row <- .build_feature_row(image_path, selected)

  consensus <- .consensus_grade_probs(tm, hg, bin, row, tier_weight_gamma = 2)
  tier_probs <- consensus$tier_probs
  grade_probs <- consensus$grade_probs

  tier <- names(tier_probs)[which.max(tier_probs)]
  tier_conf <- max(tier_probs)

  grade <- names(grade_probs)[which.max(grade_probs)]
  grade_conf <- max(grade_probs)

  upgrade_hint <- NULL
  if (grade %in% c("PSA_9", "PSA_10")) {
    bin_probs <- .predict_ranger_prob(bin$model, row)
    bin_grade <- names(bin_probs)[which.max(bin_probs)]
    bin_conf <- max(bin_probs)
    if (bin_grade == "PSA_10" && !is.na(bin_conf) && bin_conf >= psa10_upgrade_threshold && bin_conf < 0.70) {
      upgrade_hint <- "PSA 9 (Potential 10 Upgrade)"
    }
  }

  out <- list(
    image = image_path,
    tier = tier,
    tier_confidence = tier_conf,
    grade = grade,
    grade_confidence = grade_conf,
    upgrade_hint = upgrade_hint,
    tier_probabilities = tier_probs,
    grade_probabilities = grade_probs
  )

  return(out)
}

predict_batch <- function(folder,
                          models_dir = "models",
                          pattern = "\\.(jpg|jpeg|png|webp)$",
                          recursive = FALSE) {
  files <- list.files(folder, pattern = pattern, full.names = TRUE, recursive = recursive, ignore.case = TRUE)
  if (length(files) == 0) stop(paste0("No images found in: ", folder), call. = FALSE)

  results <- lapply(files, function(f) {
    r <- predict_grade(f, models_dir = models_dir)
    data.frame(
      image = r$image,
      tier = r$tier,
      tier_confidence = r$tier_confidence,
      grade = r$grade,
      grade_confidence = r$grade_confidence,
      upgrade_hint = ifelse(is.null(r$upgrade_hint), "", r$upgrade_hint),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, results)
}

# --------------------------------------------
# Real-world sanity checks (recommended)
# --------------------------------------------

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

