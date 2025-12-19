# ==============================================================================
# PSA Slab Card Extraction Tool
# ==============================================================================
# Automatically detects PSA slabs and extracts just the card portion,
# removing the grade label to prevent data leakage during training.

library(magick)

#' Detect if an image contains a PSA slab and extract the card
#' @param image_path Path to the image file
#' @param output_path Path to save the cropped image (NULL = overwrite original)
#' @param backup Whether to create a backup of the original
#' @return TRUE if successfully processed, FALSE otherwise
extract_card_from_slab <- function(image_path, output_path = NULL, backup = TRUE) {
  
  tryCatch({
    # Read image
    img <- image_read(image_path)
    info <- image_info(img)
    
    orig_width <- info$width
    orig_height <- info$height
    
    # PSA slabs are typically portrait orientation (taller than wide)
    # The card is in the lower ~75-80% of the slab
    # The label is in the top ~20-25%
    
    aspect_ratio <- orig_height / orig_width
    
    # Check if this looks like a slabbed card (portrait, aspect ratio ~1.3-1.8)
    is_likely_slab <- aspect_ratio > 1.2 && aspect_ratio < 2.0
    
    if (is_likely_slab) {
      # PSA slab layout:
      # - Top ~18-22%: Label area (grade, cert number, etc.)
      # - Bottom ~78-82%: Card area
      # - Small margins on sides (~3-5%)
      
      # Calculate crop region for the card
      # Remove top 22% (label) and small margins on sides
      top_crop_pct <- 0.22
      side_margin_pct <- 0.04
      bottom_margin_pct <- 0.02
      
      crop_x <- floor(orig_width * side_margin_pct)
      crop_y <- floor(orig_height * top_crop_pct)
      crop_width <- floor(orig_width * (1 - 2 * side_margin_pct))
      crop_height <- floor(orig_height * (1 - top_crop_pct - bottom_margin_pct))
      
      # Crop the image
      geometry <- sprintf("%dx%d+%d+%d", crop_width, crop_height, crop_x, crop_y)
      img_cropped <- image_crop(img, geometry)
      
    } else {
      # Not a slab - might be raw card or different format
      # Try to detect card edges using edge detection
      
      # Convert to grayscale and detect edges
      img_gray <- image_convert(img, colorspace = "gray")
      img_edges <- image_edge(img_gray, radius = 1)
      
      # For non-slab images, apply minimal cropping to remove any borders
      margin_pct <- 0.02
      crop_x <- floor(orig_width * margin_pct)
      crop_y <- floor(orig_height * margin_pct)
      crop_width <- floor(orig_width * (1 - 2 * margin_pct))
      crop_height <- floor(orig_height * (1 - 2 * margin_pct))
      
      geometry <- sprintf("%dx%d+%d+%d", crop_width, crop_height, crop_x, crop_y)
      img_cropped <- image_crop(img, geometry)
    }
    
    # Backup original if requested
    if (backup) {
      backup_dir <- file.path(dirname(image_path), "originals_backup")
      if (!dir.exists(backup_dir)) {
        dir.create(backup_dir, recursive = TRUE)
      }
      file.copy(image_path, file.path(backup_dir, basename(image_path)), overwrite = FALSE)
    }
    
    # Save cropped image
    if (is.null(output_path)) {
      output_path <- image_path
    }
    
    image_write(img_cropped, output_path, quality = 95)
    
    return(TRUE)
    
  }, error = function(e) {
    warning(paste("Error processing", image_path, ":", e$message))
    return(FALSE)
  })
}

#' Advanced card extraction with edge detection
#' @param image_path Path to the image file
#' @param output_path Path to save the cropped image
#' @param backup Whether to backup original
#' @param visualize Whether to show the detection result
#' @return TRUE if successfully processed
extract_card_smart <- function(image_path, output_path = NULL, backup = TRUE, visualize = FALSE) {
  
  tryCatch({
    img <- image_read(image_path)
    info <- image_info(img)
    
    orig_width <- info$width
    orig_height <- info$height
    aspect_ratio <- orig_height / orig_width
    
    # Determine crop strategy based on image characteristics
    
    if (aspect_ratio > 1.3 && aspect_ratio < 1.9) {
      # Likely a PSA slab (portrait orientation)
      # Standard PSA slab has label at top ~20%
      
      # Try to detect the label/card boundary by looking for horizontal edges
      img_gray <- image_convert(img, colorspace = "gray")
      
      # Sample the brightness at different heights to find the label boundary
      # The label area is usually brighter/different from the card
      
      # Use standard PSA proportions
      label_height_pct <- 0.20  # Label is roughly 20% of slab height
      
      # Add small margins
      margin_pct <- 0.03
      
      crop_x <- floor(orig_width * margin_pct)
      crop_y <- floor(orig_height * label_height_pct)
      crop_width <- floor(orig_width * (1 - 2 * margin_pct))
      crop_height <- floor(orig_height * (1 - label_height_pct - margin_pct))
      
    } else if (aspect_ratio > 1.9) {
      # Very tall image - might be a slab photographed with extra space
      # Crop more aggressively from top
      
      label_height_pct <- 0.25
      margin_pct <- 0.05
      
      crop_x <- floor(orig_width * margin_pct)
      crop_y <- floor(orig_height * label_height_pct)
      crop_width <- floor(orig_width * (1 - 2 * margin_pct))
      crop_height <- floor(orig_height * (0.7))  # Take middle 70%
      
    } else if (aspect_ratio > 0.6 && aspect_ratio <= 1.3) {
      # Roughly square or landscape - might be just the card
      # Minimal cropping
      
      margin_pct <- 0.02
      crop_x <- floor(orig_width * margin_pct)
      crop_y <- floor(orig_height * margin_pct)
      crop_width <- floor(orig_width * (1 - 2 * margin_pct))
      crop_height <- floor(orig_height * (1 - 2 * margin_pct))
      
    } else {
      # Unusual aspect ratio - use center crop
      margin_pct <- 0.05
      crop_x <- floor(orig_width * margin_pct)
      crop_y <- floor(orig_height * margin_pct)
      crop_width <- floor(orig_width * (1 - 2 * margin_pct))
      crop_height <- floor(orig_height * (1 - 2 * margin_pct))
    }
    
    # Ensure valid crop dimensions
    crop_x <- max(0, crop_x)
    crop_y <- max(0, crop_y)
    crop_width <- min(crop_width, orig_width - crop_x)
    crop_height <- min(crop_height, orig_height - crop_y)
    
    # Perform crop
    geometry <- sprintf("%dx%d+%d+%d", crop_width, crop_height, crop_x, crop_y)
    img_cropped <- image_crop(img, geometry)
    
    if (visualize) {
      # Show original with crop region marked
      img_annotated <- image_draw(img)
      rect(crop_x, crop_y, crop_x + crop_width, crop_y + crop_height, 
           border = "red", lwd = 3)
      dev.off()
      print(img_annotated)
      cat("\nPress Enter to continue...")
      readline()
    }
    
    # Backup original if requested
    if (backup) {
      backup_dir <- file.path(dirname(image_path), "originals_backup")
      if (!dir.exists(backup_dir)) {
        dir.create(backup_dir, recursive = TRUE)
      }
      file.copy(image_path, file.path(backup_dir, basename(image_path)), overwrite = FALSE)
    }
    
    # Save
    if (is.null(output_path)) {
      output_path <- image_path
    }
    
    image_write(img_cropped, output_path, quality = 95)
    
    return(TRUE)
    
  }, error = function(e) {
    warning(paste("Error:", e$message))
    return(FALSE)
  })
}

#' Process all images in the training directory
#' @param training_dir Path to training directory
#' @param backup Whether to backup originals
#' @param method Extraction method ("standard" or "smart")
process_all_training_images <- function(training_dir = "data/training",
                                         backup = TRUE,
                                         method = "smart") {
  
  cat("========================================\n")
  cat("PSA Slab Card Extraction Tool\n")
  cat("========================================\n\n")
  
  # Get all grade folders
  grade_folders <- list.dirs(training_dir, recursive = FALSE, full.names = TRUE)
  
  total_processed <- 0
  total_failed <- 0
  
  for (folder in grade_folders) {
    grade_name <- basename(folder)
    
    # Get all images in this folder
    images <- list.files(folder, pattern = "\\.(jpg|jpeg|png|PNG|JPG|JPEG)$",
                         full.names = TRUE, ignore.case = TRUE)
    
    # Skip backup folder
    images <- images[!grepl("originals_backup", images)]
    
    if (length(images) == 0) {
      next
    }
    
    cat("\nProcessing", grade_name, "(", length(images), "images )...\n")
    
    processed <- 0
    failed <- 0
    
    for (img_path in images) {
      cat("  ", basename(img_path), "... ")
      
      success <- if (method == "smart") {
        extract_card_smart(img_path, backup = backup)
      } else {
        extract_card_from_slab(img_path, backup = backup)
      }
      
      if (success) {
        cat("OK\n")
        processed <- processed + 1
      } else {
        cat("FAILED\n")
        failed <- failed + 1
      }
    }
    
    cat("  Processed:", processed, "| Failed:", failed, "\n")
    total_processed <- total_processed + processed
    total_failed <- total_failed + failed
  }
  
  cat("\n========================================\n")
  cat("COMPLETE\n")
  cat("========================================\n")
  cat("Total processed:", total_processed, "\n")
  cat("Total failed:", total_failed, "\n")
  
  if (backup) {
    cat("\nOriginal images backed up to 'originals_backup' folders\n")
  }
  
  return(list(processed = total_processed, failed = total_failed))
}

#' Preview crop on a single image without saving
#' @param image_path Path to image
preview_crop <- function(image_path) {
  
  img <- image_read(image_path)
  info <- image_info(img)
  
  cat("Image:", basename(image_path), "\n")
  cat("Dimensions:", info$width, "x", info$height, "\n")
  cat("Aspect ratio:", round(info$height / info$width, 2), "\n")
  
  # Calculate crop
  aspect_ratio <- info$height / info$width
  
  if (aspect_ratio > 1.3 && aspect_ratio < 1.9) {
    cat("Detected: PSA Slab (will crop top 20% label)\n")
    label_pct <- 0.20
  } else if (aspect_ratio > 1.9) {
    cat("Detected: Tall slab image (will crop top 25%)\n")
    label_pct <- 0.25
  } else {
    cat("Detected: Raw card or non-standard (minimal crop)\n")
    label_pct <- 0.02
  }
  
  margin_pct <- 0.03
  
  crop_y <- floor(info$height * label_pct)
  crop_x <- floor(info$width * margin_pct)
  crop_width <- floor(info$width * (1 - 2 * margin_pct))
  crop_height <- floor(info$height * (1 - label_pct - margin_pct))
  
  cat("\nCrop region:\n")
  cat("  Top:", crop_y, "px (", round(label_pct * 100), "% from top)\n")
  cat("  Left:", crop_x, "px\n")
  cat("  Width:", crop_width, "px\n")
  cat("  Height:", crop_height, "px\n")
  
  # Show cropped result
  geometry <- sprintf("%dx%d+%d+%d", crop_width, crop_height, crop_x, crop_y)
  img_cropped <- image_crop(img, geometry)
  
  # Create side-by-side comparison
  img_resized <- image_resize(img, "300x")
  img_cropped_resized <- image_resize(img_cropped, "300x")
  
  comparison <- image_append(c(img_resized, img_cropped_resized))
  
  cat("\nShowing: Original (left) vs Cropped (right)\n")
  print(comparison)
  
  return(invisible(img_cropped))
}

# Main execution
if (!interactive()) {
  # Run on all training images
  process_all_training_images(backup = TRUE, method = "smart")
}
