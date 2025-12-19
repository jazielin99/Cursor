# ==============================================================================
# PSA Card Grading Model - Image Collection Helper
# ==============================================================================
# Helper functions for collecting training images from various sources
#
# NOTE: The PSA website (psacard.com) is protected by Cloudflare and cannot
# be directly scraped. This script provides alternative methods for collecting
# training images.

# Set working directory
setwd("/workspace")

# Load required packages
if (!require("httr")) install.packages("httr")
if (!require("rvest")) install.packages("rvest")
if (!require("magick")) install.packages("magick")
if (!require("jsonlite")) install.packages("jsonlite")

library(httr)
library(rvest)
library(magick)
library(jsonlite)

# Source config
source("R/config.R")

# ------------------------------------------------------------------------------
# Image Download Utilities
# ------------------------------------------------------------------------------

#' Download an image from a URL
#' @param url Image URL
#' @param save_path Path to save the image
#' @param timeout Request timeout in seconds
#' @return TRUE if successful, FALSE otherwise
download_image <- function(url, save_path, timeout = 30) {
  tryCatch({
    # Set user agent to avoid blocks
    response <- GET(
      url,
      timeout(timeout),
      user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    )
    
    if (status_code(response) == 200) {
      # Write content to file
      writeBin(content(response, "raw"), save_path)
      
      # Verify it's a valid image
      img <- tryCatch({
        image_read(save_path)
      }, error = function(e) {
        file.remove(save_path)
        return(NULL)
      })
      
      if (!is.null(img)) {
        return(TRUE)
      }
    }
    
    return(FALSE)
  }, error = function(e) {
    return(FALSE)
  })
}

#' Download multiple images with progress
#' @param urls Vector of image URLs
#' @param save_dir Directory to save images
#' @param prefix Filename prefix
#' @return Number of successfully downloaded images
download_images <- function(urls, save_dir, prefix = "card") {
  
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  success_count <- 0
  
  cat("Downloading", length(urls), "images...\n")
  pb <- txtProgressBar(min = 0, max = length(urls), style = 3)
  
  for (i in seq_along(urls)) {
    # Determine file extension
    ext <- tools::file_ext(urls[i])
    if (ext == "" || !ext %in% c("jpg", "jpeg", "png", "gif")) {
      ext <- "jpg"
    }
    
    save_path <- file.path(save_dir, sprintf("%s_%04d.%s", prefix, i, ext))
    
    if (download_image(urls[i], save_path)) {
      success_count <- success_count + 1
    }
    
    setTxtProgressBar(pb, i)
    
    # Small delay to be respectful
    Sys.sleep(0.5)
  }
  close(pb)
  
  cat(sprintf("\nDownloaded %d of %d images\n", success_count, length(urls)))
  
  return(success_count)
}

# ------------------------------------------------------------------------------
# Manual Image Organization
# ------------------------------------------------------------------------------

#' Create folder structure for organizing images
#' @param base_dir Base directory for training data
create_training_folders <- function(base_dir = NULL) {
  
  if (is.null(base_dir)) {
    base_dir <- get_config("training_dir")
  }
  
  grades <- get_config("grade_classes")
  
  cat("Creating folder structure in:", base_dir, "\n\n")
  
  for (grade in grades) {
    grade_dir <- file.path(base_dir, grade)
    if (!dir.exists(grade_dir)) {
      dir.create(grade_dir, recursive = TRUE)
      cat("Created:", grade_dir, "\n")
    } else {
      n_files <- length(list.files(grade_dir))
      cat("Exists:", grade_dir, "(", n_files, "files )\n")
    }
  }
  
  cat("\nFolder structure created!\n")
  cat("Add card images to each folder based on their PSA grade.\n")
}

#' Move images to appropriate grade folders interactively
#' @param source_dir Directory containing unsorted images
#' @param target_base Target base directory (default: training dir)
interactive_sort_images <- function(source_dir, target_base = NULL) {
  
  if (!dir.exists(source_dir)) {
    stop("Source directory not found:", source_dir)
  }
  
  if (is.null(target_base)) {
    target_base <- get_config("training_dir")
  }
  
  # Get all images
  images <- list.files(source_dir, pattern = "\\.(jpg|jpeg|png|gif|bmp)$",
                       full.names = TRUE, ignore.case = TRUE)
  
  if (length(images) == 0) {
    cat("No images found in:", source_dir, "\n")
    return()
  }
  
  grades <- get_config("grade_classes")
  
  cat("=== Interactive Image Sorting ===\n")
  cat("For each image, enter the PSA grade or:\n")
  cat("  'skip' - Skip this image\n")
  cat("  'quit' - Stop sorting\n")
  cat("\nValid grades:", paste(grades, collapse = ", "), "\n\n")
  
  for (img_path in images) {
    cat("\nImage:", basename(img_path), "\n")
    cat("Enter PSA grade (or 'skip'/'quit'): ")
    
    input <- readline()
    
    if (tolower(input) == "quit") {
      cat("Sorting stopped.\n")
      break
    }
    
    if (tolower(input) == "skip") {
      cat("Skipped.\n")
      next
    }
    
    # Normalize input
    grade <- toupper(gsub("[^0-9.]", "", input))
    grade_dir <- paste0("PSA_", grade)
    
    if (tolower(input) %in% c("no", "nograde", "no_grade", "n")) {
      grade_dir <- "NO_GRADE"
    }
    
    if (!grade_dir %in% grades) {
      cat("Invalid grade. Skipping.\n")
      next
    }
    
    # Move file
    target_dir <- file.path(target_base, grade_dir)
    if (!dir.exists(target_dir)) {
      dir.create(target_dir, recursive = TRUE)
    }
    
    target_path <- file.path(target_dir, basename(img_path))
    file.copy(img_path, target_path)
    cat("Moved to:", grade_dir, "\n")
  }
  
  cat("\nSorting complete!\n")
}

# ------------------------------------------------------------------------------
# Image Validation and Cleaning
# ------------------------------------------------------------------------------

#' Validate images in training directory
#' @param training_dir Path to training directory
#' @param remove_invalid Whether to remove invalid images
validate_training_images <- function(training_dir = NULL, remove_invalid = FALSE) {
  
  if (is.null(training_dir)) {
    training_dir <- get_config("training_dir")
  }
  
  grades <- get_config("grade_classes")
  
  cat("=== Validating Training Images ===\n\n")
  
  total_valid <- 0
  total_invalid <- 0
  
  for (grade in grades) {
    grade_dir <- file.path(training_dir, grade)
    
    if (!dir.exists(grade_dir)) {
      next
    }
    
    images <- list.files(grade_dir, full.names = TRUE)
    valid_count <- 0
    invalid_images <- c()
    
    for (img_path in images) {
      tryCatch({
        img <- image_read(img_path)
        info <- image_info(img)
        
        if (info$width > 0 && info$height > 0) {
          valid_count <- valid_count + 1
        } else {
          invalid_images <- c(invalid_images, img_path)
        }
      }, error = function(e) {
        invalid_images <<- c(invalid_images, img_path)
      })
    }
    
    invalid_count <- length(invalid_images)
    total_valid <- total_valid + valid_count
    total_invalid <- total_invalid + invalid_count
    
    status <- if (invalid_count > 0) sprintf("(%d invalid)", invalid_count) else ""
    cat(sprintf("%s: %d valid %s\n", grade, valid_count, status))
    
    # Remove invalid images if requested
    if (remove_invalid && length(invalid_images) > 0) {
      for (inv_img in invalid_images) {
        file.remove(inv_img)
        cat("  Removed:", basename(inv_img), "\n")
      }
    }
  }
  
  cat(sprintf("\nTotal: %d valid, %d invalid\n", total_valid, total_invalid))
}

#' Resize all images to target size
#' @param training_dir Path to training directory
#' @param target_size Target size (width, height)
#' @param quality JPEG quality (1-100)
resize_training_images <- function(training_dir = NULL, 
                                    target_size = c(224, 224),
                                    quality = 90) {
  
  if (is.null(training_dir)) {
    training_dir <- get_config("training_dir")
  }
  
  grades <- get_config("grade_classes")
  
  cat("=== Resizing Training Images ===\n")
  cat("Target size:", target_size[1], "x", target_size[2], "\n\n")
  
  for (grade in grades) {
    grade_dir <- file.path(training_dir, grade)
    
    if (!dir.exists(grade_dir)) {
      next
    }
    
    images <- list.files(grade_dir, pattern = "\\.(jpg|jpeg|png|gif|bmp)$",
                         full.names = TRUE, ignore.case = TRUE)
    
    if (length(images) == 0) {
      next
    }
    
    cat(grade, ": resizing", length(images), "images...")
    
    for (img_path in images) {
      tryCatch({
        img <- image_read(img_path)
        img <- image_resize(img, paste0(target_size[1], "x", target_size[2], "!"))
        image_write(img, img_path, quality = quality)
      }, error = function(e) {
        # Skip failed images
      })
    }
    
    cat(" done\n")
  }
  
  cat("\nResizing complete!\n")
}

# ------------------------------------------------------------------------------
# Sample Data Generator (for testing)
# ------------------------------------------------------------------------------

#' Generate synthetic sample images for testing the pipeline
#' @param output_dir Output directory
#' @param n_per_class Number of images per class
#' @param image_size Image size (width, height)
generate_sample_data <- function(output_dir = NULL,
                                  n_per_class = 10,
                                  image_size = c(224, 224)) {
  
  if (is.null(output_dir)) {
    output_dir <- get_config("training_dir")
  }
  
  grades <- get_config("grade_classes")
  
  cat("=== Generating Sample Data ===\n")
  cat("This creates synthetic images for testing the pipeline.\n")
  cat("Replace with real card images for actual training.\n\n")
  
  set.seed(42)
  
  for (grade in grades) {
    grade_dir <- file.path(output_dir, grade)
    
    if (!dir.exists(grade_dir)) {
      dir.create(grade_dir, recursive = TRUE)
    }
    
    # Generate synthetic images with different characteristics per grade
    # Higher grades get "cleaner" images
    grade_num <- as.numeric(gsub("PSA_|NO_GRADE", "", grade))
    if (is.na(grade_num)) grade_num <- 0
    
    for (i in seq_len(n_per_class)) {
      # Create base image using magick
      # Color based on grade (higher = more white/clean)
      base_color <- sprintf("rgb(%d,%d,%d)", 
                            150 + grade_num * 10, 
                            150 + grade_num * 10, 
                            150 + grade_num * 10)
      
      img <- image_blank(image_size[1], image_size[2], color = base_color)
      
      # Add grade label
      img <- image_annotate(img, paste("PSA", grade_num), 
                            size = 24, gravity = "center", 
                            color = "black")
      
      # Add some visual noise/variation
      # Lower grades get more distortion
      if (grade_num < 10) {
        blur_sigma <- (10 - grade_num) * 0.3
        img <- image_blur(img, sigma = blur_sigma)
      }
      
      if (grade_num < 7) {
        # Add noise for lower grades
        noise_img <- image_blank(image_size[1], image_size[2], color = "white")
        noise_img <- image_noise(noise_img, noisetype = "gaussian")
        img <- image_composite(img, noise_img, operator = "blend", 
                               compose_args = paste0((10-grade_num)*5, "x100"))
      }
      
      if (grade_num < 5) {
        # Add some yellowing for very low grades
        img <- image_modulate(img, brightness = 95, saturation = 90)
      }
      
      # Add border/frame
      img <- image_border(img, color = "gray50", geometry = "3x3")
      
      # Resize back to target size
      img <- image_resize(img, paste0(image_size[1], "x", image_size[2], "!"))
      
      # Save as JPEG
      save_path <- file.path(grade_dir, sprintf("sample_%03d.jpg", i))
      image_write(img, save_path, format = "jpeg", quality = 90)
    }
    
    cat(sprintf("%s: %d sample images created\n", grade, n_per_class))
  }
  
  cat("\nSample data generation complete!\n")
  cat("NOTE: These are synthetic images for testing only.\n")
  cat("Replace with real card images for actual model training.\n")
}

# ------------------------------------------------------------------------------
# Data Collection Instructions
# ------------------------------------------------------------------------------

print_collection_instructions <- function() {
  cat("
==============================================================================
                    HOW TO COLLECT TRAINING IMAGES
==============================================================================

Since PSA's website is protected by Cloudflare and cannot be scraped,
here are alternative methods to collect training images:

METHOD 1: eBay Listings
-----------------------
1. Go to eBay.com
2. Search for 'PSA 10 [card type]' (e.g., 'PSA 10 Pokemon')
3. Look for listings with clear photos of the graded card
4. Right-click and save images showing the PSA case with grade visible
5. Repeat for each grade (PSA 9, PSA 8, etc.)

METHOD 2: Collector Forums & Social Media
-----------------------------------------
1. Visit r/baseballcards, r/footballcards, r/pokemontcg on Reddit
2. Look for posts showing PSA graded cards
3. Request permission before using images
4. Many collectors share high-quality photos

METHOD 3: Your Own Collection
-----------------------------
1. Photograph your own PSA graded cards
2. Use consistent lighting (natural light works well)
3. Include both front and back if possible
4. Ensure the grade label is visible

METHOD 4: PSA Pop Report + Search
---------------------------------
1. Use PSA's Population Report to identify graded cards
2. Search for those specific cards on collector sites
3. Verify the grade matches the label in photos

TIPS FOR GOOD TRAINING DATA:
----------------------------
• Aim for at least 50-100 images per grade
• Include variety: different card types, years, sports
• Use consistent image quality
• Crop images to show mainly the card
• Include the PSA case/label when possible
• Balance your dataset across grades

ORGANIZING YOUR IMAGES:
-----------------------
After collecting images, run:

  create_training_folders()          # Create the folder structure
  interactive_sort_images('path/to/unsorted/')  # Sort images interactively
  validate_training_images()         # Check for corrupt images
  resize_training_images()           # Resize to 224x224

==============================================================================
")
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if (interactive()) {
  cat("\n=== Image Collection Helper ===\n")
  cat("\nAvailable functions:\n")
  cat("  print_collection_instructions()  - How to collect images\n")
  cat("  create_training_folders()        - Create folder structure\n")
  cat("  interactive_sort_images(dir)     - Sort images by grade\n")
  cat("  validate_training_images()       - Check for corrupt images\n")
  cat("  resize_training_images()         - Resize to 224x224\n")
  cat("  generate_sample_data()           - Create synthetic test data\n")
  cat("  download_images(urls, save_dir)  - Download from URLs\n")
  cat("\n")
}
