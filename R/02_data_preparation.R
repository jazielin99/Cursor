# ==============================================================================
# PSA Card Grading Model - Data Preparation
# ==============================================================================
# Functions for loading, preprocessing, and augmenting card images

# Source configuration
source(file.path(getwd(), "R", "config.R"))

# Required packages
suppressPackageStartupMessages({
  library(magick)
  library(jpeg)
  library(png)
  library(tidyverse)
})

# ------------------------------------------------------------------------------
# Image Loading Functions
# ------------------------------------------------------------------------------

#' Load a single image and preprocess it
#' @param image_path Path to the image file
#' @param target_size Vector of (width, height) for resizing
#' @param normalize Whether to normalize pixel values to [0,1]
#' @return Preprocessed image array
load_and_preprocess_image <- function(image_path, 
                                       target_size = c(224, 224),
                                       normalize = TRUE) {
  tryCatch({
    # Read image using magick
    img <- image_read(image_path)
    
    # Resize to target size
    img <- image_resize(img, paste0(target_size[1], "x", target_size[2], "!"))
    
    # Convert to array - image_data returns (channels x width x height)
    img_data <- image_data(img, channels = "rgb")
    
    # Convert to integer then numeric for proper values
    img_array <- as.integer(img_data)
    
    # Reshape from (channels x width x height) to (height x width x channels)
    # img_data is 3 x width x height, we need height x width x 3
    img_array <- aperm(array(img_array, dim = dim(img_data)), c(3, 2, 1))
    
    # Normalize to [0, 1] if requested
    if (normalize) {
      img_array <- img_array / 255
    }
    
    return(img_array)
    
  }, error = function(e) {
    warning(paste("Error loading image:", image_path, "-", e$message))
    return(NULL)
  })
}

#' Load all images from a directory
#' @param dir_path Path to directory containing images
#' @param target_size Vector of (width, height) for resizing
#' @param extensions Vector of valid image extensions
#' @return List with images array and file names
load_images_from_directory <- function(dir_path,
                                        target_size = c(224, 224),
                                        extensions = c("jpg", "jpeg", "png", "gif", "bmp")) {
  
  if (!dir.exists(dir_path)) {
    stop(paste("Directory does not exist:", dir_path))
  }
  
  # Get all image files
  pattern <- paste0("\\.(", paste(extensions, collapse = "|"), ")$")
  files <- list.files(dir_path, pattern = pattern, 
                      full.names = TRUE, ignore.case = TRUE)
  
  if (length(files) == 0) {
    warning(paste("No images found in:", dir_path))
    return(list(images = NULL, files = character(0)))
  }
  
  cat("Loading", length(files), "images from", dir_path, "\n")
  
  # Load all images
  images <- list()
  valid_files <- character(0)
  
  pb <- txtProgressBar(min = 0, max = length(files), style = 3)
  
  for (i in seq_along(files)) {
    img <- load_and_preprocess_image(files[i], target_size)
    if (!is.null(img)) {
      images[[length(images) + 1]] <- img
      valid_files <- c(valid_files, basename(files[i]))
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  if (length(images) == 0) {
    return(list(images = NULL, files = character(0)))
  }
  
  # Stack images into a 4D array (n, height, width, channels)
  images_array <- array(0, dim = c(length(images), target_size[2], target_size[1], 3))
  for (i in seq_along(images)) {
    images_array[i, , , ] <- images[[i]]
  }
  
  return(list(images = images_array, files = valid_files))
}

# ------------------------------------------------------------------------------
# Dataset Loading Functions
# ------------------------------------------------------------------------------

#' Load training dataset with class labels
#' @param training_dir Path to training directory with subdirectories per class
#' @param classes Vector of class names (subdirectory names)
#' @param target_size Vector of (width, height) for resizing
#' @return List with images, labels, class_names, and file_info
load_training_dataset <- function(training_dir = NULL,
                                   classes = NULL,
                                   target_size = c(224, 224)) {
  
  if (is.null(training_dir)) {
    training_dir <- get_config("training_dir")
  }
  
  if (is.null(classes)) {
    classes <- get_config("grade_classes")
  }
  
  if (!dir.exists(training_dir)) {
    stop(paste("Training directory does not exist:", training_dir))
  }
  
  all_images <- list()
  all_labels <- numeric(0)
  all_file_info <- data.frame(
    file = character(0),
    class = character(0),
    class_idx = integer(0),
    stringsAsFactors = FALSE
  )
  
  cat("Loading training dataset...\n")
  
  for (i in seq_along(classes)) {
    class_name <- classes[i]
    class_dir <- file.path(training_dir, class_name)
    
    if (dir.exists(class_dir)) {
      result <- load_images_from_directory(class_dir, target_size)
      
      if (!is.null(result$images)) {
        n_images <- dim(result$images)[1]
        all_images[[length(all_images) + 1]] <- result$images
        all_labels <- c(all_labels, rep(i - 1, n_images))  # 0-indexed labels
        
        all_file_info <- rbind(all_file_info, data.frame(
          file = result$files,
          class = class_name,
          class_idx = i - 1,
          stringsAsFactors = FALSE
        ))
        
        cat(sprintf("  %s: %d images\n", class_name, n_images))
      }
    } else {
      cat(sprintf("  %s: directory not found\n", class_name))
    }
  }
  
  if (length(all_images) == 0) {
    stop("No images found in training directory!")
  }
  
  # Combine all images into single array
  total_images <- sum(sapply(all_images, function(x) dim(x)[1]))
  combined_images <- array(0, dim = c(total_images, target_size[2], target_size[1], 3))
  
  idx <- 1
  for (img_array in all_images) {
    n <- dim(img_array)[1]
    combined_images[idx:(idx + n - 1), , , ] <- img_array
    idx <- idx + n
  }
  
  cat(sprintf("\nTotal: %d images across %d classes\n", total_images, length(unique(all_labels))))
  
  return(list(
    images = combined_images,
    labels = all_labels,
    class_names = classes,
    file_info = all_file_info
  ))
}

# ------------------------------------------------------------------------------
# Data Augmentation Functions
# ------------------------------------------------------------------------------

#' Augment a single image
#' @param img Image array (height x width x channels)
#' @param flip_horizontal Whether to randomly flip horizontally
#' @param rotation_range Maximum rotation angle in degrees
#' @param brightness_range Range for brightness adjustment (min, max)
#' @param zoom_range Range for zoom (min, max)
#' @return Augmented image array
augment_image <- function(img,
                          flip_horizontal = TRUE,
                          rotation_range = 15,
                          brightness_range = c(0.8, 1.2),
                          zoom_range = c(0.9, 1.1)) {
  
  # Convert to magick image
  # Scale to 0-255 if normalized
  if (max(img) <= 1) {
    img_scaled <- img * 255
  } else {
    img_scaled <- img
  }
  
  # Create magick image
  img_magick <- image_read(as.raw(as.integer(img_scaled)))
  
  # Random horizontal flip
  if (flip_horizontal && runif(1) > 0.5) {
    img_magick <- image_flop(img_magick)
  }
  
  # Random rotation
  if (rotation_range > 0) {
    angle <- runif(1, -rotation_range, rotation_range)
    img_magick <- image_rotate(img_magick, angle)
  }
  
  # Random brightness adjustment
  if (!is.null(brightness_range)) {
    brightness <- runif(1, brightness_range[1], brightness_range[2])
    img_magick <- image_modulate(img_magick, brightness = brightness * 100)
  }
  
  # Convert back to array
  dims <- dim(img)
  img_magick <- image_resize(img_magick, paste0(dims[2], "x", dims[1], "!"))
  img_array <- as.numeric(image_data(img_magick, channels = "rgb"))
  img_array <- array(img_array, dim = dims)
  
  # Normalize back to [0, 1]
  img_array <- img_array / 255
  
  return(img_array)
}

#' Create augmented dataset
#' @param images Original images array (n x height x width x channels)
#' @param labels Original labels vector
#' @param augmentation_factor How many augmented copies per original
#' @return List with augmented images and labels
create_augmented_dataset <- function(images, labels, augmentation_factor = 2) {
  
  n_original <- dim(images)[1]
  n_augmented <- n_original * augmentation_factor
  
  dims <- dim(images)
  augmented_images <- array(0, dim = c(n_augmented, dims[2], dims[3], dims[4]))
  augmented_labels <- numeric(n_augmented)
  
  cat("Creating augmented dataset...\n")
  pb <- txtProgressBar(min = 0, max = n_augmented, style = 3)
  
  idx <- 1
  for (i in seq_len(n_original)) {
    for (j in seq_len(augmentation_factor)) {
      augmented_images[idx, , , ] <- augment_image(images[i, , , ])
      augmented_labels[idx] <- labels[i]
      idx <- idx + 1
      setTxtProgressBar(pb, idx)
    }
  }
  close(pb)
  
  cat(sprintf("\nCreated %d augmented images\n", n_augmented))
  
  return(list(
    images = augmented_images,
    labels = augmented_labels
  ))
}

# ------------------------------------------------------------------------------
# Train/Validation Split Functions
# ------------------------------------------------------------------------------

#' Split dataset into training and validation sets
#' @param images Images array
#' @param labels Labels vector
#' @param validation_split Proportion for validation (0-1)
#' @param stratify Whether to stratify by class
#' @param seed Random seed for reproducibility
#' @return List with train and validation data
split_dataset <- function(images, labels, 
                          validation_split = 0.2,
                          stratify = TRUE,
                          seed = 42) {
  
  set.seed(seed)
  
  n <- dim(images)[1]
  
  if (stratify) {
    # Stratified split
    train_idx <- c()
    val_idx <- c()
    
    for (class_label in unique(labels)) {
      class_idx <- which(labels == class_label)
      n_class <- length(class_idx)
      n_val <- max(1, floor(n_class * validation_split))
      
      shuffled <- sample(class_idx)
      val_idx <- c(val_idx, shuffled[1:n_val])
      train_idx <- c(train_idx, shuffled[(n_val + 1):n_class])
    }
  } else {
    # Random split
    shuffled <- sample(n)
    n_val <- floor(n * validation_split)
    val_idx <- shuffled[1:n_val]
    train_idx <- shuffled[(n_val + 1):n]
  }
  
  return(list(
    train = list(
      images = images[train_idx, , , , drop = FALSE],
      labels = labels[train_idx]
    ),
    validation = list(
      images = images[val_idx, , , , drop = FALSE],
      labels = labels[val_idx]
    )
  ))
}

# ------------------------------------------------------------------------------
# Data Generators (for Keras)
# ------------------------------------------------------------------------------

#' Create a data generator function for batch processing
#' @param images Images array
#' @param labels Labels vector (or one-hot encoded matrix)
#' @param batch_size Batch size
#' @param shuffle Whether to shuffle data
#' @param augment Whether to augment data
#' @return Generator function
create_data_generator <- function(images, labels, 
                                   batch_size = 32,
                                   shuffle = TRUE,
                                   augment = FALSE) {
  
  n <- dim(images)[1]
  n_batches <- ceiling(n / batch_size)
  current_batch <- 0
  indices <- seq_len(n)
  
  if (shuffle) {
    indices <- sample(indices)
  }
  
  # Return a function that generates batches
  function() {
    current_batch <<- current_batch + 1
    
    # Reset and reshuffle at epoch end
    if (current_batch > n_batches) {
      current_batch <<- 1
      if (shuffle) {
        indices <<- sample(indices)
      }
    }
    
    # Get batch indices
    start_idx <- (current_batch - 1) * batch_size + 1
    end_idx <- min(current_batch * batch_size, n)
    batch_indices <- indices[start_idx:end_idx]
    
    # Get batch data
    batch_images <- images[batch_indices, , , , drop = FALSE]
    batch_labels <- labels[batch_indices]
    
    # Augment if requested
    if (augment) {
      for (i in seq_len(dim(batch_images)[1])) {
        batch_images[i, , , ] <- augment_image(batch_images[i, , , ])
      }
    }
    
    # Convert labels to one-hot if needed
    if (is.vector(batch_labels)) {
      n_classes <- length(unique(labels))
      batch_labels_onehot <- matrix(0, nrow = length(batch_labels), ncol = n_classes)
      for (i in seq_along(batch_labels)) {
        batch_labels_onehot[i, batch_labels[i] + 1] <- 1  # +1 because labels are 0-indexed
      }
      batch_labels <- batch_labels_onehot
    }
    
    list(batch_images, batch_labels)
  }
}

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

#' Get class weights for imbalanced datasets
#' @param labels Labels vector
#' @return Named vector of class weights
compute_class_weights <- function(labels) {
  class_counts <- table(labels)
  total <- sum(class_counts)
  n_classes <- length(class_counts)
  
  # Compute balanced weights
  weights <- total / (n_classes * class_counts)
  
  # Normalize so minimum weight is 1
  weights <- weights / min(weights)
  
  names(weights) <- names(class_counts)
  
  return(weights)
}

#' Summarize dataset
#' @param dataset Dataset list with images and labels
#' @param class_names Vector of class names
print_dataset_summary <- function(dataset, class_names = NULL) {
  n <- dim(dataset$images)[1]
  dims <- dim(dataset$images)
  
  cat("=== Dataset Summary ===\n")
  cat(sprintf("Total images: %d\n", n))
  cat(sprintf("Image dimensions: %d x %d x %d\n", dims[2], dims[3], dims[4]))
  cat(sprintf("Data type: %s\n", typeof(dataset$images)))
  cat(sprintf("Value range: [%.3f, %.3f]\n", min(dataset$images), max(dataset$images)))
  
  cat("\nClass distribution:\n")
  class_counts <- table(dataset$labels)
  
  for (i in seq_along(class_counts)) {
    label <- names(class_counts)[i]
    count <- class_counts[i]
    pct <- round(count / n * 100, 1)
    
    if (!is.null(class_names)) {
      class_name <- class_names[as.numeric(label) + 1]
      cat(sprintf("  %s (%s): %d (%.1f%%)\n", label, class_name, count, pct))
    } else {
      cat(sprintf("  Class %s: %d (%.1f%%)\n", label, count, pct))
    }
  }
}

#' Save preprocessed dataset to file
#' @param dataset Dataset list
#' @param file_path Path to save file (.rds)
save_dataset <- function(dataset, file_path) {
  cat("Saving dataset to:", file_path, "\n")
  saveRDS(dataset, file_path)
  cat("Dataset saved successfully.\n")
}

#' Load preprocessed dataset from file
#' @param file_path Path to .rds file
#' @return Dataset list
load_dataset <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(paste("Dataset file not found:", file_path))
  }
  cat("Loading dataset from:", file_path, "\n")
  dataset <- readRDS(file_path)
  cat("Dataset loaded successfully.\n")
  return(dataset)
}
