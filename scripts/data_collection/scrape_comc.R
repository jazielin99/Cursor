# Scrape PSA graded card images from COMC.com
library(httr)
library(magick)

# Configuration
base_url <- "https://www.comc.com/Cards/Baseball"
grades <- c("PSA+10", "PSA+9", "PSA+8", "PSA+7", "PSA+6", "PSA+5", "PSA+4", "PSA+3", "PSA+2", "PSA+1")
grade_folders <- c("PSA_10", "PSA_9", "PSA_8", "PSA_7", "PSA_6", "PSA_5", "PSA_4", "PSA_3", "PSA_2", "PSA_1")
images_per_grade <- 30
output_dir <- "data/training"

# User agent to avoid blocking
user_agent <- "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Function to get image URLs from a search page
get_image_urls <- function(grade, page = 1) {
  url <- paste0(base_url, ",sp=", grade, ",pg=", page)
  cat("Fetching:", url, "\n")
  
  response <- tryCatch({
    GET(url, 
        user_agent(user_agent),
        timeout(30))
  }, error = function(e) {
    cat("Error:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(response) || status_code(response) != 200) {
    cat("Failed to fetch page\n")
    return(character(0))
  }
  
  content <- content(response, "text", encoding = "UTF-8")
  
  # Extract image URLs
  pattern <- 'https://img\\.comc\\.com/i/[^"]+\\.(jpg|png|jpeg)'
  matches <- regmatches(content, gregexpr(pattern, content, perl = TRUE))[[1]]
  
  # Get unique URLs
  unique(matches)
}

# Function to download an image
download_image <- function(url, save_path) {
  tryCatch({
    response <- GET(url, 
                    user_agent(user_agent),
                    timeout(30))
    
    if (status_code(response) == 200) {
      writeBin(content(response, "raw"), save_path)
      
      # Verify it's a valid image
      img <- image_read(save_path)
      info <- image_info(img)
      
      if (info$width > 50 && info$height > 50) {
        return(TRUE)
      } else {
        file.remove(save_path)
        return(FALSE)
      }
    }
    return(FALSE)
  }, error = function(e) {
    if (file.exists(save_path)) file.remove(save_path)
    return(FALSE)
  })
}

# Main scraping function
scrape_grade <- function(grade, folder, n_images) {
  cat("\n========================================\n")
  cat("Scraping", grade, "images\n")
  cat("========================================\n")
  
  # Create output folder
  grade_dir <- file.path(output_dir, folder)
  if (!dir.exists(grade_dir)) {
    dir.create(grade_dir, recursive = TRUE)
  }
  
  # Count existing images
  existing <- length(list.files(grade_dir, pattern = "\\.(jpg|png|jpeg)$", ignore.case = TRUE))
  cat("Existing images:", existing, "\n")
  
  if (existing >= n_images) {
    cat("Already have enough images, skipping\n")
    return(existing)
  }
  
  n_needed <- n_images - existing
  cat("Need to download:", n_needed, "more images\n")
  
  downloaded <- 0
  page <- 1
  max_pages <- 10
  
  while (downloaded < n_needed && page <= max_pages) {
    # Get image URLs from current page
    urls <- get_image_urls(grade, page)
    
    if (length(urls) == 0) {
      cat("No more images found\n")
      break
    }
    
    cat("Found", length(urls), "images on page", page, "\n")
    
    # Download images
    for (url in urls) {
      if (downloaded >= n_needed) break
      
      # Generate filename
      img_num <- existing + downloaded + 1
      ext <- tools::file_ext(url)
      if (ext == "") ext <- "jpg"
      filename <- sprintf("comc_%03d.%s", img_num, ext)
      save_path <- file.path(grade_dir, filename)
      
      # Skip if already exists
      if (file.exists(save_path)) {
        next
      }
      
      # Download
      cat("  Downloading", basename(url), "...")
      if (download_image(url, save_path)) {
        downloaded <- downloaded + 1
        cat(" OK (", downloaded, "/", n_needed, ")\n", sep = "")
      } else {
        cat(" FAILED\n")
      }
      
      # Rate limiting - be nice to the server
      Sys.sleep(0.5)
    }
    
    page <- page + 1
    Sys.sleep(1)  # Pause between pages
  }
  
  final_count <- length(list.files(grade_dir, pattern = "\\.(jpg|png|jpeg)$", ignore.case = TRUE))
  cat("Total images for", grade, ":", final_count, "\n")
  
  return(final_count)
}

# Main execution
cat("========================================\n")
cat("COMC PSA Card Image Scraper\n")
cat("========================================\n")
cat("Target images per grade:", images_per_grade, "\n")
cat("Output directory:", output_dir, "\n\n")

results <- data.frame(
  grade = character(),
  images = integer(),
  stringsAsFactors = FALSE
)

for (i in seq_along(grades)) {
  count <- scrape_grade(grades[i], grade_folders[i], images_per_grade)
  results <- rbind(results, data.frame(grade = grade_folders[i], images = count))
}

cat("\n========================================\n")
cat("SCRAPING COMPLETE\n")
cat("========================================\n")
print(results)
cat("\nTotal images:", sum(results$images), "\n")
