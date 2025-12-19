# Optional: download images referenced by PSA grading standards page.
# Run: Rscript R/scrape_psa_images.R
#
# IMPORTANT:
# - This script checks robots.txt first and only proceeds if allowed.
# - You are responsible for complying with PSA Terms of Use.

source("R/dataset_utils.R")
use_local_lib()

suppressPackageStartupMessages({
  library(rvest)
  library(httr2)
  library(robotstxt)
  library(urltools)
  library(fs)
  library(stringr)
  library(purrr)
  library(digest)
  library(xml2)
})

base_url <- "https://www.psacard.com"
page_url <- "https://www.psacard.com/gradingstandards"
out_dir <- "data/psa_site_images"

dir_create(out_dir, recurse = TRUE)

allowed <- tryCatch(
  {
    robotstxt::paths_allowed(paths = "/gradingstandards", domain = base_url)
  },
  error = function(e) NA
)

if (isFALSE(allowed)) {
  stop("robots.txt disallows scraping /gradingstandards; aborting.")
}

message("Fetching page HTML...")
ua <- "psa-grading-standards-research-bot/1.0 (respectful use)"

page_resp <- tryCatch(
  {
    httr2::request(page_url) |>
      httr2::req_user_agent(ua) |>
      httr2::req_headers(
        `accept` = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        `accept-language` = "en-US,en;q=0.9"
      ) |>
      httr2::req_timeout(30) |>
      httr2::req_perform()
  },
  error = function(e) NULL
)

if (is.null(page_resp)) {
  stop("Failed to fetch the page. PSA may be blocking automated requests from this environment.")
}

if (httr2::resp_status(page_resp) == 403) {
  stop(
    "PSA returned HTTP 403 (likely a bot-protection challenge). ",
    "In that case, fully automated scraping from this environment won't work. ",
    "Workaround: download images manually in a browser and place them into data/dataset/images/<CLASS_NAME>/."
  )
}

if (httr2::resp_status(page_resp) >= 400) {
  stop("HTTP error fetching page: ", httr2::resp_status(page_resp))
}

html <- xml2::read_html(httr2::resp_body_string(page_resp))

# Collect all <img> sources.
img_srcs <- html |>
  html_elements("img") |>
  html_attr("src") |>
  unique()

img_srcs <- img_srcs[!is.na(img_srcs) & nzchar(img_srcs)]

# Resolve relative URLs.
resolve_url <- function(src) {
  if (stringr::str_starts(src, "//")) return(paste0("https:", src))
  if (stringr::str_starts(src, "http://") || stringr::str_starts(src, "https://")) return(src)
  paste0(base_url, src)
}

urls <- purrr::map_chr(img_srcs, resolve_url) |> unique()

# Keep only image-like paths.
urls <- urls[stringr::str_detect(urls, "\\.(png|jpe?g|webp|gif)(\\?|$)")]

if (length(urls) == 0) {
  message("No image URLs found on the page.")
  quit(save = "no", status = 0)
}

# Download with a small delay.

safe_filename <- function(u) {
  path <- urltools::url_parse(u)$path
  base <- basename(path)
  base <- stringr::str_replace_all(base, "[^A-Za-z0-9._-]", "_")
  if (!nzchar(base)) base <- paste0("img_", digest::digest(u), ".bin")
  base
}

message("Found ", length(urls), " image URLs. Downloading...")

purrr::iwalk(urls, function(u, i) {
  dest <- fs::path(out_dir, sprintf("%04d_%s", i, safe_filename(u)))
  if (fs::file_exists(dest)) return(invisible(NULL))

  req <- httr2::request(u) |>
    httr2::req_user_agent(ua) |>
    httr2::req_timeout(30)

  resp <- tryCatch(httr2::req_perform(req), error = function(e) NULL)
  if (is.null(resp) || httr2::resp_status(resp) >= 400) {
    message("Skip (HTTP error): ", u)
    return(invisible(NULL))
  }

  bin <- httr2::resp_body_raw(resp)
  writeBin(bin, dest)

  Sys.sleep(0.5)
})

message("Done. Images saved under: ", out_dir)
