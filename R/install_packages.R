# Installs required packages for scraping + training.
# Run: Rscript R/install_packages.R

# Use a writable library (repo environments often don't allow /usr/local).
lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(lib)) {
  lib <- file.path(getwd(), ".Rlib")
}
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

repos <- c(CRAN = "https://cloud.r-project.org")

pkgs <- c(
  # data / utils
  "fs", "readr", "dplyr", "tibble", "purrr", "stringr",
  # scraping / downloading
  "rvest", "httr2", "robotstxt", "urltools", "progress",
  # ML
  "torch", "torchvision", "luz"
)

installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

if (length(to_install) > 0) {
  install.packages(to_install, repos = repos, lib = lib)
}

# torch needs libtorch; install it if missing
if (requireNamespace("torch", quietly = TRUE)) {
  # Safe to call multiple times
  torch::install_torch()
}
