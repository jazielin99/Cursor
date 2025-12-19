# Utilities for dataset creation / splitting.

use_local_lib <- function(project_root = getwd()) {
  lib <- Sys.getenv("R_LIBS_USER")
  if (!nzchar(lib)) lib <- file.path(project_root, ".Rlib")
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
  invisible(lib)
}

create_dataset_skeleton <- function(
  out_dir = "data/dataset/images",
  include_half_points = TRUE,
  include_no_grade = TRUE
) {
  use_local_lib()
  suppressPackageStartupMessages({
    library(fs)
    library(readr)
    library(dplyr)
    library(purrr)
  })

  source("R/psa_definitions.R")
  defs <- psa_grade_definitions(include_half_points = include_half_points)

  classes <- defs$classes |>
    dplyr::select(class, grade, label, title)

  if (isTRUE(include_no_grade)) {
    ng <- defs$no_grade |>
      dplyr::transmute(class = class, grade = NA_real_, label = paste0("NoGrade ", code), title = label)
    classes <- dplyr::bind_rows(classes, ng)
  }

  fs::dir_create(out_dir, recurse = TRUE)
  purrr::walk(classes$class, ~fs::dir_create(fs::path(out_dir, .x)))

  fs::dir_create("data/dataset", recurse = TRUE)
  readr::write_csv(classes, "data/dataset/classes.csv")

  invisible(classes)
}

split_dataset_images <- function(
  in_dir = "data/dataset/images",
  out_train = "data/dataset/train",
  out_val = "data/dataset/val",
  val_frac = 0.2,
  seed = 1
) {
  use_local_lib()
  suppressPackageStartupMessages({
    library(fs)
    library(purrr)
    library(dplyr)
    library(stringr)
  })

  set.seed(seed)

  classes <- fs::dir_ls(in_dir, type = "directory") |> basename()

  fs::dir_create(out_train, recurse = TRUE)
  fs::dir_create(out_val, recurse = TRUE)

  purrr::walk(classes, function(cls) {
    cls_in <- fs::path(in_dir, cls)
    files <- fs::dir_ls(cls_in, recurse = FALSE, type = "file")
    if (length(files) == 0) {
      fs::dir_create(fs::path(out_train, cls))
      fs::dir_create(fs::path(out_val, cls))
      return(invisible(NULL))
    }

    n_val <- max(1, floor(length(files) * val_frac))
    val_files <- sample(files, size = n_val)
    train_files <- setdiff(files, val_files)

    fs::dir_create(fs::path(out_train, cls), recurse = TRUE)
    fs::dir_create(fs::path(out_val, cls), recurse = TRUE)

    purrr::walk(train_files, ~fs::file_copy(.x, fs::path(out_train, cls, basename(.x)), overwrite = TRUE))
    purrr::walk(val_files, ~fs::file_copy(.x, fs::path(out_val, cls, basename(.x)), overwrite = TRUE))
  })

  invisible(list(train_dir = out_train, val_dir = out_val))
}
