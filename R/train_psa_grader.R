# Train a PSA grade image classifier using transfer learning (ResNet18).
#
# Expected dataset layout (image-folder format):
#   data/dataset/train/<CLASS_NAME>/*.jpg
#   data/dataset/val/<CLASS_NAME>/*.jpg
#
# Where <CLASS_NAME> are folder names like:
#   PSA_10, PSA_9, ..., PSA_1_5, PSA_1, PSA_N1, ... PSA_N9
#
# Quick start:
#   Rscript R/install_packages.R
#   Rscript -e "source('R/dataset_utils.R'); create_dataset_skeleton()"
#   # put labeled images into data/dataset/images/<CLASS_NAME>/
#   Rscript -e "source('R/dataset_utils.R'); split_dataset_images()"
#   Rscript R/train_psa_grader.R --epochs=10

source("R/dataset_utils.R")
use_local_lib()

suppressPackageStartupMessages({
  library(torch)
  library(torchvision)
  library(luz)
  library(fs)
  library(readr)
})

# --- simple arg parser: --key=value
args <- commandArgs(trailingOnly = TRUE)
parse_kv <- function(x) {
  x <- gsub("^--", "", x)
  parts <- strsplit(x, "=", fixed = TRUE)[[1]]
  if (length(parts) == 1) return(list(parts[[1]] = TRUE))
  setNames(list(parts[[2]]), parts[[1]])
}
kv <- list()
for (a in args) kv <- c(kv, parse_kv(a))

get_arg <- function(name, default = NULL) {
  if (!is.null(kv[[name]])) return(kv[[name]])
  default
}

train_dir <- get_arg("train_dir", "data/dataset/train")
val_dir <- get_arg("val_dir", "data/dataset/val")
images_dir <- get_arg("images_dir", "data/dataset/images")
out_dir <- get_arg("out_dir", "models")

epochs <- as.integer(get_arg("epochs", 10))
batch_size <- as.integer(get_arg("batch_size", 16))
seed <- as.integer(get_arg("seed", 1))

set.seed(seed)
torch_manual_seed(seed)

if (!fs::dir_exists(train_dir) || !fs::dir_exists(val_dir)) {
  if (fs::dir_exists(images_dir)) {
    message("train/val folders not found; splitting from images folder...")
    split_dataset_images(in_dir = images_dir, out_train = train_dir, out_val = val_dir, val_frac = 0.2, seed = seed)
  } else {
    stop("No dataset found. Create folders under data/dataset/images/<CLASS_NAME>/ or provide --train_dir/--val_dir.")
  }
}

compose <- function(...) {
  fns <- list(...)
  function(x) {
    for (fn in fns) x <- fn(x)
    x
  }
}

# ImageNet normalization for pretrained ResNet.
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)

train_tf <- compose(
  transform_random_resized_crop(size = c(224, 224)),
  transform_random_horizontal_flip(p = 0.5),
  transform_color_jitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.02),
  transform_to_tensor(),
  transform_normalize(mean = mean, std = std)
)

val_tf <- compose(
  transform_resize(size = 256),
  transform_center_crop(size = c(224, 224)),
  transform_to_tensor(),
  transform_normalize(mean = mean, std = std)
)

train_ds <- image_folder_dataset(root = train_dir, loader = magick_loader, transform = train_tf)
val_ds <- image_folder_dataset(root = val_dir, loader = magick_loader, transform = val_tf)

classes <- NULL
if (!is.null(train_ds$classes)) {
  classes <- train_ds$classes
} else if (!is.null(train_ds$class_to_idx)) {
  classes <- names(train_ds$class_to_idx)
} else {
  stop("Could not infer class names from dataset.")
}

num_classes <- length(classes)
message("Classes: ", num_classes)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

psa_model <- luz_module(
  initialize = function(num_classes) {
    self$backbone <- model_resnet18(pretrained = TRUE)

    # Replace final classification head.
    in_features <- self$backbone$fc$in_features
    self$backbone$fc <- nn_linear(in_features, num_classes)
  },
  forward = function(x) {
    self$backbone(x)
  }
)

model <- psa_model(num_classes = num_classes)

fitted <- model |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  ) |>
  fit(
    train_dl,
    epochs = epochs,
    valid_data = val_dl,
    verbose = TRUE
  )

fs::dir_create(out_dir, recurse = TRUE)
model_path <- fs::path(out_dir, "psa_grader.luz")
classes_path <- fs::path(out_dir, "classes.csv")

luz_save(fitted, model_path)
readr::write_csv(data.frame(class = classes, idx = seq_along(classes) - 1L), classes_path)

message("Saved model: ", model_path)
message("Saved classes: ", classes_path)
