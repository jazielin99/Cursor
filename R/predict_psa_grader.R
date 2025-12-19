# Predict PSA grade for one image.
# Example:
#   Rscript R/predict_psa_grader.R --image=/path/to/card.jpg

source("R/dataset_utils.R")
use_local_lib()

suppressPackageStartupMessages({
  library(torch)
  library(torchvision)
  library(luz)
  library(fs)
  library(readr)
})

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

image_path <- get_arg("image", NULL)
if (is.null(image_path)) stop("Provide --image=/path/to/image.jpg")

model_path <- get_arg("model", "models/psa_grader.luz")
classes_path <- get_arg("classes", "models/classes.csv")
topk <- as.integer(get_arg("topk", 5))

if (!fs::file_exists(model_path)) stop("Model not found: ", model_path)
if (!fs::file_exists(classes_path)) stop("Classes file not found: ", classes_path)
if (!fs::file_exists(image_path)) stop("Image not found: ", image_path)

classes_df <- readr::read_csv(classes_path, show_col_types = FALSE)
classes <- classes_df$class

compose <- function(...) {
  fns <- list(...)
  function(x) {
    for (fn in fns) x <- fn(x)
    x
  }
}

mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)

infer_tf <- compose(
  transform_resize(size = 256),
  transform_center_crop(size = c(224, 224)),
  transform_to_tensor(),
  transform_normalize(mean = mean, std = std)
)

fitted <- luz_load(model_path)
net <- fitted$model
net$eval()

img <- magick_loader(image_path)
x <- infer_tf(img)
# Add batch dimension
x <- x$unsqueeze(1)

logits <- net(x)
probs <- nnf_softmax(logits, dim = 2)
probs_vec <- as.numeric(probs$squeeze())

k <- min(topk, length(probs_vec))
idx <- order(probs_vec, decreasing = TRUE)[seq_len(k)]

out <- data.frame(
  class = classes[idx],
  prob = probs_vec[idx]
)

print(out, row.names = FALSE)
