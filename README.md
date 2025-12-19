# PSA card grading image model (R)

This repo contains R scripts to:

- Create a **class/folder schema** based on PSA grade + No-Grade definitions.
- Optionally **download images referenced** by PSA’s grading standards page (robots-aware).
- Train a **transfer-learning image classifier** (ResNet18 via `torch`/`torchvision`).

## Quick start

Install dependencies (installs into a writable local library at `./.Rlib`):

```bash
Rscript R/install_packages.R
```

Create the class folders (edit options inside the function call if you want to exclude half-points or No-Grade):

```bash
Rscript -e "source('R/dataset_utils.R'); create_dataset_skeleton()"
```

Put labeled images into:

```text
data/dataset/images/<CLASS_NAME>/*.jpg
```

Split into train/val:

```bash
Rscript -e "source('R/dataset_utils.R'); split_dataset_images()"
```

Train:

```bash
Rscript R/train_psa_grader.R --epochs=10 --batch_size=16
```

Predict:

```bash
Rscript R/predict_psa_grader.R --image=/path/to/card.jpg --topk=5
```

## Optional: download PSA-referenced images

```bash
Rscript R/scrape_psa_images.R
```
