#!/usr/bin/env python3
"""
Advanced Feature Extraction v3 for PSA Card Grading

Builds on v2 (engineered feature stack + v2 upgrades) and adds the
"final polish" signals:

1) Perceptual Lightness (LAB) center stats:
   - mean and std of L* channel in the card center region

2) High-res "corner attention" patches:
   - 4 fixed-size corner patches (default 256x256, clamped to image size)
   - per patch: Canny edge density, LoG energy, LoG kurtosis (safe)

NOTE:
- v2 had a fixed total feature count (6,298). v3 intentionally adds features.
- Output CSV contains stable feature names for importance selection + training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kurtosis
from skimage import img_as_float
from skimage.feature import hog, local_binary_pattern
from skimage.filters import laplace, sobel

# Baseline configuration (kept consistent with v1/v2)
IMG_SIZE = 224
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

PATCH_SIZE = 256


def load_image_bgr(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    return img


def resize_bgr(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size))


def to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_as_float(gray)


def extract_hog_features(gray_224: np.ndarray) -> np.ndarray:
    return hog(
        gray_224,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
    )


def extract_lbp_features(gray_224: np.ndarray, n_bins: int = 26) -> np.ndarray:
    # LBP expects integer-ish images; convert to uint8 for stability.
    g = np.clip(gray_224 * 255, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(g, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_centering_features(gray_224: np.ndarray) -> np.ndarray:
    edges = cv2.Canny((gray_224 * 255).astype(np.uint8), 50, 150)
    h, w = gray_224.shape

    top_edge_density = [np.sum(edges[i, :]) / w for i in range(h // 4)]
    bottom_edge_density = [np.sum(edges[i, :]) / w for i in range(h - h // 4, h)]
    left_edge_density = [np.sum(edges[:, j]) / h for j in range(w // 4)]
    right_edge_density = [np.sum(edges[:, j]) / h for j in range(w - w // 4, w)]

    def find_content_start(density_list, threshold=10):
        for i, d in enumerate(density_list):
            if d > threshold:
                return i
        return len(density_list) // 2

    top_margin = find_content_start(top_edge_density)
    bottom_margin = find_content_start(bottom_edge_density[::-1])
    left_margin = find_content_start(left_edge_density)
    right_margin = find_content_start(right_edge_density[::-1])

    total_vertical = top_margin + bottom_margin
    total_horizontal = left_margin + right_margin

    if total_vertical > 0:
        top_ratio = top_margin / total_vertical
        bottom_ratio = bottom_margin / total_vertical
    else:
        top_ratio = bottom_ratio = 0.5

    if total_horizontal > 0:
        left_ratio = left_margin / total_horizontal
        right_ratio = right_margin / total_horizontal
    else:
        left_ratio = right_ratio = 0.5

    vertical_centering = 1 - abs(0.5 - top_ratio) * 2
    horizontal_centering = 1 - abs(0.5 - left_ratio) * 2
    overall_centering = (vertical_centering + horizontal_centering) / 2

    return np.array(
        [
            top_ratio,
            bottom_ratio,
            left_ratio,
            right_ratio,
            vertical_centering,
            horizontal_centering,
            overall_centering,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
        ]
    )


def extract_corner_sharpness(gray_224: np.ndarray) -> np.ndarray:
    h, w = gray_224.shape
    corner_size = int(min(h, w) * 0.15)
    corners = [
        gray_224[:corner_size, :corner_size],
        gray_224[:corner_size, w - corner_size :],
        gray_224[h - corner_size :, :corner_size],
        gray_224[h - corner_size :, w - corner_size :],
    ]

    features: list[float] = []
    for corner in corners:
        corner_uint8 = (corner * 255).astype(np.uint8)
        edges = cv2.Canny(corner_uint8, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0.0
        edge_density = float(np.sum(edges)) / (corner_size * corner_size * 255.0)

        gx = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        mean_intensity = float(np.mean(corner))
        std_intensity = float(np.std(corner))
        white_ratio = float(np.mean(corner > 0.8))

        features.extend(
            [
                total_area / float(corner_size * corner_size),
                edge_density,
                float(np.mean(gradient_mag)),
                float(np.std(gradient_mag)),
                float(np.max(gradient_mag)),
                mean_intensity,
                std_intensity,
                white_ratio,
            ]
        )

    corner_means = [features[i * 8 + 5] for i in range(4)]
    corner_stds = [features[i * 8 + 6] for i in range(4)]
    features.extend(
        [
            float(np.std(corner_means)),
            float(np.max(corner_means) - np.min(corner_means)),
            float(np.std(corner_stds)),
        ]
    )
    return np.array(features)


def extract_log_features(gray_224: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for sigma in [1, 2, 3]:
        blurred = ndimage.gaussian_filter(gray_224, sigma)
        lap = laplace(blurred)
        features.extend(
            [
                float(np.mean(np.abs(lap))),
                float(np.std(lap)),
                float(np.max(np.abs(lap))),
                float(np.sum(np.abs(lap) > 0.1) / lap.size),
            ]
        )

    lap_direct = laplace(gray_224)
    features.extend(
        [
            float(np.mean(np.abs(lap_direct))),
            float(np.std(lap_direct)),
            float(np.percentile(np.abs(lap_direct), 95)),
        ]
    )
    return np.array(features)


def extract_surface_texture(gray_224: np.ndarray) -> np.ndarray:
    gx = sobel(gray_224, axis=1)
    gy = sobel(gray_224, axis=0)
    gradient_mag = np.sqrt(gx**2 + gy**2)

    h_diff = np.abs(gray_224[:, 1:] - gray_224[:, :-1])
    v_diff = np.abs(gray_224[1:, :] - gray_224[:-1, :])

    return np.array(
        [
            float(np.mean(gradient_mag)),
            float(np.std(gradient_mag)),
            float(np.percentile(gradient_mag, 75)),
            float(np.percentile(gradient_mag, 95)),
            float(np.mean(h_diff)),
            float(np.std(h_diff)),
            float(np.mean(v_diff)),
            float(np.std(v_diff)),
            float(np.mean(h_diff) + np.mean(v_diff)),
        ]
    )


def extract_border_features(gray_224: np.ndarray) -> np.ndarray:
    h, w = gray_224.shape
    border_width = 8
    borders = {
        "top": gray_224[:border_width, :],
        "bottom": gray_224[h - border_width :, :],
        "left": gray_224[:, :border_width],
        "right": gray_224[:, w - border_width :],
    }

    features: list[float] = []
    border_means: list[float] = []
    for border in borders.values():
        mean_val = float(np.mean(border))
        std_val = float(np.std(border))
        edge_response = float(np.mean(np.abs(np.diff(border.flatten()))))
        features.extend([mean_val, std_val, edge_response])
        border_means.append(mean_val)

    features.extend([float(np.std(border_means)), float(np.max(border_means) - np.min(border_means))])
    return np.array(features)


def extract_color_features(img_bgr_224: np.ndarray, gray_224: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr_224)
    return np.array(
        [
            float(np.mean(r) / 255.0),
            float(np.std(r) / 255.0),
            float(np.mean(g) / 255.0),
            float(np.std(g) / 255.0),
            float(np.mean(b) / 255.0),
            float(np.std(b) / 255.0),
            float(np.mean(gray_224)),
            float(np.std(gray_224)),
        ]
    )


def _corner_quadrants(gray: np.ndarray, corner_size: int):
    h, w = gray.shape
    return {
        "tl": gray[:corner_size, :corner_size],
        "tr": gray[:corner_size, w - corner_size :],
        "bl": gray[h - corner_size :, :corner_size],
        "br": gray[h - corner_size :, w - corner_size :],
    }


def _largest_contour_area_perim(edges_u8: np.ndarray) -> tuple[float, float]:
    contours, _ = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    return float(cv2.contourArea(c)), float(cv2.arcLength(c, True))


def _safe_kurtosis(x: np.ndarray) -> float:
    v = float(kurtosis(x, fisher=False, bias=False))
    return v if np.isfinite(v) else 0.0


def extract_log_kurtosis_features(gray_224: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    for sigma in [1, 2, 3]:
        blurred = ndimage.gaussian_filter(gray_224, sigma)
        lap = laplace(blurred).ravel()
        vals.append(_safe_kurtosis(lap))
    lap_direct = laplace(gray_224).ravel()
    vals.append(_safe_kurtosis(lap_direct))
    return np.array(vals)


def extract_corner_circularity_features(gray_orig: np.ndarray) -> np.ndarray:
    features: list[float] = []
    h, w = gray_orig.shape
    for pct in [0.10, 0.15, 0.20]:
        cs = max(8, int(min(h, w) * pct))
        for corner in _corner_quadrants(gray_orig, cs).values():
            corner_u8 = (np.clip(corner, 0, 1) * 255).astype(np.uint8)
            edges = cv2.Canny(corner_u8, 30, 100)
            area, perim = _largest_contour_area_perim(edges)
            circ = (4.0 * np.pi * area / (perim**2)) if perim > 1e-9 else 0.0
            features.append(float(circ))
    return np.array(features)


def extract_highres_corner_features(gray_orig: np.ndarray) -> np.ndarray:
    # Same as v2, but keeps kurtosis safe
    h, w = gray_orig.shape
    cs = max(16, int(min(h, w) * 0.15))
    corners = _corner_quadrants(gray_orig, cs)

    out: list[float] = []
    for corner in corners.values():
        c = np.clip(corner, 0, 1)
        c_u8 = (c * 255).astype(np.uint8)

        edges = cv2.Canny(c_u8, 30, 100)
        edge_density = float(np.sum(edges)) / float(cs * cs * 255.0)

        gx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3)
        gmag = np.sqrt(gx**2 + gy**2)

        mean_int = float(np.mean(c))
        std_int = float(np.std(c))
        p90 = float(np.percentile(c, 90))
        p10 = float(np.percentile(c, 10))
        white_ratio = float(np.mean(c > 0.85))
        dark_ratio = float(np.mean(c < 0.15))

        lap = laplace(c)
        lap_mean_abs = float(np.mean(np.abs(lap)))
        lap_std = float(np.std(lap))
        lap_k = _safe_kurtosis(lap.ravel())

        area, perim = _largest_contour_area_perim(edges)
        area_norm = float(area) / float(cs * cs)
        perim_norm = float(perim) / float(4 * cs)
        circularity = (4.0 * np.pi * area / (perim**2)) if perim > 1e-9 else 0.0

        hdiff = np.abs(c[:, 1:] - c[:, :-1])
        vdiff = np.abs(c[1:, :] - c[:-1, :])
        hdiff_mean = float(np.mean(hdiff))
        vdiff_mean = float(np.mean(vdiff))
        hdiff_std = float(np.std(hdiff))
        vdiff_std = float(np.std(vdiff))

        out.extend(
            [
                edge_density,
                float(np.mean(gmag)),
                float(np.std(gmag)),
                float(np.max(gmag)),
                mean_int,
                std_int,
                p90,
                p10,
                white_ratio,
                dark_ratio,
                lap_mean_abs,
                lap_std,
                lap_k,
                area_norm,
                perim_norm,
                circularity,
                hdiff_mean,
                vdiff_mean,
                hdiff_std,
                vdiff_std,
            ]
        )
    return np.array(out)


def extract_lab_center_features(img_bgr_orig: np.ndarray) -> np.ndarray:
    """
    Perceptual lightness: mean/std of L* channel in the center region.
    Center region: middle 50% x 50% of the image.
    """
    lab = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)  # 0..255
    h, w = L.shape
    r1, r2 = int(h * 0.25), int(h * 0.75)
    c1, c2 = int(w * 0.25), int(w * 0.75)
    center = L[r1:r2, c1:c2]
    return np.array([float(np.mean(center)), float(np.std(center))])


def extract_corner_patch_features(img_bgr_orig: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Fixed-size corner patches (corner attention).
    Per patch:
      - canny edge density
      - LoG energy (mean abs Laplacian)
      - LoG kurtosis (safe)
    """
    h, w = img_bgr_orig.shape[:2]
    ps = int(min(patch_size, h, w))
    if ps < 16:
        return np.zeros(12, dtype=float)

    corners = {
        "tl": img_bgr_orig[:ps, :ps],
        "tr": img_bgr_orig[:ps, w - ps :],
        "bl": img_bgr_orig[h - ps :, :ps],
        "br": img_bgr_orig[h - ps :, w - ps :],
    }

    feats: list[float] = []
    for patch in corners.values():
        gray = to_gray_float(patch)
        u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        edges = cv2.Canny(u8, 30, 100)
        edge_density = float(np.sum(edges)) / float(ps * ps * 255.0)

        lap = laplace(gray)
        log_energy = float(np.mean(np.abs(lap)))
        log_k = _safe_kurtosis(lap.ravel())

        feats.extend([edge_density, log_energy, log_k])

    return np.array(feats)


def extract_all_features_v3(img_bgr_orig: np.ndarray) -> tuple[np.ndarray, list[str]]:
    img_bgr_224 = resize_bgr(img_bgr_orig, IMG_SIZE)
    gray_224 = to_gray_float(img_bgr_224)
    gray_orig = to_gray_float(img_bgr_orig)

    hog_feats = extract_hog_features(gray_224)
    lbp_feats = extract_lbp_features(gray_224)
    centering_feats = extract_centering_features(gray_224)
    corner_feats = extract_corner_sharpness(gray_224)
    log_feats = extract_log_features(gray_224)
    texture_feats = extract_surface_texture(gray_224)
    border_feats = extract_border_features(gray_224)
    color_feats = extract_color_features(img_bgr_224, gray_224)

    # v2 additions
    log_kurt = extract_log_kurtosis_features(gray_224)  # 4
    corner_circ = extract_corner_circularity_features(gray_orig)  # 12
    hires_corner = extract_highres_corner_features(gray_orig)  # 80

    # v3 additions
    lab_center = extract_lab_center_features(img_bgr_orig)  # 2
    patch_feats = extract_corner_patch_features(img_bgr_orig)  # 12

    feats = np.concatenate(
        [
            hog_feats,
            lbp_feats,
            centering_feats,
            corner_feats,
            log_feats,
            texture_feats,
            border_feats,
            color_feats,
            log_kurt,
            corner_circ,
            hires_corner,
            lab_center,
            patch_feats,
        ]
    )

    # build names (stable)
    names: list[str] = []
    names.extend([f"hog_{i}" for i in range(hog_feats.shape[0])])
    names.extend([f"lbp_{i}" for i in range(lbp_feats.shape[0])])
    names.extend(
        [
            "centering_top_ratio",
            "centering_bottom_ratio",
            "centering_left_ratio",
            "centering_right_ratio",
            "centering_vertical_quality",
            "centering_horizontal_quality",
            "centering_overall_quality",
            "centering_top_margin",
            "centering_bottom_margin",
            "centering_left_margin",
            "centering_right_margin",
        ]
    )

    corner_pos = ["tl", "tr", "bl", "br"]
    corner_fields = [
        "contour_area_norm",
        "edge_density",
        "grad_mean",
        "grad_std",
        "grad_max",
        "mean_intensity",
        "std_intensity",
        "white_ratio",
    ]
    for p in corner_pos:
        names.extend([f"corner_{p}_{f}" for f in corner_fields])
    names.extend(
        [
            "corner_consistency_mean_std",
            "corner_consistency_mean_range",
            "corner_consistency_std_std",
        ]
    )

    for sigma in [1, 2, 3]:
        names.extend(
            [
                f"log_sigma{sigma}_energy",
                f"log_sigma{sigma}_std",
                f"log_sigma{sigma}_max",
                f"log_sigma{sigma}_high_ratio",
            ]
        )
    names.extend(["log_direct_energy", "log_direct_std", "log_direct_p95"])

    names.extend(
        [
            "texture_grad_mean",
            "texture_grad_std",
            "texture_grad_p75",
            "texture_grad_p95",
            "texture_hdiff_mean",
            "texture_hdiff_std",
            "texture_vdiff_mean",
            "texture_vdiff_std",
            "texture_energy",
        ]
    )

    border_sides = ["top", "bottom", "left", "right"]
    for s in border_sides:
        names.extend([f"border_{s}_mean", f"border_{s}_std", f"border_{s}_edge_response"])
    names.extend(["border_consistency_std", "border_consistency_range"])

    names.extend(
        [
            "color_r_mean",
            "color_r_std",
            "color_g_mean",
            "color_g_std",
            "color_b_mean",
            "color_b_std",
            "color_gray_mean",
            "color_gray_std",
        ]
    )

    names.extend(["log_kurtosis_sigma1", "log_kurtosis_sigma2", "log_kurtosis_sigma3", "log_kurtosis_direct"])

    for pct in ["0p10", "0p15", "0p20"]:
        for p in corner_pos:
            names.append(f"corner_circularity_s{pct}_{p}")

    hi_fields = [
        "edge_density",
        "grad_mean",
        "grad_std",
        "grad_max",
        "mean_intensity",
        "std_intensity",
        "p90_intensity",
        "p10_intensity",
        "white_ratio",
        "dark_ratio",
        "log_energy",
        "log_std",
        "log_kurtosis",
        "contour_area_norm",
        "contour_perim_norm",
        "circularity",
        "hdiff_mean",
        "vdiff_mean",
        "hdiff_std",
        "vdiff_std",
    ]
    for p in corner_pos:
        names.extend([f"hires_corner_{p}_{f}" for f in hi_fields])

    # v3: LAB L* center
    names.extend(["lab_center_L_mean", "lab_center_L_std"])

    # v3: fixed corner patches
    for p in corner_pos:
        names.extend(
            [
                f"patch_{p}_canny_edge_density",
                f"patch_{p}_log_energy",
                f"patch_{p}_log_kurtosis",
            ]
        )

    if len(names) != feats.shape[0]:
        raise RuntimeError(f"Feature name count mismatch: {len(names)} vs {feats.shape[0]}")

    return feats, names


def process_dataset(data_dir: str, output_base: str) -> None:
    print("=" * 60)
    print("Advanced Feature Extraction v3")
    print("=" * 60)

    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")])

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_paths: list[str] = []
    feature_names: list[str] | None = None

    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name == "NO_GRADE" or "backup" in class_name.lower():
            continue

        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.webp"))
        image_files = [f for f in image_files if "backup" not in str(f).lower()]

        print(f"\nProcessing {class_name}: {len(image_files)} images")
        for i, img_path in enumerate(image_files):
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(image_files)}")
            img_bgr = load_image_bgr(str(img_path))
            if img_bgr is None:
                continue

            feats, names = extract_all_features_v3(img_bgr)
            if feature_names is None:
                feature_names = names
            all_features.append(feats)
            all_labels.append(class_name)
            all_paths.append(str(img_path))

    if feature_names is None:
        raise SystemExit("No images processed.")

    X = np.vstack(all_features)
    y = np.array(all_labels)

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = str(base.with_suffix(".pkl"))
    csv_path = str(base.with_suffix(".csv"))

    import pickle

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "X": X,
                "y": y,
                "paths": all_paths,
                "feature_names": feature_names,
                "version": "v3",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df["path"] = all_paths
    df.to_csv(csv_path, index=False)

    print(f"\nTotal samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Saved:\n  {pkl_path}\n  {csv_path}")


def extract_single(image_path: str, output_csv: str) -> None:
    img_bgr = load_image_bgr(image_path)
    if img_bgr is None:
        raise SystemExit(f"Could not load image: {image_path}")
    feats, names = extract_all_features_v3(img_bgr)
    df = pd.DataFrame([feats], columns=names)
    df["path"] = image_path
    df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract advanced features v3 (adds LAB + patch features).")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--output-base", default="models/advanced_features_v3")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.image:
        out = args.output_csv or "models/_single_advanced_features_v3.csv"
        extract_single(args.image, out)
        print(f"Wrote: {out}")
        return

    process_dataset(args.data_dir, args.output_base)


if __name__ == "__main__":
    main()

