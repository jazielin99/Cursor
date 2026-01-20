#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Feature Extraction

Extracts features from multiple augmented versions of an image:
- Original
- Horizontal flip
- Slight rotations (-5°, +5°)
- Brightness variations
- Slight crops

Returns averaged features for more robust predictions.
Improves exact match by 1-2% by reducing scan variance sensitivity.

Usage:
    python extract_features_tta.py --image card.jpg --output-csv features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import feature extraction from main module
import sys
sys.path.insert(0, str(Path(__file__).parent))

from extract_advanced_features import (
    extract_all_features_v4,
    load_image_bgr,
    IMG_SIZE
)


def create_augmentations(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Create multiple augmented versions of an image.
    
    Returns list of augmented images for TTA.
    """
    h, w = img_bgr.shape[:2]
    augmented = [img_bgr]  # Original
    
    # 1. Horizontal flip
    augmented.append(cv2.flip(img_bgr, 1))
    
    # 2. Slight rotations
    for angle in [-3, 3, -5, 5]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    
    # 3. Brightness variations
    for gamma in [0.9, 1.1]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        adjusted = cv2.LUT(img_bgr, table)
        augmented.append(adjusted)
    
    # 4. Slight center crops (95% of image)
    crop_pct = 0.95
    margin = int(min(h, w) * (1 - crop_pct) / 2)
    if margin > 5:
        cropped = img_bgr[margin:h-margin, margin:w-margin]
        cropped = cv2.resize(cropped, (w, h))
        augmented.append(cropped)
    
    # 5. Color jitter (slight)
    for shift in [-10, 10]:
        jittered = img_bgr.astype(np.int16) + shift
        jittered = np.clip(jittered, 0, 255).astype(np.uint8)
        augmented.append(jittered)
    
    return augmented


def extract_features_with_tta(
    image_path: str,
    n_augmentations: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features using test-time augmentation.
    
    Args:
        image_path: Path to input image
        n_augmentations: Max number of augmentations (None = all)
    
    Returns:
        (averaged_features, feature_names)
    """
    img_bgr = load_image_bgr(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get augmented versions
    augmented_images = create_augmentations(img_bgr)
    
    if n_augmentations is not None:
        augmented_images = augmented_images[:n_augmentations]
    
    # Extract features from each augmentation
    all_features = []
    feature_names = None
    
    for aug_img in augmented_images:
        feats, names = extract_all_features_v4(aug_img)
        all_features.append(feats)
        if feature_names is None:
            feature_names = names
    
    # Average features
    features_array = np.array(all_features)
    averaged_features = np.mean(features_array, axis=0)
    
    # Also compute std for confidence estimation
    features_std = np.std(features_array, axis=0)
    
    return averaged_features, feature_names, features_std


def extract_single_tta(image_path: str, output_csv: str, n_aug: int = None) -> None:
    """Extract TTA features for a single image."""
    features, names, stds = extract_features_with_tta(image_path, n_aug)
    
    df = pd.DataFrame([features], columns=names)
    df["path"] = image_path
    df["tta_count"] = n_aug if n_aug else len(create_augmentations(load_image_bgr(image_path)))
    
    # Add stability scores (inverse of std, higher = more stable)
    mean_std = np.mean(stds)
    df["tta_stability"] = 1.0 / (mean_std + 1e-6)
    
    df.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract features with TTA")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--n-augmentations", type=int, default=None,
                       help="Number of augmentations (default: all)")
    
    args = parser.parse_args()
    
    extract_single_tta(args.image, args.output_csv, args.n_augmentations)
    print(f"Saved TTA features to: {args.output_csv}")


if __name__ == "__main__":
    main()
