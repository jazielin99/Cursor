#!/usr/bin/env python3
"""
Advanced Feature Extraction for PSA Card Grading
Implements: HOG, LBP, Centering, Corner Sharpness, LoG, and more
"""

import os
import sys
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.filters import laplace, sobel
from skimage import img_as_float
from scipy import ndimage
import pickle
from pathlib import Path

# Configuration
IMG_SIZE = 224
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


def load_image(path, size=IMG_SIZE):
    """Load and resize image"""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, (size, size))
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_hog_features(gray):
    """
    Histogram of Oriented Gradients - detects edges and corner shapes
    PSA 10: sharp HOG signatures, PSA 6: noisy edges
    """
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True
    )
    return features


def extract_lbp_features(gray, n_bins=26):
    """
    Local Binary Patterns - texture analysis
    Detects surface wear, print quality, microscopic defects
    """
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    # Histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_centering_features(gray):
    """
    Centering Detection using Canny Edge + Contour Analysis
    Returns: top, bottom, left, right border ratios
    """
    features = []
    
    # Apply Canny edge detection
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    
    h, w = gray.shape
    
    # Find the main content region using edge density
    # Top border
    top_edge_density = []
    for i in range(h // 4):
        top_edge_density.append(np.sum(edges[i, :]) / w)
    
    # Bottom border
    bottom_edge_density = []
    for i in range(h - h // 4, h):
        bottom_edge_density.append(np.sum(edges[i, :]) / w)
    
    # Left border
    left_edge_density = []
    for j in range(w // 4):
        left_edge_density.append(np.sum(edges[:, j]) / h)
    
    # Right border
    right_edge_density = []
    for j in range(w - w // 4, w):
        right_edge_density.append(np.sum(edges[:, j]) / h)
    
    # Find where content starts (first significant edge)
    def find_content_start(density_list, threshold=10):
        for i, d in enumerate(density_list):
            if d > threshold:
                return i
        return len(density_list) // 2
    
    top_margin = find_content_start(top_edge_density)
    bottom_margin = find_content_start(bottom_edge_density[::-1])
    left_margin = find_content_start(left_edge_density)
    right_margin = find_content_start(right_edge_density[::-1])
    
    # Calculate centering ratios
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
    
    # Centering quality (0.5 is perfect)
    vertical_centering = 1 - abs(0.5 - top_ratio) * 2
    horizontal_centering = 1 - abs(0.5 - left_ratio) * 2
    overall_centering = (vertical_centering + horizontal_centering) / 2
    
    features = [
        top_ratio, bottom_ratio, left_ratio, right_ratio,
        vertical_centering, horizontal_centering, overall_centering,
        top_margin, bottom_margin, left_margin, right_margin
    ]
    
    return np.array(features)


def extract_corner_sharpness(gray):
    """
    Corner Sharpness Analysis
    Crops corners and measures contour area, edge sharpness
    PSA 10: perfect right angle, PSA 8: slight curve
    """
    h, w = gray.shape
    corner_size = int(min(h, w) * 0.15)
    
    # Extract four corners
    corners = [
        gray[:corner_size, :corner_size],           # Top-left
        gray[:corner_size, w-corner_size:],         # Top-right
        gray[h-corner_size:, :corner_size],         # Bottom-left
        gray[h-corner_size:, w-corner_size:]        # Bottom-right
    ]
    
    features = []
    
    for i, corner in enumerate(corners):
        # Convert to uint8 for contour detection
        corner_uint8 = (corner * 255).astype(np.uint8)
        
        # Edge detection
        edges = cv2.Canny(corner_uint8, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Total contour area
        total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
        
        # Edge density
        edge_density = np.sum(edges) / (corner_size * corner_size * 255)
        
        # Gradient magnitude (sharpness)
        gx = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        # Corner statistics
        mean_intensity = np.mean(corner)
        std_intensity = np.std(corner)
        white_ratio = np.mean(corner > 0.8)  # Wear shows as white
        
        features.extend([
            total_area / (corner_size * corner_size),  # Normalized contour area
            edge_density,
            np.mean(gradient_mag),
            np.std(gradient_mag),
            np.max(gradient_mag),
            mean_intensity,
            std_intensity,
            white_ratio
        ])
    
    # Corner consistency (all 4 should be similar for high grades)
    corner_means = [features[i*8 + 5] for i in range(4)]
    corner_stds = [features[i*8 + 6] for i in range(4)]
    features.extend([
        np.std(corner_means),
        np.max(corner_means) - np.min(corner_means),
        np.std(corner_stds)
    ])
    
    return np.array(features)


def extract_log_features(gray):
    """
    Laplacian of Gaussian - Surface defect detection
    Finds "blobs" and "pitted surfaces" indicating surface wear
    High LoG energy = likely PSA 6 or 7 due to surface wear
    """
    # Multi-scale LoG
    features = []
    
    for sigma in [1, 2, 3]:
        # Gaussian blur then Laplacian
        blurred = ndimage.gaussian_filter(gray, sigma)
        lap = laplace(blurred)
        
        # LoG statistics
        features.extend([
            np.mean(np.abs(lap)),      # LoG energy
            np.std(lap),               # LoG variance
            np.max(np.abs(lap)),       # Max response
            np.sum(np.abs(lap) > 0.1) / lap.size,  # High response ratio
        ])
    
    # Overall Laplacian (no Gaussian)
    lap_direct = laplace(gray)
    features.extend([
        np.mean(np.abs(lap_direct)),
        np.std(lap_direct),
        np.percentile(np.abs(lap_direct), 95)
    ])
    
    return np.array(features)


def extract_surface_texture(gray):
    """
    Additional surface texture features
    """
    features = []
    
    # Sobel gradients
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    features.extend([
        np.mean(gradient_mag),
        np.std(gradient_mag),
        np.percentile(gradient_mag, 75),
        np.percentile(gradient_mag, 95)
    ])
    
    # Gray Level Co-occurrence approximation (simplified)
    # Compute differences between adjacent pixels
    h_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    v_diff = np.abs(gray[1:, :] - gray[:-1, :])
    
    features.extend([
        np.mean(h_diff),
        np.std(h_diff),
        np.mean(v_diff),
        np.std(v_diff),
        np.mean(h_diff) + np.mean(v_diff),  # Total texture energy
    ])
    
    return np.array(features)


def extract_border_features(gray):
    """
    Border quality analysis
    """
    h, w = gray.shape
    border_width = 8
    
    borders = {
        'top': gray[:border_width, :],
        'bottom': gray[h-border_width:, :],
        'left': gray[:, :border_width],
        'right': gray[:, w-border_width:]
    }
    
    features = []
    border_means = []
    
    for name, border in borders.items():
        mean_val = np.mean(border)
        std_val = np.std(border)
        edge_response = np.mean(np.abs(np.diff(border.flatten())))
        
        features.extend([mean_val, std_val, edge_response])
        border_means.append(mean_val)
    
    # Border consistency
    features.extend([
        np.std(border_means),
        np.max(border_means) - np.min(border_means)
    ])
    
    return np.array(features)


def extract_all_features(img):
    """
    Extract all advanced features from an image
    """
    # Convert to grayscale and float
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = img_as_float(gray)
    
    # Extract all feature types
    hog_feats = extract_hog_features(gray)
    lbp_feats = extract_lbp_features(gray)
    centering_feats = extract_centering_features(gray)
    corner_feats = extract_corner_sharpness(gray)
    log_feats = extract_log_features(gray)
    texture_feats = extract_surface_texture(gray)
    border_feats = extract_border_features(gray)
    
    # Basic color/intensity features
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        color_feats = np.array([
            np.mean(r)/255, np.std(r)/255,
            np.mean(g)/255, np.std(g)/255,
            np.mean(b)/255, np.std(b)/255,
            np.mean(gray), np.std(gray)
        ])
    else:
        color_feats = np.array([np.mean(gray), np.std(gray)] * 4)
    
    # Concatenate all features
    all_features = np.concatenate([
        hog_feats,
        lbp_feats,
        centering_feats,
        corner_feats,
        log_feats,
        texture_feats,
        border_feats,
        color_feats
    ])
    
    return all_features


def process_dataset(data_dir, output_file):
    """
    Process all images in the training directory
    """
    print("=" * 50)
    print("Advanced Feature Extraction")
    print("=" * 50)
    
    data_path = Path(data_dir)
    
    all_features = []
    all_labels = []
    all_paths = []
    
    # Find all class directories
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"\nFound {len(class_dirs)} classes")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name == "NO_GRADE" or "backup" in class_name.lower():
            continue
        
        # Get all images in this class
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        image_files = [f for f in image_files if "backup" not in str(f).lower()]
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(image_files)}")
            
            img = load_image(str(img_path))
            if img is None:
                continue
            
            try:
                features = extract_all_features(img)
                all_features.append(features)
                all_labels.append(class_name)
                all_paths.append(str(img_path))
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'=' * 50}")
    print(f"Feature extraction complete!")
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"{'=' * 50}")
    
    # Feature breakdown
    print(f"\nFeature breakdown:")
    print(f"  HOG features: {len(extract_hog_features(np.zeros((IMG_SIZE, IMG_SIZE))))}")
    print(f"  LBP features: 26")
    print(f"  Centering features: 11")
    print(f"  Corner sharpness features: 35")
    print(f"  LoG features: 15")
    print(f"  Texture features: 9")
    print(f"  Border features: 14")
    print(f"  Color features: 8")
    
    # Save features
    output = {
        'X': X,
        'y': y,
        'paths': all_paths,
        'feature_names': [
            'hog', 'lbp', 'centering', 'corner_sharpness',
            'log', 'texture', 'border', 'color'
        ]
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nFeatures saved to: {output_file}")
    
    # Also save as CSV for R
    csv_file = output_file.replace('.pkl', '.csv')
    import pandas as pd
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")
    
    return X, y


if __name__ == "__main__":
    data_dir = "data/training"
    output_file = "models/advanced_features.pkl"
    
    X, y = process_dataset(data_dir, output_file)
