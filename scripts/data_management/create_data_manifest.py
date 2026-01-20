#!/usr/bin/env python3
"""
Data Manifest Generator with Deduplication and Leakage Control

Creates a data_manifest.csv with:
- Unique identifiers for each image
- Base card ID detection (for front/back pairing)
- Near-duplicate detection via perceptual hashing
- Group IDs for cross-validation (prevents same-card leakage)
- Card type detection (Pokemon, sports, etc.)

Usage:
    python create_data_manifest.py --data-dir data/training --output data_manifest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming


def compute_phash(img_path: str, hash_size: int = 16) -> str:
    """
    Compute perceptual hash (pHash) for near-duplicate detection.
    Similar images will have similar hashes.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        
        # Resize to hash_size + 1 for DCT
        img = cv2.resize(img, (hash_size + 1, hash_size))
        
        # Compute differences
        diff = img[:, 1:] > img[:, :-1]
        
        # Convert to hex string
        return ''.join(['1' if b else '0' for b in diff.flatten()])
    except Exception:
        return ""


def compute_file_hash(img_path: str) -> str:
    """Compute MD5 hash for exact duplicate detection."""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def extract_base_id(filename: str) -> tuple[str, bool]:
    """
    Extract base card ID and detect if front/back.
    
    Returns:
        (base_id, is_back)
    """
    name = Path(filename).stem.lower()
    
    # Common patterns for front/back
    back_patterns = [r'_back$', r'_b$', r'_rear$', r'_reverse$', r'-back$', r'-b$']
    front_patterns = [r'_front$', r'_f$', r'-front$', r'-f$']
    
    is_back = any(re.search(p, name) for p in back_patterns)
    
    # Remove front/back suffix to get base ID
    base_id = name
    for pattern in back_patterns + front_patterns:
        base_id = re.sub(pattern, '', base_id)
    
    # Also try to extract card-specific identifiers
    # e.g., "psa_cert_12345678" -> "12345678"
    cert_match = re.search(r'cert[_-]?(\d+)', base_id)
    if cert_match:
        base_id = cert_match.group(1)
    
    # Remove common prefixes
    base_id = re.sub(r'^(psa|card|img|image)[_-]?', '', base_id)
    
    return base_id, is_back


def detect_card_type(img_path: str, filename: str) -> str:
    """
    Detect card type based on filename patterns and image characteristics.
    """
    name = filename.lower()
    
    # Filename-based detection
    if any(kw in name for kw in ['pokemon', 'poke', 'pikachu', 'charizard']):
        return 'pokemon'
    if any(kw in name for kw in ['baseball', 'basketball', 'football', 'hockey', 'sports']):
        return 'sports'
    if any(kw in name for kw in ['magic', 'mtg', 'yugioh', 'yu-gi-oh']):
        return 'tcg'
    
    # Could add image-based detection here (color analysis, etc.)
    return 'unknown'


def find_near_duplicates(phashes: list[str], threshold: float = 0.1) -> list[set[int]]:
    """
    Find groups of near-duplicate images based on perceptual hash similarity.
    
    Args:
        phashes: List of perceptual hashes
        threshold: Maximum hamming distance ratio for duplicates (0.1 = 10% different bits)
    
    Returns:
        List of sets, each containing indices of duplicate images
    """
    n = len(phashes)
    if n == 0:
        return []
    
    # Build adjacency list of similar images
    groups = []
    visited = set()
    
    for i in range(n):
        if i in visited or not phashes[i]:
            continue
        
        group = {i}
        visited.add(i)
        
        for j in range(i + 1, n):
            if j in visited or not phashes[j]:
                continue
            
            # Compute hamming distance
            if len(phashes[i]) != len(phashes[j]):
                continue
            
            dist = sum(c1 != c2 for c1, c2 in zip(phashes[i], phashes[j])) / len(phashes[i])
            
            if dist <= threshold:
                group.add(j)
                visited.add(j)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups


def create_manifest(data_dir: str, output_path: str, dedupe_threshold: float = 0.1) -> pd.DataFrame:
    """
    Create data manifest with deduplication and grouping.
    """
    print("=" * 60)
    print("Creating Data Manifest")
    print("=" * 60)
    
    data_path = Path(data_dir)
    
    records = []
    phashes = []
    
    # Collect all images
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    for class_dir in class_dirs:
        grade = class_dir.name
        if grade == "NO_GRADE" or "backup" in grade.lower():
            continue
        
        # Skip PSA_1.5
        if grade == "PSA_1.5":
            continue
        
        image_files = (
            list(class_dir.glob("*.jpg")) + 
            list(class_dir.glob("*.jpeg")) + 
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.webp"))
        )
        image_files = [f for f in image_files if "backup" not in str(f).lower()]
        
        print(f"Processing {grade}: {len(image_files)} images...")
        
        for img_path in image_files:
            path_str = str(img_path)
            filename = img_path.name
            
            base_id, is_back = extract_base_id(filename)
            card_type = detect_card_type(path_str, filename)
            file_hash = compute_file_hash(path_str)
            phash = compute_phash(path_str)
            
            records.append({
                'path': path_str,
                'filename': filename,
                'grade': grade,
                'grade_num': int(grade.replace('PSA_', '').replace('.5', '')),
                'base_id': base_id,
                'is_back': is_back,
                'card_type': card_type,
                'file_hash': file_hash,
                'phash': phash,
            })
            phashes.append(phash)
    
    df = pd.DataFrame(records)
    print(f"\nTotal images: {len(df)}")
    
    # Find exact duplicates (same file hash)
    print("\nFinding exact duplicates...")
    exact_dupe_groups = df.groupby('file_hash').apply(lambda x: list(x.index) if len(x) > 1 else None)
    exact_dupe_groups = [g for g in exact_dupe_groups if g is not None]
    exact_dupe_count = sum(len(g) - 1 for g in exact_dupe_groups)
    print(f"  Found {exact_dupe_count} exact duplicates in {len(exact_dupe_groups)} groups")
    
    # Find near-duplicates
    print(f"Finding near-duplicates (threshold={dedupe_threshold})...")
    near_dupe_groups = find_near_duplicates(phashes, threshold=dedupe_threshold)
    near_dupe_count = sum(len(g) - 1 for g in near_dupe_groups)
    print(f"  Found {near_dupe_count} near-duplicates in {len(near_dupe_groups)} groups")
    
    # Mark duplicates
    df['is_exact_duplicate'] = False
    df['is_near_duplicate'] = False
    df['duplicate_group'] = -1
    
    for i, group in enumerate(exact_dupe_groups):
        for idx in group[1:]:  # Keep first, mark rest as duplicates
            df.loc[idx, 'is_exact_duplicate'] = True
            df.loc[idx, 'duplicate_group'] = i
    
    for i, group in enumerate(near_dupe_groups):
        group_list = list(group)
        for idx in group_list[1:]:
            if not df.loc[idx, 'is_exact_duplicate']:
                df.loc[idx, 'is_near_duplicate'] = True
                df.loc[idx, 'duplicate_group'] = 1000 + i
    
    # Create group IDs for CV (prevents same-card leakage)
    # Group by: base_id + grade (cards with same base_id should be in same fold)
    df['cv_group'] = df['base_id'] + '_' + df['grade']
    
    # Also create a simpler group based on file similarity
    # Images with same phash prefix (first 32 bits) are grouped
    df['phash_group'] = df['phash'].apply(lambda x: x[:32] if len(x) >= 32 else x)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(df)}")
    print(f"Unique images (no exact dupes): {len(df[~df['is_exact_duplicate']])}")
    print(f"Unique images (no near dupes): {len(df[~df['is_near_duplicate'] & ~df['is_exact_duplicate']])}")
    print(f"Front images: {len(df[~df['is_back']])}")
    print(f"Back images: {len(df[df['is_back']])}")
    print(f"Unique CV groups: {df['cv_group'].nunique()}")
    
    print("\nGrade distribution:")
    print(df['grade'].value_counts().sort_index())
    
    print("\nCard type distribution:")
    print(df['card_type'].value_counts())
    
    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved manifest to: {output_path}")
    
    # Also save a clean version (no duplicates)
    clean_path = output_path.with_stem(output_path.stem + "_clean")
    df_clean = df[~df['is_exact_duplicate'] & ~df['is_near_duplicate']]
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved clean manifest (no dupes) to: {clean_path}")
    print(f"  Clean dataset: {len(df_clean)} images")
    
    return df


def create_grouped_cv_splits(manifest_path: str, n_folds: int = 5, output_path: str = None) -> pd.DataFrame:
    """
    Create grouped CV splits that prevent same-card leakage.
    Cards with the same cv_group are always in the same fold.
    """
    df = pd.read_csv(manifest_path)
    
    # Remove duplicates for training
    df = df[~df['is_exact_duplicate'] & ~df['is_near_duplicate']].copy()
    
    # Get unique groups
    groups = df['cv_group'].unique()
    np.random.seed(42)
    np.random.shuffle(groups)
    
    # Assign groups to folds
    group_to_fold = {g: i % n_folds for i, g in enumerate(groups)}
    df['cv_fold'] = df['cv_group'].map(group_to_fold)
    
    print(f"\nCreated {n_folds}-fold grouped CV splits:")
    for fold in range(n_folds):
        fold_df = df[df['cv_fold'] == fold]
        print(f"  Fold {fold}: {len(fold_df)} images, {fold_df['cv_group'].nunique()} groups")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved splits to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Create data manifest with deduplication")
    parser.add_argument("--data-dir", default="data/training", help="Training data directory")
    parser.add_argument("--output", default="data/data_manifest.csv", help="Output manifest path")
    parser.add_argument("--dedupe-threshold", type=float, default=0.1, 
                       help="pHash threshold for near-duplicate detection (0.1 = 10%% different)")
    parser.add_argument("--create-splits", action="store_true", help="Also create CV splits")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    df = create_manifest(args.data_dir, args.output, args.dedupe_threshold)
    
    if args.create_splits:
        splits_path = Path(args.output).with_stem(Path(args.output).stem + "_splits")
        create_grouped_cv_splits(args.output, args.n_folds, str(splits_path))


if __name__ == "__main__":
    main()
