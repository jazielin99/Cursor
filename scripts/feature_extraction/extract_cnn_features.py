#!/usr/bin/env python3
"""
CNN Feature Extraction using MobileNetV2
Extracts 1,280-dimensional embeddings for fusion with engineered features
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def extract_cnn_features(data_dir: str, output_csv: str, batch_size: int = 32):
    """Extract MobileNetV2 features from images"""
    
    # Import TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
    except ImportError:
        print("TensorFlow not available. Installing...")
        os.system("pip install tensorflow --quiet")
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
    
    print("=" * 60)
    print("CNN Feature Extraction (MobileNetV2)")
    print("=" * 60)
    
    # Load model (without top classification layer)
    print("\nLoading MobileNetV2 model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    print(f"Model output shape: {base_model.output_shape}")  # Should be (None, 1280)
    
    # Collect all images
    data_path = Path(data_dir)
    all_images = []
    
    for grade_dir in sorted(data_path.iterdir()):
        if grade_dir.is_dir() and grade_dir.name.startswith('PSA_'):
            grade = grade_dir.name
            for img_path in grade_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    all_images.append((str(img_path), grade))
    
    print(f"\nFound {len(all_images)} images")
    
    # Process in batches
    results = []
    
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size]
        batch_imgs = []
        batch_info = []
        
        for img_path, grade in batch:
            try:
                img = keras_image.load_img(img_path, target_size=(224, 224))
                img_array = keras_image.img_to_array(img)
                img_array = preprocess_input(img_array)
                batch_imgs.append(img_array)
                batch_info.append((img_path, grade))
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
                continue
        
        if batch_imgs:
            batch_array = np.array(batch_imgs)
            features = base_model.predict(batch_array, verbose=0)
            
            for j, (img_path, grade) in enumerate(batch_info):
                row = {'path': img_path, 'label': grade}
                for k in range(features.shape[1]):
                    row[f'cnn_{k}'] = features[j, k]
                results.append(row)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(all_images))}/{len(all_images)} images")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(df)} samples with {features.shape[1]} CNN features to {output_csv}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MobileNetV2 CNN features")
    parser.add_argument("--data-dir", type=str, default="data/training_front", help="Input data directory")
    parser.add_argument("--output", type=str, default="models/cnn_features.csv", help="Output CSV")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    extract_cnn_features(args.data_dir, args.output, args.batch_size)
