#!/usr/bin/env python3
"""
Memory-Efficient CNN Feature Extractor
Processes images in batches to avoid OOM
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
BATCH_SIZE = 50  # Small batches for memory


def create_feature_extractor():
    """Create MobileNetV2 feature extractor (lighter than ResNet)"""
    print("Loading MobileNetV2 as feature extractor...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    print(f"Feature vector size: {base_model.output_shape[-1]}")
    return base_model


def load_image(img_path):
    """Load and preprocess a single image"""
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        return None


def process_dataset(data_dir, output_file):
    """Process images in batches"""
    print("=" * 50)
    print("CNN Feature Extraction (Batch Mode)")
    print("=" * 50)
    
    model = create_feature_extractor()
    
    data_path = Path(data_dir)
    
    all_features = []
    all_labels = []
    
    class_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')])
    
    print(f"\nFound {len(class_dirs)} classes")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name == "NO_GRADE" or "backup" in class_name.lower():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.jpeg")) + \
                      list(class_dir.glob("*.png"))
        image_files = [f for f in image_files if "backup" not in str(f).lower()]
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        # Process in batches
        for batch_start in range(0, len(image_files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            # Load batch
            batch_images = []
            batch_valid_labels = []
            
            for img_path in batch_files:
                img = load_image(str(img_path))
                if img is not None:
                    batch_images.append(img)
                    batch_valid_labels.append(class_name)
            
            if len(batch_images) == 0:
                continue
            
            # Extract features
            batch_array = np.array(batch_images)
            batch_features = model.predict(batch_array, verbose=0)
            
            all_features.append(batch_features)
            all_labels.extend(batch_valid_labels)
            
            # Clear memory
            del batch_images, batch_array, batch_features
            gc.collect()
            
            if (batch_end) % 200 == 0 or batch_end == len(image_files):
                print(f"  {batch_end}/{len(image_files)}")
    
    # Combine all features
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'=' * 50}")
    print(f"Feature extraction complete!")
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"{'=' * 50}")
    
    # Save as pickle
    output = {'X': X, 'y': y, 'model_name': 'mobilenetv2'}
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nFeatures saved to: {output_file}")
    
    # Save as CSV
    csv_file = output_file.replace('.pkl', '.csv')
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")
    
    return X, y


if __name__ == "__main__":
    data_dir = "data/training"
    output_file = "models/cnn_features_resnet50.pkl"
    X, y = process_dataset(data_dir, output_file)
