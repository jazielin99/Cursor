#!/usr/bin/env python3
"""
Python-based PSA Card Grade Predictor

This module provides card grading predictions using advanced feature extraction
and machine learning, without requiring R dependencies.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add the scripts directory to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'scripts' / 'feature_extraction'))

# Grade names mapping
GRADE_NAMES = {
    'PSA_1': 'Poor', 'PSA_2': 'Good', 'PSA_3': 'Very Good',
    'PSA_4': 'VG-EX', 'PSA_5': 'Excellent', 'PSA_6': 'EX-MT',
    'PSA_7': 'Near Mint', 'PSA_8': 'NM-MT', 'PSA_9': 'Mint',
    'PSA_10': 'Gem Mint'
}

GRADE_ORDER = ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4', 'PSA_5', 
               'PSA_6', 'PSA_7', 'PSA_8', 'PSA_9', 'PSA_10']


def extract_features(image_path: str) -> tuple:
    """Extract features from an image using the advanced feature extractor."""
    try:
        from extract_advanced_features import load_image_bgr, extract_all_features_v4
        
        img = load_image_bgr(str(image_path))
        if img is None:
            return None, None, "Could not load image"
        
        features, feature_names = extract_all_features_v4(img)
        return features, feature_names, None
    except Exception as e:
        return None, None, str(e)


def get_model_path() -> Path:
    """Get path to the Python model file."""
    return WORKSPACE_ROOT / 'models' / 'psa_python_model.pkl'


def load_model():
    """Load the trained Python model."""
    model_path = get_model_path()
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def heuristic_prediction(features: np.ndarray, feature_names: list) -> dict:
    """
    Make a prediction using feature-based heuristics when no trained model is available.
    
    Uses key features like centering, corner condition, and surface quality to estimate grade.
    """
    # Create a feature dictionary for easier access
    feat_dict = {name: val for name, val in zip(feature_names, features)}
    
    # Start with a base score of 7 (middle of the range)
    score = 7.0
    
    # --- Centering Analysis ---
    # Art-box centering is very important for high grades
    artbox_overall = feat_dict.get('artbox_overall_score', 0.5)
    artbox_lr_quality = feat_dict.get('artbox_lr_quality', 0.5)
    artbox_tb_quality = feat_dict.get('artbox_tb_quality', 0.5)
    
    # PSA 10 requires 55/45 centering
    passes_lr = feat_dict.get('artbox_passes_psa10_lr', 0)
    passes_tb = feat_dict.get('artbox_passes_psa10_tb', 0)
    
    if artbox_overall > 0.9 and passes_lr and passes_tb:
        score += 1.5  # Excellent centering
    elif artbox_overall > 0.8:
        score += 0.5  # Good centering
    elif artbox_overall < 0.6:
        score -= 1.5  # Poor centering
    elif artbox_overall < 0.7:
        score -= 0.75
    
    # --- Corner Condition ---
    # Average whitening scores across corners
    corner_whitening_scores = []
    for corner in ['tl', 'tr', 'bl', 'br']:
        ws = feat_dict.get(f'adaptive_patch_{corner}_whitening_score', 0)
        corner_whitening_scores.append(ws)
    
    avg_whitening = np.mean(corner_whitening_scores)
    max_whitening = np.max(corner_whitening_scores)
    
    if max_whitening > 0.5:
        score -= 2.0  # Severe corner damage
    elif max_whitening > 0.3:
        score -= 1.0  # Noticeable corner wear
    elif max_whitening > 0.2:
        score -= 0.5  # Minor corner wear
    elif avg_whitening < 0.1:
        score += 0.5  # Clean corners
    
    # --- Edge Quality ---
    # High resolution corner features
    edge_densities = []
    for corner in ['tl', 'tr', 'bl', 'br']:
        ed = feat_dict.get(f'hires_corner_{corner}_edge_density', 0)
        edge_densities.append(ed)
    
    avg_edge_density = np.mean(edge_densities)
    if avg_edge_density > 0.15:
        score -= 0.75  # Rough edges
    elif avg_edge_density < 0.05:
        score += 0.25  # Clean edges
    
    # --- Surface Quality ---
    texture_grad_mean = feat_dict.get('texture_grad_mean', 0.1)
    log_direct_energy = feat_dict.get('log_direct_energy', 0.1)
    
    # High texture variation can indicate surface issues
    if texture_grad_mean > 0.15:
        score -= 0.5
    elif texture_grad_mean < 0.06:
        score += 0.25
    
    # --- Color Consistency ---
    color_gray_std = feat_dict.get('color_gray_std', 0.1)
    if color_gray_std > 0.25:
        score -= 0.5  # Uneven color/fading
    
    # --- Border Quality ---
    border_consistency_range = feat_dict.get('border_consistency_range', 0.1)
    if border_consistency_range > 0.2:
        score -= 0.5  # Inconsistent borders
    
    # Clamp score to valid range
    score = max(1.0, min(10.0, score))
    
    # Convert to integer grade with some probability distribution
    base_grade = int(round(score))
    
    # Create probability distribution around the predicted grade
    probs = {}
    for i, grade in enumerate(GRADE_ORDER):
        grade_num = i + 1
        # Gaussian-like distribution centered on predicted grade
        distance = abs(grade_num - score)
        prob = np.exp(-distance * distance / 2.0)
        probs[grade] = prob
    
    # Normalize probabilities
    total = sum(probs.values())
    for grade in probs:
        probs[grade] /= total
    
    # Get predicted grade (highest probability)
    predicted_grade = max(probs, key=probs.get)
    confidence = probs[predicted_grade]
    
    return {
        'predicted_grade': predicted_grade,
        'confidence': float(confidence),
        'probabilities': probs,
        'method': 'heuristic',
        'raw_score': float(score)
    }


def model_prediction(model, features: np.ndarray, feature_names: list) -> dict:
    """Make prediction using a trained model."""
    try:
        # Get the features the model expects
        model_features = model.get('feature_names', feature_names)
        
        # Create feature array in correct order
        feat_dict = {name: val for name, val in zip(feature_names, features)}
        X = np.array([[feat_dict.get(f, 0) for f in model_features]])
        
        # Get classifier
        clf = model.get('classifier')
        if clf is None:
            return heuristic_prediction(features, feature_names)
        
        # Predict
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        
        prob_dict = {}
        for i, cls in enumerate(classes):
            # Convert numpy string to Python string
            prob_dict[str(cls)] = float(probs[i])
        
        # Ensure all grades are represented
        for grade in GRADE_ORDER:
            if grade not in prob_dict:
                prob_dict[grade] = 0.0
        
        predicted_grade = str(classes[np.argmax(probs)])
        confidence = float(np.max(probs))
        
        return {
            'predicted_grade': predicted_grade,
            'confidence': confidence,
            'probabilities': prob_dict,
            'method': 'model'
        }
    except Exception as e:
        return heuristic_prediction(features, feature_names)


def predict_grade(image_path: str) -> dict:
    """
    Main prediction function.
    
    Returns a dictionary with:
    - predicted_grade: e.g., 'PSA_8'
    - confidence: 0.0 to 1.0
    - probabilities: dict of grade -> probability
    - grade_name: e.g., 'NM-MT'
    """
    # Extract features
    features, feature_names, error = extract_features(image_path)
    
    if error:
        return {
            'predicted_grade': 'Unknown',
            'confidence': 0.0,
            'probabilities': {g: 0.0 for g in GRADE_ORDER},
            'error': error
        }
    
    # Try to load trained model
    model = load_model()
    
    if model:
        result = model_prediction(model, features, feature_names)
    else:
        result = heuristic_prediction(features, feature_names)
    
    # Add grade name
    result['grade_name'] = GRADE_NAMES.get(result['predicted_grade'], 'Unknown')
    result['image'] = str(image_path)
    
    return result


def train_quick_model(data_dir: str = None, n_samples_per_class: int = 50, output_path: str = None):
    """
    Train a quick Random Forest model on a sample of the training data.
    
    This is used to bootstrap the model when no pre-trained model exists.
    """
    from sklearn.ensemble import RandomForestClassifier
    from extract_advanced_features import load_image_bgr, extract_all_features_v4
    
    if data_dir is None:
        data_dir = WORKSPACE_ROOT / 'data' / 'training_front'
    else:
        data_dir = Path(data_dir)
    
    if output_path is None:
        output_path = get_model_path()
    else:
        output_path = Path(output_path)
    
    print(f"Training model from {data_dir}")
    print(f"Sampling {n_samples_per_class} images per class")
    
    X_all = []
    y_all = []
    feature_names = None
    
    # Process each grade directory
    for grade_dir in sorted(data_dir.iterdir()):
        if not grade_dir.is_dir() or grade_dir.name.startswith('.'):
            continue
        
        grade = grade_dir.name
        if grade not in GRADE_ORDER and grade != 'PSA_1.5':
            continue
        
        # Skip PSA_1.5 (half grades not in standard model)
        if grade == 'PSA_1.5':
            continue
        
        print(f"  Processing {grade}...")
        
        # Get image files
        images = list(grade_dir.glob('*.jpg')) + list(grade_dir.glob('*.jpeg')) + \
                 list(grade_dir.glob('*.png')) + list(grade_dir.glob('*.webp'))
        
        # Sample images
        np.random.seed(42)
        if len(images) > n_samples_per_class:
            images = np.random.choice(images, n_samples_per_class, replace=False)
        
        for img_path in images:
            try:
                img = load_image_bgr(str(img_path))
                if img is None:
                    continue
                
                feats, names = extract_all_features_v4(img)
                
                if feature_names is None:
                    feature_names = names
                
                X_all.append(feats)
                y_all.append(grade)
            except Exception as e:
                continue
    
    if len(X_all) == 0:
        print("No training data found!")
        return None
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    print(f"Training on {len(X)} samples with {X.shape[1]} features")
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X, y)
    
    # Save model
    model = {
        'classifier': clf,
        'feature_names': feature_names,
        'classes': GRADE_ORDER,
        'n_samples': len(X)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_path}")
    
    # Quick accuracy check
    train_pred = clf.predict(X)
    train_acc = np.mean(train_pred == y)
    print(f"Training accuracy: {train_acc:.2%}")
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PSA Card Grade Predictor')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--image', type=str, help='Image to predict')
    parser.add_argument('--samples', type=int, default=50, help='Samples per class for training')
    args = parser.parse_args()
    
    if args.train:
        train_quick_model(n_samples_per_class=args.samples)
    elif args.image:
        result = predict_grade(args.image)
        import json
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python predictor.py --train OR python predictor.py --image <path>")
