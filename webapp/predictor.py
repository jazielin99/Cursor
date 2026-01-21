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


def analyze_card_condition(features: np.ndarray, feature_names: list) -> dict:
    """
    Analyze card condition and return detailed explanations.
    """
    feat_dict = {name: val for name, val in zip(feature_names, features)}
    
    issues = []
    positives = []
    
    # --- Centering Analysis ---
    artbox_overall = feat_dict.get('artbox_overall_score', 0.5)
    artbox_lr = feat_dict.get('artbox_lr_ratio', 0.5)
    artbox_tb = feat_dict.get('artbox_tb_ratio', 0.5)
    passes_lr = feat_dict.get('artbox_passes_psa10_lr', 0)
    passes_tb = feat_dict.get('artbox_passes_psa10_tb', 0)
    
    lr_pct = int(artbox_lr * 100)
    tb_pct = int(artbox_tb * 100)
    centering_str = f"{lr_pct}/{100-lr_pct} L/R, {tb_pct}/{100-tb_pct} T/B"
    
    if artbox_overall > 0.9 and passes_lr and passes_tb:
        positives.append(f"Excellent centering ({centering_str})")
        centering_score = 2
    elif artbox_overall > 0.8:
        positives.append(f"Good centering ({centering_str})")
        centering_score = 1
    elif artbox_overall < 0.6:
        issues.append(f"Poor centering ({centering_str}) - significant off-center")
        centering_score = -2
    elif artbox_overall < 0.7:
        issues.append(f"Off-center ({centering_str})")
        centering_score = -1
    else:
        centering_score = 0
    
    # --- Corner Analysis ---
    corner_issues = []
    corner_whitening = []
    for corner, name in [('tl', 'Top-Left'), ('tr', 'Top-Right'), ('bl', 'Bottom-Left'), ('br', 'Bottom-Right')]:
        ws = feat_dict.get(f'adaptive_patch_{corner}_whitening_score', 0)
        corner_whitening.append(ws)
        if ws > 0.5:
            corner_issues.append(f"{name}: severe whitening/wear")
        elif ws > 0.3:
            corner_issues.append(f"{name}: visible wear")
    
    max_whitening = max(corner_whitening)
    avg_whitening = np.mean(corner_whitening)
    
    if max_whitening > 0.5:
        issues.append("Corner damage detected - " + "; ".join(corner_issues))
        corner_score = -3
    elif max_whitening > 0.3:
        issues.append("Corner wear visible - " + "; ".join(corner_issues))
        corner_score = -2
    elif max_whitening > 0.2:
        issues.append("Minor corner wear")
        corner_score = -1
    elif avg_whitening < 0.1:
        positives.append("Sharp, clean corners")
        corner_score = 1
    else:
        corner_score = 0
    
    # --- Surface/Texture Analysis ---
    texture_grad = feat_dict.get('texture_grad_mean', 0.1)
    log_energy = feat_dict.get('log_direct_energy', 0.1)
    
    if texture_grad > 0.2 or log_energy > 0.15:
        issues.append("Surface damage/scratches detected")
        surface_score = -2
    elif texture_grad > 0.15:
        issues.append("Minor surface wear")
        surface_score = -1
    elif texture_grad < 0.06:
        positives.append("Clean, smooth surface")
        surface_score = 1
    else:
        surface_score = 0
    
    # --- Edge Analysis ---
    edge_densities = [feat_dict.get(f'hires_corner_{c}_edge_density', 0) for c in ['tl', 'tr', 'bl', 'br']]
    avg_edge = np.mean(edge_densities)
    
    if avg_edge > 0.2:
        issues.append("Rough/damaged edges")
        edge_score = -2
    elif avg_edge > 0.15:
        issues.append("Minor edge wear")
        edge_score = -1
    elif avg_edge < 0.05:
        positives.append("Clean edges")
        edge_score = 1
    else:
        edge_score = 0
    
    # --- Color/Print Quality ---
    color_std = feat_dict.get('color_gray_std', 0.1)
    if color_std > 0.3:
        issues.append("Color fading or staining")
        color_score = -1
    else:
        color_score = 0
    
    # Calculate tier
    total_score = centering_score + corner_score + surface_score + edge_score + color_score
    
    if total_score >= 3:
        tier = "NearMint_8_10"
        tier_reason = "Card shows excellent condition"
    elif total_score >= 0:
        tier = "Mid_5_7"
        tier_reason = "Card shows moderate wear"
    else:
        tier = "Low_1_4"
        tier_reason = "Card shows significant wear/damage"
    
    return {
        'issues': issues,
        'positives': positives,
        'tier': tier,
        'tier_reason': tier_reason,
        'scores': {
            'centering': centering_score,
            'corners': corner_score,
            'surface': surface_score,
            'edges': edge_score,
            'color': color_score,
            'total': total_score
        }
    }


def hierarchical_prediction(features: np.ndarray, feature_names: list, model=None) -> dict:
    """
    Hierarchical prediction that rules out unlikely grades based on condition analysis.
    """
    # Analyze condition first
    analysis = analyze_card_condition(features, feature_names)
    tier = analysis['tier']
    
    # Get model probabilities if available
    if model is not None:
        clf = model.get('classifier')
        model_features = model.get('feature_names', feature_names)
        feat_dict = {name: val for name, val in zip(feature_names, features)}
        X = np.array([[feat_dict.get(f, 0) for f in model_features]])
        
        raw_probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        probs = {str(cls): float(p) for cls, p in zip(classes, raw_probs)}
    else:
        # Use heuristic
        probs = {g: 0.1 for g in GRADE_ORDER}
    
    # Apply hierarchical filtering - zero out impossible grades
    if tier == "Low_1_4":
        # Low condition - can't be 8-10
        for g in ['PSA_8', 'PSA_9', 'PSA_10']:
            probs[g] = 0.0
    elif tier == "Mid_5_7":
        # Mid condition - unlikely to be 1-2 or 10
        probs['PSA_1'] *= 0.1
        probs['PSA_2'] *= 0.3
        probs['PSA_10'] *= 0.2
    elif tier == "NearMint_8_10":
        # High condition - can't be 1-4
        for g in ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4']:
            probs[g] = 0.0
    
    # Ensure all grades present
    for g in GRADE_ORDER:
        if g not in probs:
            probs[g] = 0.0
    
    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {g: p/total for g, p in probs.items()}
    
    predicted_grade = max(probs, key=probs.get)
    confidence = probs[predicted_grade]
    
    # Build explanation
    explanation_parts = []
    if analysis['issues']:
        explanation_parts.append("Issues: " + "; ".join(analysis['issues'][:3]))
    if analysis['positives']:
        explanation_parts.append("Strengths: " + "; ".join(analysis['positives'][:3]))
    
    explanation = " | ".join(explanation_parts) if explanation_parts else "Standard condition"
    
    return {
        'predicted_grade': predicted_grade,
        'confidence': float(confidence),
        'probabilities': probs,
        'tier': tier,
        'tier_reason': analysis['tier_reason'],
        'explanation': explanation,
        'issues': analysis['issues'],
        'positives': analysis['positives'],
        'condition_scores': analysis['scores']
    }


def heuristic_prediction(features: np.ndarray, feature_names: list) -> dict:
    """
    Make a prediction using feature-based heuristics when no trained model is available.
    """
    return hierarchical_prediction(features, feature_names, model=None)


def model_prediction(model, features: np.ndarray, feature_names: list) -> dict:
    """Make prediction using a trained model with hierarchical filtering."""
    try:
        # Use hierarchical prediction which applies tier filtering
        result = hierarchical_prediction(features, feature_names, model=model)
        result['method'] = 'model'
        return result
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
