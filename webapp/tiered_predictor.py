#!/usr/bin/env python3
"""
Tiered PSA Card Grade Predictor

Multi-specialist architecture for improved accuracy:
1. Binary Triage: Near Mint (8-10) vs Market Grade (1-7)
2. Market Grade Route: Low (1-4) vs Mid (5-7)
3. Specialist Models for each tier
4. PSA 9 vs 10 binary classifier for high-grade distinction
5. Ordinal regression blending for smoothing
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Add the scripts directory to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'scripts' / 'feature_extraction'))

GRADE_NAMES = {
    'PSA_1': 'Poor', 'PSA_2': 'Good', 'PSA_3': 'Very Good',
    'PSA_4': 'VG-EX', 'PSA_5': 'Excellent', 'PSA_6': 'EX-MT',
    'PSA_7': 'Near Mint', 'PSA_8': 'NM-MT', 'PSA_9': 'Mint',
    'PSA_10': 'Gem Mint'
}

GRADE_ORDER = ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4', 'PSA_5', 
               'PSA_6', 'PSA_7', 'PSA_8', 'PSA_9', 'PSA_10']


def extract_features(image_path: str) -> tuple:
    """Extract features from an image."""
    try:
        from extract_advanced_features import load_image_bgr, extract_all_features_v4
        
        img = load_image_bgr(str(image_path))
        if img is None:
            return None, None, "Could not load image"
        
        features, feature_names = extract_all_features_v4(img)
        return features, feature_names, None
    except Exception as e:
        return None, None, str(e)


def get_tiered_model_path() -> Path:
    """Get path to the tiered Python model file."""
    return WORKSPACE_ROOT / 'models' / 'psa_tiered_python_model.pkl'


def load_tiered_model():
    """Load the trained tiered Python model."""
    model_path = get_tiered_model_path()
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def select_important_features(X, y, n_features=400, critical_prefixes=None):
    """Select important features using Random Forest importance."""
    if critical_prefixes is None:
        critical_prefixes = ['centering_', 'corner_', 'artbox_', 'adaptive_patch_', 
                            'hires_corner_', 'whitening_', 'log_']
    
    # Train a quick RF to get importances
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Always include critical feature prefixes
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    
    critical_indices = set()
    for prefix in critical_prefixes:
        for i, name in enumerate(feature_names):
            if name.startswith(prefix):
                critical_indices.add(i)
    
    # Combine critical + top important
    selected = list(critical_indices)
    for idx in indices:
        if len(selected) >= n_features:
            break
        if idx not in selected:
            selected.append(idx)
    
    return sorted(selected[:n_features])


class TieredPredictor:
    """Multi-tier specialist model for PSA grading."""
    
    def __init__(self):
        self.binary_triage = None  # Near Mint vs Market Grade
        self.market_router = None  # Low vs Mid
        self.low_specialist = None  # PSA 1-4
        self.mid_specialist = None  # PSA 5-7
        self.high_specialist = None  # PSA 8-10
        self.psa_9v10 = None  # PSA 9 vs 10
        
        self.feature_names = None
        self.binary_features = None
        self.market_features = None
        self.low_features = None
        self.mid_features = None
        self.high_features = None
        self.psa_9v10_features = None
        
        self.tier_gamma = 2.0
        self.reg_weight = 0.2
        self.reg_sigma = 0.85
    
    def fit(self, X, y, feature_names):
        """Train all specialists."""
        import pandas as pd
        
        self.feature_names = feature_names
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['grade_num'] = df['label'].apply(lambda x: int(x.replace('PSA_', '')))
        
        # Create tier labels
        df['binary_tier'] = df['grade_num'].apply(
            lambda x: 'NearMint_8_10' if x >= 8 else 'MarketGrade_1_7'
        )
        df['market_tier'] = df['grade_num'].apply(
            lambda x: 'Low_1_4' if x <= 4 else 'Mid_5_7'
        )
        
        X_df = df[feature_names]
        
        # 1. Binary Triage (Near Mint vs Market Grade)
        print("Training Binary Triage (Near Mint vs Market Grade)...")
        self.binary_features = select_important_features(
            X_df, df['binary_tier'], n_features=400,
            critical_prefixes=['centering_', 'corner_', 'hires_corner_', 'artbox_', 
                             'log_', 'adaptive_patch_', 'whitening_']
        )
        X_binary = X_df.iloc[:, self.binary_features].values
        self.binary_triage = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.binary_triage.fit(X_binary, df['binary_tier'])
        
        # 2. Market Router (Low vs Mid) - only for market grades
        print("Training Market Router (Low vs Mid)...")
        market_mask = df['grade_num'] <= 7
        self.market_features = select_important_features(
            X_df[market_mask], df.loc[market_mask, 'market_tier'], n_features=350,
            critical_prefixes=['centering_', 'corner_', 'hires_corner_', 'log_', 
                             'texture_', 'adaptive_patch_']
        )
        X_market = X_df[market_mask].iloc[:, self.market_features].values
        self.market_router = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.market_router.fit(X_market, df.loc[market_mask, 'market_tier'])
        
        # 3. Low Specialist (PSA 1-4)
        print("Training Low Specialist (PSA 1-4)...")
        low_mask = df['grade_num'] <= 4
        self.low_features = select_important_features(
            X_df[low_mask], df.loc[low_mask, 'label'], n_features=350,
            critical_prefixes=['log_', 'texture_', 'hog_', 'lbp_', 'adaptive_patch_', 
                             'corner_', 'border_']
        )
        X_low = X_df[low_mask].iloc[:, self.low_features].values
        self.low_specialist = RandomForestClassifier(
            n_estimators=500, max_depth=25, min_samples_split=3,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.low_specialist.fit(X_low, df.loc[low_mask, 'label'])
        
        # 4. Mid Specialist (PSA 5-7)
        print("Training Mid Specialist (PSA 5-7)...")
        mid_mask = (df['grade_num'] >= 5) & (df['grade_num'] <= 7)
        self.mid_features = select_important_features(
            X_df[mid_mask], df.loc[mid_mask, 'label'], n_features=350,
            critical_prefixes=['centering_', 'corner_', 'hog_', 'log_', 
                             'adaptive_patch_', 'artbox_']
        )
        X_mid = X_df[mid_mask].iloc[:, self.mid_features].values
        self.mid_specialist = RandomForestClassifier(
            n_estimators=500, max_depth=25, min_samples_split=3,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.mid_specialist.fit(X_mid, df.loc[mid_mask, 'label'])
        
        # 5. High Specialist (PSA 8-10)
        print("Training High Specialist (PSA 8-10)...")
        high_mask = df['grade_num'] >= 8
        self.high_features = select_important_features(
            X_df[high_mask], df.loc[high_mask, 'label'], n_features=400,
            critical_prefixes=['artbox_', 'adaptive_patch_', 'hires_corner_', 
                             'corner_circularity_', 'whitening_', 'log_kurtosis_', 
                             'centering_', 'corner_']
        )
        X_high = X_df[high_mask].iloc[:, self.high_features].values
        self.high_specialist = RandomForestClassifier(
            n_estimators=700, max_depth=30, min_samples_split=3,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.high_specialist.fit(X_high, df.loc[high_mask, 'label'])
        
        # 6. PSA 9 vs 10 (Glass Ceiling Breaker)
        print("Training PSA 9 vs 10 Specialist...")
        bin_mask = df['grade_num'].isin([9, 10])
        self.psa_9v10_features = select_important_features(
            X_df[bin_mask], df.loc[bin_mask, 'label'], n_features=300,
            critical_prefixes=['artbox_', 'adaptive_patch_', 'hires_corner_', 
                             'corner_circularity_', 'whitening_', 'log_kurtosis_', 
                             'centering_']
        )
        X_9v10 = X_df[bin_mask].iloc[:, self.psa_9v10_features].values
        self.psa_9v10 = RandomForestClassifier(
            n_estimators=800, max_depth=30, min_samples_split=3,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.psa_9v10.fit(X_9v10, df.loc[bin_mask, 'label'])
        
        print("Training complete!")
        return self
    
    def predict_proba(self, features):
        """Get class probabilities using tiered architecture."""
        # Prepare feature subsets
        X_binary = features[self.binary_features].reshape(1, -1)
        X_market = features[self.market_features].reshape(1, -1)
        X_low = features[self.low_features].reshape(1, -1)
        X_mid = features[self.mid_features].reshape(1, -1)
        X_high = features[self.high_features].reshape(1, -1)
        X_9v10 = features[self.psa_9v10_features].reshape(1, -1)
        
        # Binary triage
        binary_probs = self.binary_triage.predict_proba(X_binary)[0]
        binary_classes = self.binary_triage.classes_
        
        nm_idx = np.where(binary_classes == 'NearMint_8_10')[0]
        mg_idx = np.where(binary_classes == 'MarketGrade_1_7')[0]
        
        nm_prob = binary_probs[nm_idx[0]] if len(nm_idx) > 0 else 0.5
        mg_prob = binary_probs[mg_idx[0]] if len(mg_idx) > 0 else 0.5
        
        # Sharpen
        nm_prob = nm_prob ** self.tier_gamma
        mg_prob = mg_prob ** self.tier_gamma
        total = nm_prob + mg_prob + 1e-12
        nm_prob, mg_prob = nm_prob / total, mg_prob / total
        
        # Initialize final probabilities
        final_probs = {grade: 0.0 for grade in GRADE_ORDER}
        
        # Near Mint path -> High specialist
        if nm_prob > 0.01:
            high_probs = self.high_specialist.predict_proba(X_high)[0]
            high_classes = self.high_specialist.classes_
            for i, cls in enumerate(high_classes):
                if cls in final_probs:
                    final_probs[cls] += nm_prob * high_probs[i]
        
        # Market Grade path -> Low/Mid routing
        if mg_prob > 0.01:
            market_probs = self.market_router.predict_proba(X_market)[0]
            market_classes = self.market_router.classes_
            
            low_idx = np.where(market_classes == 'Low_1_4')[0]
            mid_idx = np.where(market_classes == 'Mid_5_7')[0]
            
            low_prob = market_probs[low_idx[0]] if len(low_idx) > 0 else 0.5
            mid_prob = market_probs[mid_idx[0]] if len(mid_idx) > 0 else 0.5
            
            # Sharpen
            low_prob = low_prob ** self.tier_gamma
            mid_prob = mid_prob ** self.tier_gamma
            m_total = low_prob + mid_prob + 1e-12
            low_prob, mid_prob = low_prob / m_total, mid_prob / m_total
            
            # Low specialist
            if low_prob > 0.01:
                low_probs = self.low_specialist.predict_proba(X_low)[0]
                low_classes = self.low_specialist.classes_
                for i, cls in enumerate(low_classes):
                    if cls in final_probs:
                        final_probs[cls] += mg_prob * low_prob * low_probs[i]
            
            # Mid specialist
            if mid_prob > 0.01:
                mid_probs = self.mid_specialist.predict_proba(X_mid)[0]
                mid_classes = self.mid_specialist.classes_
                for i, cls in enumerate(mid_classes):
                    if cls in final_probs:
                        final_probs[cls] += mg_prob * mid_prob * mid_probs[i]
        
        # PSA 9 vs 10 reweighting
        mass_9_10 = final_probs.get('PSA_9', 0) + final_probs.get('PSA_10', 0)
        if mass_9_10 > 0.1:
            bin_probs = self.psa_9v10.predict_proba(X_9v10)[0]
            bin_classes = self.psa_9v10.classes_
            
            p9_idx = np.where(bin_classes == 'PSA_9')[0]
            p10_idx = np.where(bin_classes == 'PSA_10')[0]
            
            if len(p9_idx) > 0 and len(p10_idx) > 0:
                p9 = bin_probs[p9_idx[0]]
                p10 = bin_probs[p10_idx[0]]
                b_total = p9 + p10 + 1e-12
                final_probs['PSA_9'] = mass_9_10 * (p9 / b_total)
                final_probs['PSA_10'] = mass_9_10 * (p10 / b_total)
        
        # Ordinal regression smoothing
        if self.reg_weight > 0:
            # Compute expected grade from current probs
            expected = sum(
                int(g.replace('PSA_', '')) * p 
                for g, p in final_probs.items()
            )
            
            # Gaussian smoothing around expected grade
            for grade in GRADE_ORDER:
                grade_num = int(grade.replace('PSA_', ''))
                reg_prob = np.exp(-((grade_num - expected) ** 2) / (2 * self.reg_sigma ** 2))
                final_probs[grade] = (1 - self.reg_weight) * final_probs[grade] + \
                                    self.reg_weight * reg_prob
        
        # Normalize
        total = sum(final_probs.values()) + 1e-12
        for grade in final_probs:
            final_probs[grade] /= total
        
        return final_probs
    
    def predict(self, features):
        """Get predicted grade."""
        probs = self.predict_proba(features)
        return max(probs, key=probs.get)


def predict_with_tiered_dict(model_dict: dict, features: np.ndarray) -> dict:
    """Get class probabilities using tiered architecture from dict-based model."""
    tier_gamma = model_dict.get('tier_gamma', 2.0)
    reg_weight = model_dict.get('reg_weight', 0.2)
    reg_sigma = model_dict.get('reg_sigma', 0.85)
    
    # Prepare feature subsets
    X_binary = features[model_dict['binary_features']].reshape(1, -1)
    X_market = features[model_dict['market_features']].reshape(1, -1)
    X_low = features[model_dict['low_features']].reshape(1, -1)
    X_mid = features[model_dict['mid_features']].reshape(1, -1)
    X_high = features[model_dict['high_features']].reshape(1, -1)
    X_9v10 = features[model_dict['psa_9v10_features']].reshape(1, -1)
    
    binary_triage = model_dict['binary_triage']
    market_router = model_dict['market_router']
    low_specialist = model_dict['low_specialist']
    mid_specialist = model_dict['mid_specialist']
    high_specialist = model_dict['high_specialist']
    psa_9v10 = model_dict['psa_9v10']
    
    # Binary triage
    binary_probs = binary_triage.predict_proba(X_binary)[0]
    binary_classes = binary_triage.classes_
    
    nm_idx = np.where(binary_classes == 'NearMint_8_10')[0]
    mg_idx = np.where(binary_classes == 'MarketGrade_1_7')[0]
    
    nm_prob = binary_probs[nm_idx[0]] if len(nm_idx) > 0 else 0.5
    mg_prob = binary_probs[mg_idx[0]] if len(mg_idx) > 0 else 0.5
    
    # Sharpen
    nm_prob = nm_prob ** tier_gamma
    mg_prob = mg_prob ** tier_gamma
    total = nm_prob + mg_prob + 1e-12
    nm_prob, mg_prob = nm_prob / total, mg_prob / total
    
    # Initialize final probabilities
    final_probs = {grade: 0.0 for grade in GRADE_ORDER}
    
    # Near Mint path -> High specialist
    if nm_prob > 0.01:
        high_probs = high_specialist.predict_proba(X_high)[0]
        high_classes = high_specialist.classes_
        for i, cls in enumerate(high_classes):
            if cls in final_probs:
                final_probs[cls] += nm_prob * high_probs[i]
    
    # Market Grade path -> Low/Mid routing
    if mg_prob > 0.01:
        market_probs = market_router.predict_proba(X_market)[0]
        market_classes = market_router.classes_
        
        low_idx = np.where(market_classes == 'Low_1_4')[0]
        mid_idx = np.where(market_classes == 'Mid_5_7')[0]
        
        low_prob = market_probs[low_idx[0]] if len(low_idx) > 0 else 0.5
        mid_prob = market_probs[mid_idx[0]] if len(mid_idx) > 0 else 0.5
        
        # Sharpen
        low_prob = low_prob ** tier_gamma
        mid_prob = mid_prob ** tier_gamma
        m_total = low_prob + mid_prob + 1e-12
        low_prob, mid_prob = low_prob / m_total, mid_prob / m_total
        
        # Low specialist
        if low_prob > 0.01:
            low_probs = low_specialist.predict_proba(X_low)[0]
            low_classes = low_specialist.classes_
            for i, cls in enumerate(low_classes):
                if cls in final_probs:
                    final_probs[cls] += mg_prob * low_prob * low_probs[i]
        
        # Mid specialist
        if mid_prob > 0.01:
            mid_probs = mid_specialist.predict_proba(X_mid)[0]
            mid_classes = mid_specialist.classes_
            for i, cls in enumerate(mid_classes):
                if cls in final_probs:
                    final_probs[cls] += mg_prob * mid_prob * mid_probs[i]
    
    # PSA 9 vs 10 reweighting
    mass_9_10 = final_probs.get('PSA_9', 0) + final_probs.get('PSA_10', 0)
    if mass_9_10 > 0.1:
        bin_probs = psa_9v10.predict_proba(X_9v10)[0]
        bin_classes = psa_9v10.classes_
        
        p9_idx = np.where(bin_classes == 'PSA_9')[0]
        p10_idx = np.where(bin_classes == 'PSA_10')[0]
        
        if len(p9_idx) > 0 and len(p10_idx) > 0:
            p9 = bin_probs[p9_idx[0]]
            p10 = bin_probs[p10_idx[0]]
            b_total = p9 + p10 + 1e-12
            final_probs['PSA_9'] = mass_9_10 * (p9 / b_total)
            final_probs['PSA_10'] = mass_9_10 * (p10 / b_total)
    
    # Ordinal regression smoothing
    if reg_weight > 0:
        expected = sum(
            int(g.replace('PSA_', '')) * p 
            for g, p in final_probs.items()
        )
        
        for grade in GRADE_ORDER:
            grade_num = int(grade.replace('PSA_', ''))
            reg_prob = np.exp(-((grade_num - expected) ** 2) / (2 * reg_sigma ** 2))
            final_probs[grade] = (1 - reg_weight) * final_probs[grade] + \
                                reg_weight * reg_prob
    
    # Normalize
    total = sum(final_probs.values()) + 1e-12
    for grade in final_probs:
        final_probs[grade] /= total
    
    return final_probs


def predict_grade_tiered(image_path: str) -> dict:
    """
    Main prediction function using tiered model.
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
    
    # Load tiered model
    model_path = get_tiered_model_path()
    if not model_path.exists():
        return {
            'predicted_grade': 'Unknown',
            'confidence': 0.0,
            'probabilities': {g: 0.0 for g in GRADE_ORDER},
            'error': 'Tiered model not trained. Run: python tiered_predictor.py --train'
        }
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Predict using dict-based model
    probs = predict_with_tiered_dict(model_dict, features)
    predicted_grade = max(probs, key=probs.get)
    confidence = probs[predicted_grade]
    
    return {
        'predicted_grade': str(predicted_grade),
        'confidence': float(confidence),
        'probabilities': {str(k): float(v) for k, v in probs.items()},
        'method': 'tiered',
        'grade_name': GRADE_NAMES.get(predicted_grade, 'Unknown'),
        'image': str(image_path)
    }


def train_tiered_model(data_dir=None, n_samples_per_class=200, output_path=None):
    """
    Train the tiered model on training data.
    """
    from extract_advanced_features import load_image_bgr, extract_all_features_v4
    
    if data_dir is None:
        data_dir = WORKSPACE_ROOT / 'data' / 'training_front'
    else:
        data_dir = Path(data_dir)
    
    if output_path is None:
        output_path = get_tiered_model_path()
    else:
        output_path = Path(output_path)
    
    print(f"Training tiered model from {data_dir}")
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
        
        # Skip PSA_1.5
        if grade == 'PSA_1.5':
            continue
        
        print(f"  Processing {grade}...")
        
        # Get image files
        images = list(grade_dir.glob('*.jpg')) + list(grade_dir.glob('*.jpeg')) + \
                 list(grade_dir.glob('*.png')) + list(grade_dir.glob('*.webp'))
        
        # Sample images
        np.random.seed(42)
        if len(images) > n_samples_per_class:
            images = list(np.random.choice(images, n_samples_per_class, replace=False))
        
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
    
    print(f"\nTraining on {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution:")
    for grade in GRADE_ORDER:
        count = np.sum(y == grade)
        print(f"  {grade}: {count}")
    
    # Train tiered model
    model = TieredPredictor()
    model.fit(X, y, feature_names)
    
    # Convert to dict for serialization
    model_dict = {
        'binary_triage': model.binary_triage,
        'market_router': model.market_router,
        'low_specialist': model.low_specialist,
        'mid_specialist': model.mid_specialist,
        'high_specialist': model.high_specialist,
        'psa_9v10': model.psa_9v10,
        'feature_names': model.feature_names,
        'binary_features': model.binary_features,
        'market_features': model.market_features,
        'low_features': model.low_features,
        'mid_features': model.mid_features,
        'high_features': model.high_features,
        'psa_9v10_features': model.psa_9v10_features,
        'tier_gamma': model.tier_gamma,
        'reg_weight': model.reg_weight,
        'reg_sigma': model.reg_sigma,
    }
    
    # Save model as dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"\nModel saved to {output_path}")
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    correct = 0
    within1 = 0
    
    for i in range(len(X)):
        probs = predict_with_tiered_dict(model_dict, X[i])
        pred = max(probs, key=probs.get)
        true_grade = y[i]
        
        pred_num = int(pred.replace('PSA_', ''))
        true_num = int(true_grade.replace('PSA_', ''))
        
        if pred == true_grade:
            correct += 1
        if abs(pred_num - true_num) <= 1:
            within1 += 1
    
    print(f"Training Exact Match: {correct/len(X):.1%}")
    print(f"Training Within ±1: {within1/len(X):.1%}")
    
    return model_dict


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tiered PSA Card Grade Predictor')
    parser.add_argument('--train', action='store_true', help='Train a new tiered model')
    parser.add_argument('--image', type=str, help='Image to predict')
    parser.add_argument('--samples', type=int, default=200, help='Samples per class for training')
    args = parser.parse_args()
    
    if args.train:
        train_tiered_model(n_samples_per_class=args.samples)
    elif args.image:
        result = predict_grade_tiered(args.image)
        import json
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python tiered_predictor.py --train OR python tiered_predictor.py --image <path>")
