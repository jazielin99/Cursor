#!/usr/bin/env python3
"""
PSA Card Grading Web App
Mobile-friendly web interface for card grading predictions
"""

import os
import sys
import json
import uuid
import subprocess
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

# Path to workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_prediction(image_path):
    """Run R prediction script on image"""
    predict_script = WORKSPACE_ROOT / 'Prediction_New' / 'predict_new.R'
    
    # Create a temporary R script that sources and runs prediction
    r_code = f'''
    setwd("{WORKSPACE_ROOT}")
    .libPaths(c("~/R/library", .libPaths()))
    suppressPackageStartupMessages(source("Prediction_New/predict_new.R"))
    result <- predict_grade("{image_path}")
    cat(jsonlite::toJSON(result, auto_unbox = TRUE))
    '''
    
    try:
        result = subprocess.run(
            ['Rscript', '-e', r_code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(WORKSPACE_ROOT)
        )
        
        if result.returncode != 0:
            # Try simpler fallback prediction
            return run_simple_prediction(image_path)
        
        # Parse JSON from stdout
        output = result.stdout.strip()
        # Find JSON in output (skip any warnings)
        json_start = output.find('{')
        if json_start >= 0:
            return json.loads(output[json_start:])
        else:
            return run_simple_prediction(image_path)
            
    except subprocess.TimeoutExpired:
        return {"error": "Prediction timed out", "predicted_grade": "Unknown"}
    except Exception as e:
        return run_simple_prediction(image_path)

def run_simple_prediction(image_path):
    """Fallback: Run feature extraction and simple prediction"""
    try:
        # Extract features using Python
        sys.path.insert(0, str(WORKSPACE_ROOT / 'scripts' / 'feature_extraction'))
        from extract_advanced_features import extract_all_features
        import numpy as np
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            return {"error": "Could not read image", "predicted_grade": "Unknown"}
        
        features = extract_all_features(img)
        
        # Load model if available
        model_path = WORKSPACE_ROOT / 'models' / 'simple_rf_model.pkl'
        if model_path.exists():
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            probs = model.predict_proba([features])[0]
            grades = ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4', 'PSA_5', 
                     'PSA_6', 'PSA_7', 'PSA_8', 'PSA_9', 'PSA_10']
            pred_idx = np.argmax(probs)
            return {
                "predicted_grade": grades[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {g: float(p) for g, p in zip(grades, probs)}
            }
        else:
            # No model - return placeholder
            return {
                "predicted_grade": "PSA_7",
                "confidence": 0.5,
                "note": "Model not loaded - using placeholder",
                "features_extracted": len(features)
            }
    except Exception as e:
        return {"error": str(e), "predicted_grade": "Unknown"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)
    
    try:
        # Run prediction
        result = run_prediction(str(filepath))
        result['image_id'] = filename
        return jsonify(result)
    finally:
        # Clean up after delay (keep for display)
        pass

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route('/grades')
def grades_info():
    """Return information about PSA grades"""
    return jsonify({
        "grades": [
            {"grade": "PSA 1", "name": "Poor", "description": "Card may be missing pieces, heavily creased"},
            {"grade": "PSA 2", "name": "Good", "description": "Significant wear, staining, or damage"},
            {"grade": "PSA 3", "name": "Very Good", "description": "Well-rounded corners, moderate wear"},
            {"grade": "PSA 4", "name": "VG-EX", "description": "Corners show wear, some surface wear"},
            {"grade": "PSA 5", "name": "Excellent", "description": "Minor corner wear, good centering"},
            {"grade": "PSA 6", "name": "EX-MT", "description": "Slight corner wear, minor print spots"},
            {"grade": "PSA 7", "name": "Near Mint", "description": "Very slight corner wear, centered"},
            {"grade": "PSA 8", "name": "NM-MT", "description": "Nearly perfect, minor flaw allowed"},
            {"grade": "PSA 9", "name": "Mint", "description": "Superb condition, one minor flaw"},
            {"grade": "PSA 10", "name": "Gem Mint", "description": "Virtually perfect, pristine"},
        ]
    })

if __name__ == '__main__':
    # Run on all interfaces for mobile access
    app.run(host='0.0.0.0', port=5000, debug=True)
