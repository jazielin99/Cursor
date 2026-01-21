#!/usr/bin/env python3
"""
PSA Card Grading Web App
Mobile-friendly web interface for card grading predictions
"""

import os
import sys
import json
import uuid
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

# Add webapp to path for predictor import
sys.path.insert(0, str(Path(__file__).parent))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_prediction(image_path):
    """Run Python-based prediction on image with hierarchical filtering"""
    try:
        from predictor import predict_grade, GRADE_ORDER, GRADE_NAMES
        result = predict_grade(str(image_path))
        
        # Ensure all values are JSON serializable
        clean_result = {
            'predicted_grade': str(result.get('predicted_grade', 'Unknown')),
            'confidence': float(result.get('confidence', 0.0)),
            'probabilities': {},
            'method': str(result.get('method', 'unknown')),
            'grade_name': GRADE_NAMES.get(result.get('predicted_grade', ''), 'Unknown'),
            'tier': str(result.get('tier', '')),
            'tier_reason': str(result.get('tier_reason', '')),
            'explanation': str(result.get('explanation', '')),
            'issues': result.get('issues', []),
            'positives': result.get('positives', [])
        }
        
        # Clean up probabilities
        probs = result.get('probabilities', {})
        for grade in GRADE_ORDER:
            clean_result['probabilities'][grade] = float(probs.get(grade, 0.0))
        
        if 'error' in result:
            clean_result['error'] = str(result['error'])
            
        return clean_result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e), 
            "predicted_grade": "Unknown", 
            "confidence": 0.0,
            "probabilities": {f"PSA_{i}": 0.0 for i in range(1, 11)}
        }

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

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch image uploads"""
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({"error": "No images selected"}), 400
    
    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({"filename": file.filename, "error": "Invalid file type"})
            continue
        
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Run prediction
        result = run_prediction(str(filepath))
        result['filename'] = file.filename
        result['image_id'] = filename
        results.append(result)
    
    return jsonify({"results": results, "count": len(results)})

# Feedback storage for reinforcement learning
FEEDBACK_FILE = Path(__file__).parent / 'feedback_data.json'

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on a prediction for reinforcement learning"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required = ['image_id', 'predicted_grade', 'correct_grade']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    
    feedback_entry = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'image_id': data['image_id'],
        'predicted_grade': data['predicted_grade'],
        'correct_grade': data['correct_grade'],
        'confidence': data.get('confidence', 0),
        'was_correct': data['predicted_grade'] == data['correct_grade']
    }
    
    # Load existing feedback
    feedback_list = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_list = json.load(f)
        except:
            feedback_list = []
    
    feedback_list.append(feedback_entry)
    
    # Save feedback
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=2)
    
    return jsonify({
        "status": "ok", 
        "message": "Feedback recorded for model improvement",
        "total_feedback": len(feedback_list)
    })

@app.route('/feedback/stats', methods=['GET'])
def feedback_stats():
    """Get feedback statistics"""
    if not FEEDBACK_FILE.exists():
        return jsonify({"total": 0, "correct": 0, "accuracy": 0})
    
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_list = json.load(f)
        
        total = len(feedback_list)
        correct = sum(1 for f in feedback_list if f.get('was_correct'))
        
        return jsonify({
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0
        })
    except:
        return jsonify({"total": 0, "correct": 0, "accuracy": 0})

if __name__ == '__main__':
    # Run on all interfaces for mobile access
    app.run(host='0.0.0.0', port=5000, debug=True)
