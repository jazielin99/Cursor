#!/usr/bin/env python3
"""
PSA Card Grading API Server

FastAPI backend for the iOS grading app.
Receives card images and returns grade predictions.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Or run directly:
    python api_server.py
"""

import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Try to import feature extraction
try:
    from scripts.feature_extraction.extract_advanced_features import (
        load_image_bgr, extract_all_features_v4
    )
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTION_AVAILABLE = False
    print("Warning: Feature extraction not available. Will use R subprocess.")

app = FastAPI(
    title="PSA Card Grading API",
    description="AI-powered PSA card grade prediction",
    version="1.0.0"
)

# Enable CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GradingResult(BaseModel):
    """Response model for grading predictions"""
    success: bool
    grade: Optional[str] = None
    grade_confidence: Optional[float] = None
    tier: Optional[str] = None
    tier_confidence: Optional[float] = None
    grade_probabilities: Optional[dict] = None
    grading_notes: Optional[dict] = None
    upgrade_hint: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    feature_extraction: bool
    models_available: bool


def check_models_available() -> bool:
    """Check if required model files exist"""
    models_dir = PROJECT_ROOT / "models"
    required = ["tiered_model.rds", "high_grade_specialist.rds", "psa_9_vs_10.rds"]
    return all((models_dir / f).exists() for f in required)


def run_r_prediction(image_path: str) -> dict:
    """
    Run R prediction script and parse results.
    Falls back to this when direct Python prediction isn't available.
    """
    # Create a temporary R script that returns JSON
    r_script = f'''
    suppressPackageStartupMessages({{
      library(jsonlite)
    }})
    
    setwd("{PROJECT_ROOT}")
    source("Prediction_New/predict_new.R")
    
    result <- predict_grade("{image_path}")
    
    # Convert to JSON-serializable format
    output <- list(
      success = TRUE,
      grade = result$grade,
      grade_confidence = result$grade_confidence,
      tier = result$tier,
      tier_confidence = result$tier_confidence,
      grade_probabilities = as.list(result$grade_probabilities),
      grading_notes = result$grading_notes,
      upgrade_hint = result$upgrade_hint
    )
    
    cat(toJSON(output, auto_unbox = TRUE))
    '''
    
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"R script failed: {result.stderr}"
            }
        
        # Parse JSON output
        output = result.stdout.strip()
        # Find JSON in output (skip any warnings/messages)
        json_start = output.find('{')
        if json_start >= 0:
            output = output[json_start:]
        
        return json.loads(output)
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Prediction timed out"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse result: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        feature_extraction=FEATURE_EXTRACTION_AVAILABLE,
        models_available=check_models_available()
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Alternative health check endpoint"""
    return await health_check()


@app.post("/predict", response_model=GradingResult)
async def predict_grade(image: UploadFile = File(...)):
    """
    Predict PSA grade from uploaded card image.
    
    Args:
        image: Card image file (JPEG, PNG, or WebP)
    
    Returns:
        GradingResult with predicted grade and confidence
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, or WebP)"
        )
    
    # Check models
    if not check_models_available():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    # Save uploaded file temporarily
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Run prediction
        result = run_r_prediction(tmp_path)
        
        if not result.get("success", False):
            return GradingResult(
                success=False,
                error=result.get("error", "Unknown error")
            )
        
        return GradingResult(
            success=True,
            grade=result.get("grade"),
            grade_confidence=result.get("grade_confidence"),
            tier=result.get("tier"),
            tier_confidence=result.get("tier_confidence"),
            grade_probabilities=result.get("grade_probabilities"),
            grading_notes=result.get("grading_notes"),
            upgrade_hint=result.get("upgrade_hint")
        )
        
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/predict/batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    """
    Predict grades for multiple images.
    
    Args:
        images: List of card image files
    
    Returns:
        List of GradingResults
    """
    results = []
    for image in images:
        try:
            result = await predict_grade(image)
            results.append(result.dict())
        except HTTPException as e:
            results.append({
                "success": False,
                "error": e.detail,
                "filename": image.filename
            })
    
    return {"results": results}


@app.get("/grades")
async def get_grades():
    """Return list of possible grades"""
    return {
        "grades": [
            {"value": "PSA_1", "label": "PSA 1 (Poor)", "tier": "Low"},
            {"value": "PSA_2", "label": "PSA 2 (Good)", "tier": "Low"},
            {"value": "PSA_3", "label": "PSA 3 (VG)", "tier": "Low"},
            {"value": "PSA_4", "label": "PSA 4 (VG-EX)", "tier": "Low"},
            {"value": "PSA_5", "label": "PSA 5 (EX)", "tier": "Mid"},
            {"value": "PSA_6", "label": "PSA 6 (EX-MT)", "tier": "Mid"},
            {"value": "PSA_7", "label": "PSA 7 (NM)", "tier": "Mid"},
            {"value": "PSA_8", "label": "PSA 8 (NM-MT)", "tier": "High"},
            {"value": "PSA_9", "label": "PSA 9 (MINT)", "tier": "High"},
            {"value": "PSA_10", "label": "PSA 10 (GEM MINT)", "tier": "High"},
        ]
    }


if __name__ == "__main__":
    print("Starting PSA Card Grading API Server...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Feature extraction available: {FEATURE_EXTRACTION_AVAILABLE}")
    print(f"Models available: {check_models_available()}")
    print("\nEndpoints:")
    print("  GET  /         - Health check")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Predict grade from image")
    print("  GET  /grades   - List possible grades")
    print("\nStarting server on http://0.0.0.0:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
