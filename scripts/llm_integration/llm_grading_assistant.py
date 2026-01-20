#!/usr/bin/env python3
"""
LLM Integration for PSA Card Grading

This module provides infrastructure for integrating Large Language Models
to boost grading accuracy through:

1. Visual Expert Auditor - Second opinion on high-grade cards using GPT-4o/Gemini
2. Synthetic Data Augmentation - Generate training prompts for defect images
3. Automated Grading Notes - Translate numerical features into human-readable reports

IMPORTANT: Only triggered for High-Grade (8-10) tier to minimize API costs.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Optional imports for LLM providers
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

import cv2
import numpy as np


@dataclass
class GradingNote:
    """Human-readable grading explanation."""
    grade: int
    confidence: float
    centering_note: str
    corner_notes: list[str]
    surface_notes: list[str]
    edge_notes: list[str]
    overall_summary: str
    downgrade_reasons: list[str] = field(default_factory=list)
    upgrade_potential: str | None = None


@dataclass
class VisualAuditResult:
    """Result from LLM visual inspection."""
    provider: str
    corner_scores: dict[str, float]  # tl, tr, bl, br -> 1-10 score
    defects_detected: list[str]
    whitening_detected: bool
    chipping_detected: bool
    edge_crispness: float  # 1-10
    recommended_grade: int
    confidence: float
    raw_response: str
    
    
class LLMGradingAssistant:
    """
    Integrates LLM providers for enhanced card grading.
    
    Usage:
        assistant = LLMGradingAssistant(provider="openai")  # or "gemini"
        
        # Visual audit for high-grade candidate
        result = assistant.audit_high_grade_card(image_path, current_prediction=10)
        
        # Generate grading notes from features
        notes = assistant.generate_grading_notes(features_dict)
        
        # Get synthetic data prompts for training
        prompts = assistant.get_synthetic_data_prompts(grade=6, defect_type="whitening")
    """
    
    VISUAL_AUDIT_PROMPT = """You are an expert PSA card grader analyzing a trading card corner.

Inspect this trading card corner image for grading defects. Focus on:
1. WHITENING: White specks, lines, or areas along the edge where color has worn off
2. CHIPPING: Small pieces missing from the edge, creating an irregular border
3. EDGE CRISPNESS: How clean and sharp the edge appears (1=damaged, 10=perfect)
4. CORNER WEAR: Rounding, soft edges, or fraying at the corner point

On a scale of 1-10, rate the overall condition of this corner where:
- 10: Pristine, gem mint - no visible defects under magnification
- 9: Near mint - very minor imperfection only visible under magnification
- 8: Near mint-mint - slight visible wear, still excellent
- 7: Near mint - noticeable minor wear
- 6: Excellent-mint - moderate wear visible
- 5 or below: Significant defects

Respond in JSON format:
{
    "corner_score": <1-10>,
    "whitening_detected": <true/false>,
    "chipping_detected": <true/false>,
    "edge_crispness": <1-10>,
    "defects": ["list", "of", "specific", "defects"],
    "recommendation": "<brief explanation>"
}"""

    GRADING_NOTES_PROMPT = """You are a PSA card grading expert. Given the following numerical features 
from an AI analysis, generate a clear, professional grading explanation that a collector would understand.

Features:
{features}

Current predicted grade: {grade}
Confidence: {confidence}%

Generate a professional grading report explaining:
1. Why this card received this grade
2. What specific factors contributed to any downgrade from a perfect 10
3. Whether there's potential for a higher grade on resubmission

Keep the explanation concise (3-4 sentences) but informative."""

    SYNTHETIC_DATA_PROMPTS = {
        "whitening": [
            "A trading card corner showing minor edge whitening, with small white specks visible along the blue border. The whitening is subtle but noticeable under close inspection.",
            "Trading card corner with moderate whitening along the edge. White lines form where the color layer has worn away from handling.",
            "Close-up of a card corner exhibiting heavy whitening. Multiple white spots and lines visible along the entire edge length.",
        ],
        "chipping": [
            "Trading card corner with a single small chip missing from the edge. The chip creates an irregular notch in the otherwise smooth border.",
            "Card corner showing multiple tiny chips along the edge, creating a slightly rough or bumpy appearance.",
            "Heavily chipped trading card corner with several pieces of the edge layer missing, visible under normal viewing.",
        ],
        "surface_scratches": [
            "Trading card surface with a very faint hairline scratch visible only under direct light. The scratch runs diagonally across the card art.",
            "Card surface showing light scratches from sleeve wear. Multiple fine lines visible when tilted toward light.",
            "Trading card with moderate surface scratching. Clear scratch marks visible without magnification.",
        ],
        "print_defects": [
            "Trading card with minor print line running vertically through the image. The line is faint but visible.",
            "Card showing slight color shift or registration error. The printing appears slightly offset.",
            "Trading card with small ink spot or print defect in the border area.",
        ],
        "centering": [
            "Trading card with perfect 50/50 centering. Borders are exactly equal on all sides.",
            "Card with 55/45 centering shift to the left. Left border is noticeably thinner than the right.",
            "Trading card with 60/40 centering. Significant border thickness difference between top and bottom.",
            "Heavily off-center card with 70/30 centering. One border is more than twice as thick as the opposite.",
        ],
    }

    def __init__(
        self, 
        provider: Literal["openai", "gemini", "none"] = "none",
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the LLM grading assistant.
        
        Args:
            provider: LLM provider to use ("openai", "gemini", or "none" for offline)
            api_key: API key (or set via environment variable)
            model: Model name (defaults to best available for provider)
        """
        self.provider = provider
        self.enabled = provider != "none"
        
        if provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model = model or "gpt-4o"
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                print("Warning: OPENAI_API_KEY not set. LLM features will be disabled.")
                self.enabled = False
                
        elif provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            self.model = model or "gemini-1.5-pro"
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            else:
                print("Warning: GOOGLE_API_KEY not set. LLM features will be disabled.")
                self.enabled = False
        else:
            self.enabled = False
            
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for API transmission."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_corner_patches(self, image_path: str, patch_size: int = 256) -> dict[str, np.ndarray]:
        """Extract corner patches from image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        h, w = img.shape[:2]
        ps = min(patch_size, h // 2, w // 2)
        
        return {
            "tl": img[:ps, :ps],
            "tr": img[:ps, w - ps:],
            "bl": img[h - ps:, :ps],
            "br": img[h - ps:, w - ps:],
        }
    
    def _save_temp_patch(self, patch: np.ndarray) -> str:
        """Save patch to temporary file and return path."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(path, patch)
        return path
    
    def audit_corner_openai(self, corner_image_path: str) -> dict[str, Any]:
        """Audit a single corner using OpenAI GPT-4o."""
        if not self.enabled or self.provider != "openai":
            return {"error": "OpenAI not configured"}
            
        image_b64 = self._encode_image_base64(corner_image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.VISUAL_AUDIT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
        )
        
        raw = response.choices[0].message.content
        
        # Parse JSON response
        try:
            # Find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
            
        return {"raw_response": raw, "error": "Could not parse JSON"}
    
    def audit_corner_gemini(self, corner_image_path: str) -> dict[str, Any]:
        """Audit a single corner using Google Gemini."""
        if not self.enabled or self.provider != "gemini":
            return {"error": "Gemini not configured"}
            
        import PIL.Image
        img = PIL.Image.open(corner_image_path)
        
        response = self.client.generate_content([
            self.VISUAL_AUDIT_PROMPT,
            img
        ])
        
        raw = response.text
        
        # Parse JSON response
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
            
        return {"raw_response": raw, "error": "Could not parse JSON"}
    
    def audit_high_grade_card(
        self, 
        image_path: str, 
        current_prediction: int = 10,
        prediction_confidence: float = 0.95,
    ) -> VisualAuditResult | None:
        """
        Perform LLM visual audit on a high-grade candidate.
        
        IMPORTANT: Only call this for cards predicted as 8, 9, or 10 with high confidence.
        This is expensive (API costs) and slow.
        
        Args:
            image_path: Path to the card image
            current_prediction: Current model prediction (8, 9, or 10)
            prediction_confidence: Model confidence (0-1)
            
        Returns:
            VisualAuditResult with LLM assessment, or None if not enabled
        """
        if not self.enabled:
            print("LLM auditing not enabled. Set provider and API key to enable.")
            return None
            
        if current_prediction < 8:
            print("LLM audit only recommended for high-grade (8-10) predictions.")
            return None
            
        if prediction_confidence < 0.85:
            print("LLM audit only recommended for high-confidence predictions (>85%).")
            return None
        
        # Extract corner patches
        patches = self._get_corner_patches(image_path)
        
        corner_scores: dict[str, float] = {}
        all_defects: list[str] = []
        whitening_found = False
        chipping_found = False
        edge_scores: list[float] = []
        
        # Audit each corner
        for corner_name, patch in patches.items():
            temp_path = self._save_temp_patch(patch)
            
            try:
                if self.provider == "openai":
                    result = self.audit_corner_openai(temp_path)
                else:
                    result = self.audit_corner_gemini(temp_path)
                    
                if "error" not in result:
                    corner_scores[corner_name] = result.get("corner_score", 0)
                    edge_scores.append(result.get("edge_crispness", 0))
                    
                    if result.get("whitening_detected"):
                        whitening_found = True
                    if result.get("chipping_detected"):
                        chipping_found = True
                        
                    defects = result.get("defects", [])
                    all_defects.extend([f"{corner_name}: {d}" for d in defects])
                    
            finally:
                os.remove(temp_path)
        
        # Calculate recommended grade based on corner scores
        if corner_scores:
            min_score = min(corner_scores.values())
            avg_score = sum(corner_scores.values()) / len(corner_scores)
            
            # Use minimum corner score as the limiting factor
            recommended = int(min_score)
            if whitening_found and recommended > 9:
                recommended = 9
            if chipping_found and recommended > 8:
                recommended = 8
                
            confidence = avg_score / 10.0
        else:
            recommended = current_prediction
            confidence = 0.5
        
        return VisualAuditResult(
            provider=self.provider,
            corner_scores=corner_scores,
            defects_detected=all_defects,
            whitening_detected=whitening_found,
            chipping_detected=chipping_found,
            edge_crispness=sum(edge_scores) / len(edge_scores) if edge_scores else 0,
            recommended_grade=recommended,
            confidence=confidence,
            raw_response=str(corner_scores),
        )
    
    def generate_grading_notes(
        self,
        features: dict[str, float],
        predicted_grade: int,
        confidence: float,
    ) -> GradingNote:
        """
        Generate human-readable grading explanation from numerical features.
        
        This works offline (no LLM required) for basic notes, or uses LLM for enhanced notes.
        
        Args:
            features: Dictionary of feature name -> value
            predicted_grade: The predicted grade (1-10)
            confidence: Model confidence (0-1)
            
        Returns:
            GradingNote with human-readable explanation
        """
        # Extract key features for explanation
        centering_quality = features.get("artbox_overall_score", features.get("centering_overall_quality", 0.5))
        lr_ratio = features.get("artbox_lr_ratio", features.get("centering_left_ratio", 0.5))
        tb_ratio = features.get("artbox_tb_ratio", features.get("centering_top_ratio", 0.5))
        
        # Corner analysis
        corner_notes = []
        downgrade_reasons = []
        
        for corner in ["tl", "tr", "bl", "br"]:
            whitening = features.get(f"adaptive_patch_{corner}_whitening_score", 0)
            edge_density = features.get(f"adaptive_patch_{corner}_canny_edge_density", 0)
            
            if whitening > 0.5:
                corner_notes.append(f"{corner.upper()}: Whitening detected (score: {whitening:.2f})")
                if predicted_grade >= 9:
                    downgrade_reasons.append(f"Whitening at {corner.upper()} corner")
            elif whitening > 0.3:
                corner_notes.append(f"{corner.upper()}: Minor wear visible")
            else:
                corner_notes.append(f"{corner.upper()}: Good condition")
        
        # Centering note
        lr_pct = int(lr_ratio * 100)
        tb_pct = int(tb_ratio * 100)
        centering_str = f"{lr_pct}/{100-lr_pct} left/right, {tb_pct}/{100-tb_pct} top/bottom"
        
        if centering_quality > 0.9:
            centering_note = f"Excellent centering: {centering_str}"
        elif centering_quality > 0.8:
            centering_note = f"Good centering: {centering_str}"
        else:
            centering_note = f"Off-center: {centering_str}"
            if predicted_grade >= 9:
                downgrade_reasons.append(f"Centering at {centering_str}")
        
        # Surface analysis
        surface_notes = []
        texture_energy = features.get("texture_energy", 0)
        if texture_energy > 0.1:
            surface_notes.append("Surface shows signs of handling")
        else:
            surface_notes.append("Clean surface condition")
        
        # Edge analysis
        edge_notes = []
        for corner in ["tl", "tr", "bl", "br"]:
            circularity = features.get(f"corner_circularity_s0p15_{corner}", 0)
            if circularity > 0.8:
                edge_notes.append(f"{corner.upper()}: Sharp corners")
            elif circularity > 0.5:
                edge_notes.append(f"{corner.upper()}: Slight rounding")
            else:
                edge_notes.append(f"{corner.upper()}: Noticeable wear")
        
        # Overall summary
        if predicted_grade == 10:
            overall = "Gem Mint condition. All corners sharp, centering within PSA 10 standards, no visible defects."
        elif predicted_grade == 9:
            overall = "Near Mint-Mint condition. Very minor imperfections visible only under close inspection."
        elif predicted_grade == 8:
            overall = "Near Mint-Mint condition. Minor wear visible at edges or corners."
        elif predicted_grade >= 5:
            overall = "Moderate wear visible. Suitable for collectors seeking playable copies."
        else:
            overall = "Significant wear or damage. Best suited for set completion."
        
        # Upgrade potential
        upgrade_potential = None
        if predicted_grade == 9 and not downgrade_reasons:
            upgrade_potential = "This card may have potential for a 10 on resubmission."
        elif predicted_grade == 8 and confidence < 0.7:
            upgrade_potential = "Borderline case - could receive a 9 with favorable grading."
        
        return GradingNote(
            grade=predicted_grade,
            confidence=confidence,
            centering_note=centering_note,
            corner_notes=corner_notes,
            surface_notes=surface_notes,
            edge_notes=edge_notes,
            overall_summary=overall,
            downgrade_reasons=downgrade_reasons,
            upgrade_potential=upgrade_potential,
        )
    
    def get_synthetic_data_prompts(
        self,
        target_grade: int,
        defect_type: str = "general",
        count: int = 5,
    ) -> list[str]:
        """
        Get prompts for generating synthetic training data using image generation AI.
        
        Use these prompts with DALL-E 3, Midjourney, or similar to create training images
        for underrepresented defect types.
        
        Args:
            target_grade: The PSA grade to simulate (1-10)
            defect_type: Type of defect ("whitening", "chipping", "surface_scratches", 
                        "print_defects", "centering", or "general")
            count: Number of prompts to return
            
        Returns:
            List of prompts for image generation
        """
        base_prompts = self.SYNTHETIC_DATA_PROMPTS.get(defect_type, [])
        
        # Grade-specific modifiers
        if target_grade >= 9:
            severity = "extremely subtle, barely visible under magnification"
        elif target_grade >= 7:
            severity = "minor, visible only under close inspection"
        elif target_grade >= 5:
            severity = "moderate, visible during normal handling"
        else:
            severity = "significant, clearly visible"
        
        # Build prompts with grade context
        prompts = []
        for base in base_prompts[:count]:
            prompt = f"""High-resolution macro photograph of a trading card (similar to Pokemon, 
Magic: The Gathering, or sports cards). The card should show defects consistent with 
PSA grade {target_grade}. 

Specific defect: {base}

The defect severity should be {severity}. 
The image should be:
- Sharp and in focus
- Well-lit with even lighting
- Showing the card at a slight angle to reveal surface defects
- High resolution (suitable for AI training)

Style: Documentary photography, macro lens, professional card grading studio lighting."""
            prompts.append(prompt)
        
        # Add general prompts if needed
        while len(prompts) < count:
            prompts.append(f"""High-resolution photograph of a trading card corner showing wear 
consistent with PSA grade {target_grade}. The defects should be {severity}. 
Professional macro photography style with studio lighting.""")
        
        return prompts[:count]
    
    def format_grading_report(self, note: GradingNote) -> str:
        """Format a GradingNote as a printable report."""
        lines = [
            "=" * 60,
            f"PSA GRADE PREDICTION: {note.grade}",
            f"Confidence: {note.confidence:.1%}",
            "=" * 60,
            "",
            "CENTERING:",
            f"  {note.centering_note}",
            "",
            "CORNERS:",
        ]
        for cn in note.corner_notes:
            lines.append(f"  • {cn}")
        
        lines.extend(["", "SURFACE:"])
        for sn in note.surface_notes:
            lines.append(f"  • {sn}")
        
        lines.extend(["", "EDGES:"])
        for en in note.edge_notes:
            lines.append(f"  • {en}")
        
        if note.downgrade_reasons:
            lines.extend(["", "DOWNGRADE FACTORS:"])
            for dr in note.downgrade_reasons:
                lines.append(f"  ⚠ {dr}")
        
        lines.extend(["", "SUMMARY:", f"  {note.overall_summary}"])
        
        if note.upgrade_potential:
            lines.extend(["", "UPGRADE POTENTIAL:", f"  ✓ {note.upgrade_potential}"])
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Grading Assistant for PSA Cards")
    parser.add_argument("--image", help="Path to card image for visual audit")
    parser.add_argument("--provider", choices=["openai", "gemini", "none"], default="none",
                       help="LLM provider to use")
    parser.add_argument("--grade", type=int, default=9, help="Current predicted grade")
    parser.add_argument("--confidence", type=float, default=0.9, help="Model confidence")
    parser.add_argument("--synthetic-prompts", action="store_true", 
                       help="Generate synthetic data prompts")
    parser.add_argument("--defect-type", default="whitening",
                       choices=["whitening", "chipping", "surface_scratches", "print_defects", "centering"],
                       help="Defect type for synthetic prompts")
    parser.add_argument("--target-grade", type=int, default=8,
                       help="Target grade for synthetic prompts")
    
    args = parser.parse_args()
    
    assistant = LLMGradingAssistant(provider=args.provider)
    
    if args.synthetic_prompts:
        print(f"\nSynthetic Data Prompts for PSA {args.target_grade} ({args.defect_type}):\n")
        prompts = assistant.get_synthetic_data_prompts(
            target_grade=args.target_grade,
            defect_type=args.defect_type,
            count=3
        )
        for i, p in enumerate(prompts, 1):
            print(f"--- Prompt {i} ---")
            print(p)
            print()
        return
    
    if args.image:
        if assistant.enabled:
            print(f"Auditing {args.image} with {args.provider}...")
            result = assistant.audit_high_grade_card(
                args.image, 
                current_prediction=args.grade,
                prediction_confidence=args.confidence
            )
            if result:
                print(f"\nVisual Audit Result:")
                print(f"  Corner Scores: {result.corner_scores}")
                print(f"  Whitening: {result.whitening_detected}")
                print(f"  Chipping: {result.chipping_detected}")
                print(f"  Edge Crispness: {result.edge_crispness:.1f}/10")
                print(f"  Recommended Grade: {result.recommended_grade}")
                print(f"  Defects: {result.defects_detected}")
        else:
            print("LLM not configured. Generating offline grading notes...")
            # Demo with sample features
            sample_features = {
                "artbox_overall_score": 0.85,
                "artbox_lr_ratio": 0.52,
                "artbox_tb_ratio": 0.48,
                "adaptive_patch_tl_whitening_score": 0.2,
                "adaptive_patch_tr_whitening_score": 0.1,
                "adaptive_patch_bl_whitening_score": 0.15,
                "adaptive_patch_br_whitening_score": 0.25,
                "texture_energy": 0.05,
            }
            note = assistant.generate_grading_notes(sample_features, args.grade, args.confidence)
            print(assistant.format_grading_report(note))
    else:
        print("LLM Grading Assistant")
        print("=" * 40)
        print(f"Provider: {args.provider}")
        print(f"Enabled: {assistant.enabled}")
        print("\nUsage:")
        print("  --image PATH        Audit a card image")
        print("  --synthetic-prompts Generate training data prompts")
        print("\nExample:")
        print("  python llm_grading_assistant.py --image card.jpg --provider openai")


if __name__ == "__main__":
    main()
