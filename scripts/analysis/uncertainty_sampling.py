#!/usr/bin/env python3
"""
Uncertainty Sampling for Active Learning

Identifies images where the model is most uncertain, which are
the most valuable for manual review and relabeling.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def compute_entropy(probs):
    """Compute entropy of probability distribution"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def compute_margin(probs):
    """Compute margin between top two predictions"""
    sorted_probs = sorted(probs, reverse=True)
    if len(sorted_probs) >= 2:
        return sorted_probs[0] - sorted_probs[1]
    return 1.0

def analyze_predictions(predictions_csv: str, output_dir: str, top_k: int = 100):
    """
    Analyze predictions and identify uncertain samples.
    
    Args:
        predictions_csv: CSV with columns: path, true_label, pred_label, and prob_PSA_1 through prob_PSA_10
        output_dir: Directory to save results
        top_k: Number of most uncertain samples to flag
    """
    
    df = pd.read_csv(predictions_csv)
    
    # Get probability columns
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    grades = [c.replace('prob_', '') for c in prob_cols]
    
    results = []
    
    for idx, row in df.iterrows():
        probs = [row[c] for c in prob_cols]
        
        # Compute uncertainty metrics
        entropy = compute_entropy(probs)
        margin = compute_margin(probs)
        max_prob = max(probs)
        
        # Get top 2 predictions
        sorted_idx = np.argsort(probs)[::-1]
        top1_grade = grades[sorted_idx[0]]
        top1_prob = probs[sorted_idx[0]]
        top2_grade = grades[sorted_idx[1]] if len(sorted_idx) > 1 else None
        top2_prob = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0
        
        # Check if prediction was correct
        is_correct = row.get('pred_label', top1_grade) == row.get('true_label', '')
        
        results.append({
            'path': row['path'],
            'true_label': row.get('true_label', ''),
            'pred_label': top1_grade,
            'top1_prob': top1_prob,
            'top2_grade': top2_grade,
            'top2_prob': top2_prob,
            'entropy': entropy,
            'margin': margin,
            'is_correct': is_correct,
            'uncertainty_score': entropy * (1 - margin)  # Combined metric
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by uncertainty (high entropy, low margin)
    results_df = results_df.sort_values('uncertainty_score', ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # All results
    results_df.to_csv(f"{output_dir}/all_predictions_with_uncertainty.csv", index=False)
    
    # Top uncertain samples
    uncertain = results_df.head(top_k)
    uncertain.to_csv(f"{output_dir}/top_uncertain_samples.csv", index=False)
    
    # Generate report
    report = []
    report.append("# Uncertainty Sampling Report\n")
    report.append(f"Total samples analyzed: {len(results_df)}\n")
    report.append(f"Top {top_k} most uncertain samples flagged for review\n\n")
    
    # Statistics
    report.append("## Uncertainty Statistics\n")
    report.append(f"- Mean entropy: {results_df['entropy'].mean():.3f}\n")
    report.append(f"- Mean margin: {results_df['margin'].mean():.3f}\n")
    report.append(f"- Mean confidence (top1_prob): {results_df['top1_prob'].mean():.3f}\n\n")
    
    # Uncertain samples by grade
    report.append("## Uncertain Samples by Predicted Grade\n")
    uncertain_by_grade = uncertain.groupby('pred_label').size().sort_values(ascending=False)
    for grade, count in uncertain_by_grade.items():
        report.append(f"- {grade}: {count} samples\n")
    
    report.append("\n## Most Common Confusion Pairs in Uncertain Samples\n")
    confusion_pairs = uncertain.groupby(['pred_label', 'top2_grade']).size().sort_values(ascending=False).head(10)
    for (g1, g2), count in confusion_pairs.items():
        report.append(f"- {g1} vs {g2}: {count} samples\n")
    
    report.append("\n## Top 20 Most Uncertain Samples\n")
    report.append("| Path | Pred | Prob | 2nd | Prob | Entropy | Margin |\n")
    report.append("|------|------|------|-----|------|---------|--------|\n")
    for _, row in uncertain.head(20).iterrows():
        path = os.path.basename(row['path'])[:30]
        report.append(f"| {path}... | {row['pred_label']} | {row['top1_prob']:.2f} | "
                     f"{row['top2_grade']} | {row['top2_prob']:.2f} | "
                     f"{row['entropy']:.2f} | {row['margin']:.2f} |\n")
    
    report.append("\n## Recommendations\n")
    report.append("1. **Review top uncertain samples**: These have the highest potential to improve model accuracy\n")
    report.append("2. **Focus on common confusion pairs**: Add more examples to distinguish between these grades\n")
    report.append("3. **Consider relabeling**: Some may be incorrectly labeled in the training set\n")
    
    with open(f"{output_dir}/uncertainty_report.md", 'w') as f:
        f.writelines(report)
    
    print(f"Saved results to {output_dir}/")
    print(f"- all_predictions_with_uncertainty.csv")
    print(f"- top_uncertain_samples.csv")
    print(f"- uncertainty_report.md")
    
    return results_df

def create_sample_predictions(data_dir: str, output_csv: str):
    """Create a sample predictions CSV for testing (placeholder)"""
    
    # This would normally come from running the model
    # For now, create a template
    grades = ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4', 'PSA_5', 
              'PSA_6', 'PSA_7', 'PSA_8', 'PSA_9', 'PSA_10']
    
    rows = []
    
    for grade_dir in Path(data_dir).iterdir():
        if grade_dir.is_dir() and grade_dir.name.startswith('PSA_'):
            true_label = grade_dir.name
            for img_path in list(grade_dir.glob('*.jpg'))[:10]:  # Sample 10 per grade
                # Simulate predictions (would come from actual model)
                probs = np.random.dirichlet(np.ones(10) * 0.5)
                pred_idx = np.argmax(probs)
                
                row = {
                    'path': str(img_path),
                    'true_label': true_label,
                    'pred_label': grades[pred_idx]
                }
                for i, g in enumerate(grades):
                    row[f'prob_{g}'] = probs[i]
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Created sample predictions: {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty sampling for active learning")
    parser.add_argument("--predictions", type=str, help="CSV with predictions and probabilities")
    parser.add_argument("--data-dir", type=str, default="data/training", help="Data directory (for creating sample)")
    parser.add_argument("--output", type=str, default="analysis/uncertainty", help="Output directory")
    parser.add_argument("--top-k", type=int, default=100, help="Number of uncertain samples to flag")
    parser.add_argument("--create-sample", action="store_true", help="Create sample predictions CSV")
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample_csv = f"{args.output}/sample_predictions.csv"
        os.makedirs(args.output, exist_ok=True)
        create_sample_predictions(args.data_dir, sample_csv)
        args.predictions = sample_csv
    
    if args.predictions:
        analyze_predictions(args.predictions, args.output, args.top_k)
    else:
        print("Please provide --predictions CSV or use --create-sample")
