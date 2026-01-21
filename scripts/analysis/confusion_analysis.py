#!/usr/bin/env python3
"""
Confusion Analysis for PSA Card Grading Model

Analyzes prediction errors to identify:
1. Most confused grade pairs
2. Per-grade accuracy breakdown
3. Recommendations for targeted improvement
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_confusion(predictions_csv: str, output_dir: str):
    """Analyze predictions and generate confusion report"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(predictions_csv)
    
    # Get grades
    grades = ['PSA_1', 'PSA_2', 'PSA_3', 'PSA_4', 'PSA_5', 
              'PSA_6', 'PSA_7', 'PSA_8', 'PSA_9', 'PSA_10']
    
    # Ensure columns exist
    if 'true_label' not in df.columns or 'pred_label' not in df.columns:
        print("Error: CSV must have 'true_label' and 'pred_label' columns")
        return
    
    # Build confusion matrix
    n = len(grades)
    confusion = np.zeros((n, n), dtype=int)
    
    for _, row in df.iterrows():
        true = row['true_label']
        pred = row['pred_label']
        if true in grades and pred in grades:
            i = grades.index(true)
            j = grades.index(pred)
            confusion[i, j] += 1
    
    # Per-grade accuracy
    per_grade = []
    for i, grade in enumerate(grades):
        total = confusion[i, :].sum()
        correct = confusion[i, i]
        acc = correct / total * 100 if total > 0 else 0
        per_grade.append({
            'grade': grade,
            'correct': correct,
            'total': total,
            'accuracy': acc
        })
    
    per_grade_df = pd.DataFrame(per_grade)
    per_grade_df.to_csv(f"{output_dir}/per_grade_accuracy.csv", index=False)
    
    # Find top confusion pairs
    confusion_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and confusion[i, j] > 0:
                confusion_pairs.append({
                    'true_grade': grades[i],
                    'pred_grade': grades[j],
                    'count': confusion[i, j],
                    'pct_of_true': confusion[i, j] / confusion[i, :].sum() * 100 if confusion[i, :].sum() > 0 else 0
                })
    
    confusion_pairs_df = pd.DataFrame(confusion_pairs)
    confusion_pairs_df = confusion_pairs_df.sort_values('count', ascending=False)
    confusion_pairs_df.to_csv(f"{output_dir}/confusion_pairs.csv", index=False)
    
    # Save confusion matrix
    confusion_df = pd.DataFrame(confusion, index=grades, columns=grades)
    confusion_df.to_csv(f"{output_dir}/confusion_matrix.csv")
    
    # Generate report
    report = []
    report.append("# Confusion Analysis Report\n\n")
    
    # Overall stats
    total = len(df)
    correct = (df['true_label'] == df['pred_label']).sum()
    report.append(f"## Overall Statistics\n")
    report.append(f"- Total samples: {total}\n")
    report.append(f"- Correct predictions: {correct} ({correct/total*100:.1f}%)\n")
    report.append(f"- Errors: {total - correct} ({(total-correct)/total*100:.1f}%)\n\n")
    
    # Per-grade accuracy
    report.append("## Per-Grade Accuracy\n\n")
    report.append("| Grade | Accuracy | Correct/Total |\n")
    report.append("|-------|----------|---------------|\n")
    for row in per_grade:
        report.append(f"| {row['grade']} | {row['accuracy']:.1f}% | {row['correct']}/{row['total']} |\n")
    report.append("\n")
    
    # Top confusion pairs
    report.append("## Top 15 Confusion Pairs\n\n")
    report.append("These are the most common errors:\n\n")
    report.append("| True Grade | Predicted | Count | % of True Class |\n")
    report.append("|------------|-----------|-------|----------------|\n")
    for _, row in confusion_pairs_df.head(15).iterrows():
        report.append(f"| {row['true_grade']} | {row['pred_grade']} | {row['count']} | {row['pct_of_true']:.1f}% |\n")
    report.append("\n")
    
    # Recommendations
    report.append("## Recommendations\n\n")
    
    # Find worst grades
    worst_grades = per_grade_df.nsmallest(3, 'accuracy')
    report.append("### 1. Focus on Low-Accuracy Grades\n\n")
    for _, row in worst_grades.iterrows():
        report.append(f"- **{row['grade']}** ({row['accuracy']:.1f}%): Collect more examples, review labels\n")
    report.append("\n")
    
    # Find biggest confusion pairs
    report.append("### 2. Train Confusion-Pair Specialists\n\n")
    report.append("Add specialized models for these commonly confused pairs:\n\n")
    top_pairs = confusion_pairs_df.head(5)
    for _, row in top_pairs.iterrows():
        report.append(f"- **{row['true_grade']} vs {row['pred_grade']}**: {row['count']} errors\n")
    report.append("\n")
    
    # Adjacent vs distant errors
    adjacent_errors = 0
    distant_errors = 0
    for _, row in confusion_pairs_df.iterrows():
        true_num = int(row['true_grade'].replace('PSA_', ''))
        pred_num = int(row['pred_grade'].replace('PSA_', ''))
        if abs(true_num - pred_num) == 1:
            adjacent_errors += row['count']
        else:
            distant_errors += row['count']
    
    report.append("### 3. Error Distance Analysis\n\n")
    report.append(f"- Adjacent grade errors (off by 1): {adjacent_errors} ({adjacent_errors/(adjacent_errors+distant_errors)*100:.1f}%)\n")
    report.append(f"- Distant errors (off by 2+): {distant_errors} ({distant_errors/(adjacent_errors+distant_errors)*100:.1f}%)\n\n")
    
    if distant_errors > adjacent_errors * 0.3:
        report.append("**Warning**: High rate of distant errors suggests feature quality issues.\n")
    else:
        report.append("**Good**: Most errors are adjacent grades, suggesting reasonable feature quality.\n")
    
    # Save report
    with open(f"{output_dir}/confusion_report.md", 'w') as f:
        f.writelines(report)
    
    print(f"\nSaved analysis to {output_dir}/:")
    print("  - confusion_matrix.csv")
    print("  - per_grade_accuracy.csv")
    print("  - confusion_pairs.csv")
    print("  - confusion_report.md")
    
    # Print summary
    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)
    print(f"\nOverall Accuracy: {correct/total*100:.1f}%")
    print("\nWorst Grades:")
    for _, row in worst_grades.iterrows():
        print(f"  {row['grade']}: {row['accuracy']:.1f}%")
    print("\nTop Confusion Pairs:")
    for _, row in top_pairs.iterrows():
        print(f"  {row['true_grade']} -> {row['pred_grade']}: {row['count']} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model confusion")
    parser.add_argument("--predictions", type=str, default="models/full_pipeline_predictions.csv",
                       help="CSV with true_label and pred_label columns")
    parser.add_argument("--output", type=str, default="analysis/confusion", help="Output directory")
    
    args = parser.parse_args()
    
    if os.path.exists(args.predictions):
        analyze_confusion(args.predictions, args.output)
    else:
        print(f"Predictions file not found: {args.predictions}")
        print("Run training first to generate predictions.")
