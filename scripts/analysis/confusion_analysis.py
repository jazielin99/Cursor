#!/usr/bin/env python3
"""
Confusion Analysis for Targeted Data Collection

Analyzes model errors to identify:
1. Most confused grade pairs (e.g., 6↔7, 9↔10)
2. Per-grade accuracy breakdown
3. Samples to prioritize for collection

Outputs actionable recommendations for improving exact match accuracy.

Usage:
    python confusion_analysis.py --predictions predictions.csv --output analysis/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_predictions(csv_path: str) -> pd.DataFrame:
    """Load predictions CSV with columns: path, true_grade, pred_grade, confidence"""
    df = pd.read_csv(csv_path)
    required_cols = ['true_grade', 'pred_grade']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert grades to numeric
    df['true_num'] = df['true_grade'].apply(lambda x: int(str(x).replace('PSA_', '')))
    df['pred_num'] = df['pred_grade'].apply(lambda x: int(str(x).replace('PSA_', '')))
    
    return df


def compute_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute confusion matrix from predictions."""
    grades = sorted(df['true_num'].unique())
    grade_labels = [f'PSA_{g}' for g in grades]
    
    matrix = pd.DataFrame(0, index=grade_labels, columns=grade_labels)
    
    for _, row in df.iterrows():
        true = f"PSA_{row['true_num']}"
        pred = f"PSA_{row['pred_num']}"
        if true in matrix.index and pred in matrix.columns:
            matrix.loc[true, pred] += 1
    
    return matrix


def identify_confusion_pairs(conf_matrix: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Identify most confused grade pairs.
    
    Returns list of dicts with:
    - pair: (grade1, grade2)
    - errors: number of misclassifications
    - direction: which direction is more common
    """
    pairs = []
    grades = conf_matrix.index.tolist()
    
    for i, g1 in enumerate(grades):
        for g2 in grades[i+1:]:
            errors_12 = conf_matrix.loc[g1, g2]  # g1 predicted as g2
            errors_21 = conf_matrix.loc[g2, g1]  # g2 predicted as g1
            total_errors = errors_12 + errors_21
            
            if total_errors > 0:
                pairs.append({
                    'pair': (g1, g2),
                    'total_errors': total_errors,
                    'errors_forward': errors_12,
                    'errors_backward': errors_21,
                    'direction': f"{g1}→{g2}" if errors_12 > errors_21 else f"{g2}→{g1}"
                })
    
    # Sort by total errors
    pairs.sort(key=lambda x: x['total_errors'], reverse=True)
    return pairs[:top_n]


def compute_per_grade_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy metrics per grade."""
    results = []
    
    for grade in sorted(df['true_num'].unique()):
        grade_df = df[df['true_num'] == grade]
        total = len(grade_df)
        
        if total == 0:
            continue
        
        exact = (grade_df['pred_num'] == grade).sum()
        within_1 = (abs(grade_df['pred_num'] - grade) <= 1).sum()
        within_2 = (abs(grade_df['pred_num'] - grade) <= 2).sum()
        
        # Common misclassifications
        errors = grade_df[grade_df['pred_num'] != grade]['pred_num'].value_counts()
        top_error = errors.index[0] if len(errors) > 0 else None
        top_error_count = errors.iloc[0] if len(errors) > 0 else 0
        
        results.append({
            'grade': f'PSA_{grade}',
            'total': total,
            'exact_match': exact,
            'exact_pct': exact / total * 100,
            'within_1': within_1,
            'within_1_pct': within_1 / total * 100,
            'within_2': within_2,
            'within_2_pct': within_2 / total * 100,
            'top_confusion': f'PSA_{top_error}' if top_error else None,
            'top_confusion_count': top_error_count
        })
    
    return pd.DataFrame(results)


def generate_collection_recommendations(
    per_grade: pd.DataFrame,
    confusion_pairs: list[dict],
    current_counts: dict = None
) -> list[str]:
    """
    Generate recommendations for targeted data collection.
    """
    recommendations = []
    
    # 1. Low accuracy grades need more samples
    low_acc = per_grade[per_grade['exact_pct'] < 50].sort_values('exact_pct')
    for _, row in low_acc.iterrows():
        recommendations.append(
            f"PRIORITY: Collect more {row['grade']} images "
            f"(current accuracy: {row['exact_pct']:.1f}%, confused with {row['top_confusion']})"
        )
    
    # 2. High-confusion pairs need boundary examples
    for pair in confusion_pairs[:5]:
        g1, g2 = pair['pair']
        recommendations.append(
            f"BOUNDARY: Collect {g1}/{g2} boundary cases "
            f"({pair['total_errors']} errors, mainly {pair['direction']})"
        )
    
    # 3. High-grade distinctions are critical for value
    high_grade_pairs = [p for p in confusion_pairs if 'PSA_9' in p['pair'] or 'PSA_10' in p['pair']]
    if high_grade_pairs:
        for pair in high_grade_pairs[:3]:
            recommendations.append(
                f"HIGH-VALUE: Focus on {pair['pair'][0]}/{pair['pair'][1]} distinction "
                f"({pair['total_errors']} errors)"
            )
    
    return recommendations


def plot_confusion_matrix(conf_matrix: pd.DataFrame, output_path: str):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Normalize by row for better visualization
    conf_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis=0) * 100
    
    sns.heatmap(conf_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=conf_matrix.columns,
                yticklabels=conf_matrix.index)
    
    plt.title('Confusion Matrix (% of True Grade)', fontsize=14)
    plt.xlabel('Predicted Grade', fontsize=12)
    plt.ylabel('True Grade', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_by_grade(per_grade: pd.DataFrame, output_path: str):
    """Plot accuracy by grade."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(per_grade))
    width = 0.25
    
    ax.bar([i - width for i in x], per_grade['exact_pct'], width, label='Exact Match', color='#2ecc71')
    ax.bar([i for i in x], per_grade['within_1_pct'], width, label='Within 1', color='#3498db')
    ax.bar([i + width for i in x], per_grade['within_2_pct'], width, label='Within 2', color='#9b59b6')
    
    ax.set_xlabel('Grade', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Grade', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(per_grade['grade'], rotation=45)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add 60% target line
    ax.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='60% Target')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(
    df: pd.DataFrame,
    conf_matrix: pd.DataFrame,
    per_grade: pd.DataFrame,
    confusion_pairs: list[dict],
    recommendations: list[str],
    output_path: str
):
    """Generate markdown analysis report."""
    overall_exact = (df['true_num'] == df['pred_num']).mean() * 100
    overall_w1 = (abs(df['true_num'] - df['pred_num']) <= 1).mean() * 100
    overall_w2 = (abs(df['true_num'] - df['pred_num']) <= 2).mean() * 100
    
    report = f"""# Confusion Analysis Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| **Exact Match** | **{overall_exact:.1f}%** |
| Within 1 Grade | {overall_w1:.1f}% |
| Within 2 Grades | {overall_w2:.1f}% |
| Total Samples | {len(df)} |

## Target: 60%+ Exact Match

Current gap: **{max(0, 60 - overall_exact):.1f}%** improvement needed

## Per-Grade Accuracy

| Grade | Exact Match | Within 1 | Within 2 | Top Confusion |
|-------|-------------|----------|----------|---------------|
"""
    
    for _, row in per_grade.iterrows():
        report += f"| {row['grade']} | {row['exact_pct']:.1f}% | {row['within_1_pct']:.1f}% | {row['within_2_pct']:.1f}% | {row['top_confusion']} ({row['top_confusion_count']}) |\n"
    
    report += """
## Most Confused Grade Pairs

| Pair | Total Errors | Direction |
|------|--------------|-----------|
"""
    
    for pair in confusion_pairs[:10]:
        report += f"| {pair['pair'][0]} ↔ {pair['pair'][1]} | {pair['total_errors']} | {pair['direction']} |\n"
    
    report += """
## Data Collection Recommendations

"""
    
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += """
## Visualizations

- `confusion_matrix.png` - Heatmap of grade confusions
- `accuracy_by_grade.png` - Bar chart of per-grade accuracy

## Action Items

1. **Immediate**: Focus data collection on confusion pairs (6↔7, 7↔8, 8↔9, 9↔10)
2. **High-value**: Prioritize PSA 9/10 distinction (highest collector impact)
3. **Low-hanging fruit**: Add more samples for grades with <50% accuracy
4. **Quality**: Remove/relabel suspicious samples in confusion hotspots
"""
    
    with open(output_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Confusion analysis for targeted collection")
    parser.add_argument("--predictions", required=True, help="Predictions CSV path")
    parser.add_argument("--output", default="analysis", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading predictions...")
    df = load_predictions(args.predictions)
    print(f"  Loaded {len(df)} predictions")
    
    print("Computing confusion matrix...")
    conf_matrix = compute_confusion_matrix(df)
    
    print("Identifying confusion pairs...")
    confusion_pairs = identify_confusion_pairs(conf_matrix)
    
    print("Computing per-grade accuracy...")
    per_grade = compute_per_grade_accuracy(df)
    
    print("Generating recommendations...")
    recommendations = generate_collection_recommendations(per_grade, confusion_pairs)
    
    print("Generating visualizations...")
    plot_confusion_matrix(conf_matrix, str(output_dir / "confusion_matrix.png"))
    plot_accuracy_by_grade(per_grade, str(output_dir / "accuracy_by_grade.png"))
    
    print("Generating report...")
    generate_report(
        df, conf_matrix, per_grade, confusion_pairs, recommendations,
        str(output_dir / "confusion_report.md")
    )
    
    # Save data
    conf_matrix.to_csv(output_dir / "confusion_matrix.csv")
    per_grade.to_csv(output_dir / "per_grade_accuracy.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}/")
    print("\nTop Recommendations:")
    for rec in recommendations[:5]:
        print(f"  • {rec}")


if __name__ == "__main__":
    main()
