# Confusion Analysis Report

## Overall Statistics
- Total samples: 11700
- Correct predictions: 5938 (50.8%)
- Errors: 5762 (49.2%)

## Per-Grade Accuracy

| Grade | Accuracy | Correct/Total |
|-------|----------|---------------|
| PSA_1 | 67.7% | 636/939 |
| PSA_2 | 73.6% | 560/761 |
| PSA_3 | 55.1% | 681/1236 |
| PSA_4 | 60.7% | 675/1112 |
| PSA_5 | 67.9% | 509/750 |
| PSA_6 | 25.3% | 396/1564 |
| PSA_7 | 35.0% | 531/1516 |
| PSA_8 | 74.0% | 853/1153 |
| PSA_9 | 26.9% | 374/1392 |
| PSA_10 | 56.6% | 723/1277 |

## Top 15 Confusion Pairs

These are the most common errors:

| True Grade | Predicted | Count | % of True Class |
|------------|-----------|-------|----------------|
| PSA_9 | PSA_8 | 568 | 40.8% |
| PSA_6 | PSA_8 | 495 | 31.6% |
| PSA_7 | PSA_8 | 364 | 24.0% |
| PSA_10 | PSA_8 | 337 | 26.4% |
| PSA_3 | PSA_2 | 323 | 26.1% |
| PSA_4 | PSA_5 | 311 | 28.0% |
| PSA_1 | PSA_2 | 194 | 20.7% |
| PSA_9 | PSA_10 | 193 | 13.9% |
| PSA_6 | PSA_2 | 190 | 12.1% |
| PSA_7 | PSA_3 | 169 | 11.1% |
| PSA_7 | PSA_6 | 147 | 9.7% |
| PSA_6 | PSA_3 | 140 | 9.0% |
| PSA_6 | PSA_7 | 106 | 6.8% |
| PSA_6 | PSA_5 | 91 | 5.8% |
| PSA_10 | PSA_9 | 89 | 7.0% |

## Recommendations

### 1. Focus on Low-Accuracy Grades

- **PSA_6** (25.3%): Collect more examples, review labels
- **PSA_9** (26.9%): Collect more examples, review labels
- **PSA_7** (35.0%): Collect more examples, review labels

### 2. Train Confusion-Pair Specialists

Add specialized models for these commonly confused pairs:

- **PSA_9 vs PSA_8**: 568 errors
- **PSA_6 vs PSA_8**: 495 errors
- **PSA_7 vs PSA_8**: 364 errors
- **PSA_10 vs PSA_8**: 337 errors
- **PSA_3 vs PSA_2**: 323 errors

### 3. Error Distance Analysis

- Adjacent grade errors (off by 1): 2675 (46.4%)
- Distant errors (off by 2+): 3087 (53.6%)

**Warning**: High rate of distant errors suggests feature quality issues.
