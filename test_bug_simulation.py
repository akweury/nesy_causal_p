#!/usr/bin/env python3
"""Test to verify the bug analysis"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Simulate the exact scenario: model predicts all negatives, ~35% of labels are negative
np.random.seed(42)
n_samples = 1000

# 35% negative (0), 65% positive (1) - true labels
true_labels = np.concatenate([np.zeros(350), np.ones(650)])
np.random.shuffle(true_labels)

# Model predicts all negative (0)
predictions = np.zeros(n_samples)

# Calculate metrics
acc = accuracy_score(true_labels, predictions)
prec = precision_score(true_labels, predictions, zero_division=0)
rec = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)

# Confusion matrix
tp = int(((predictions == 1) & (true_labels == 1)).sum())
fp = int(((predictions == 1) & (true_labels == 0)).sum())
tn = int(((predictions == 0) & (true_labels == 0)).sum())
fn = int(((predictions == 0) & (true_labels == 1)).sum())

print('='*70)
print('SIMULATION: Model Predicts ALL Negative (All Pairs = Different Groups)')
print('='*70)
print(f'\nTrue Distribution:')
print(f'  Positive (same group):     {int(true_labels.sum())}')
print(f'  Negative (different group): {n_samples - int(true_labels.sum())}')
print(f'\nModel Predictions:')
print(f'  Positive: {int(predictions.sum())} (0.0%)')
print(f'  Negative: {n_samples - int(predictions.sum())} (100.0%)')

print(f'\nConfusion Matrix:')
print(f'  True Positives (TP):  {tp}')
print(f'  False Positives (FP): {fp}')
print(f'  True Negatives (TN):  {tn}')
print(f'  False Negatives (FN): {fn}')

print(f'\nMetrics:')
print(f'  Accuracy:  {acc:.4f} = (TP + TN) / Total = ({tp} + {tn}) / {n_samples}')
print(f'  Precision: {prec:.4f} = TP / (TP + FP)')
print(f'  Recall:    {rec:.4f} = TP / (TP + FN)')
print(f'  F1 Score:  {f1:.4f}')

print('\n' + '='*70)
print('CONCLUSION:')
print('='*70)
print(f'✓ Accuracy = {acc:.4f} ≈ 0.3459 (YOUR RESULT)')
print(f'✓ F1 Score = {f1:.4f} (YOUR RESULT)')
print('\nThis confirms the bug analysis:')
print('  - Test accuracy is NON-ZERO because model correctly predicts')
print('    negative class (which makes up ~35% of data)')
print('  - F1 is ZERO because model NEVER predicts positive class (TP=0)')
print('  - This means: Model has collapsed to always predicting "not same group"')
print('='*70)
