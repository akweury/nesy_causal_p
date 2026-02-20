# Summary: Bug Analysis and Fix for grp_comparison.py

## Problem Statement

Your evaluation showed confusing metrics:
```
Model                     Train Acc    Test Acc     Precision    Recall       F1 Score     
SimplifiedNN              0.7510       0.3459       0.0000       0.0000       0.0000       
TransformerScorer         0.7510       0.3459       0.0000       0.0000       0.0000      
```

**Question**: Why is test accuracy non-zero (0.3459) but F1 score is zero?

## Answer: This is NOT a Bug in the Code

The metrics are actually **correct** - this is the expected behavior when:

### The Model is Predicting Everything as Negative (Class 0)

Your models are outputting probabilities < 0.5 for ALL pairs, meaning they predict every pair as "different groups" (negative class).

#### Why This Causes These Specific Metrics:

1. **Test Accuracy = 0.3459 (34.59%)**
   - Accuracy = (TP + TN) / Total
   - When model predicts all negative: TP = 0, FP = 0
   - Accuracy = TN / Total = (correct negatives) / (all predictions)
   - **0.3459 means ~34.59% of test pairs are truly different groups**
   - The model gets these right by default!

2. **Precision = Recall = F1 = 0.0**
   - Precision = TP / (TP + FP) = 0 / 0 = 0 (with zero_division=0)
   - Recall = TP / (TP + FN) = 0 / (0 + all_positives) = 0
   - F1 = 2 × (0 × 0) / (0 + 0) = 0

#### Verified with Simulation:

```python
# 35% truly negative, 65% truly positive
# Model predicts 100% negative

Confusion Matrix:
  TP = 0    FP = 0
  FN = 650  TN = 350

Accuracy  = (0 + 350) / 1000 = 0.35 ✓
Precision = 0 / 0 = 0 ✓
Recall    = 0 / 650 = 0 ✓
F1        = 0 ✓
```

## Root Cause: Model Training Failure

Your models have **not actually learned the task**. Despite 75% training accuracy, they're likely just predicting the majority class during training too.

### Possible Reasons:

1. **Class Imbalance During Training**
   - If training data has many more negative pairs than positive pairs
   - Model learns "always predict negative" as an easy strategy
   - Gets high accuracy without learning the real pattern

2. **Weak Features**
   - The masked features (contour) may not be sufficient
   - Model can't distinguish same-group from different-group pairs

3. **Insufficient Training**
   - 50 epochs may not be enough
   - Learning rate might be too high/low

4. **Architecture Issues**
   - Model capacity too small
   - Not using context effectively

## Changes Made to Code

### 1. Added Detailed Diagnostics

The updated code now shows:

```python
# Confusion matrix with explicit counts
print(f"  Confusion Matrix:")
print(f"    True Positives (TP):  {tp}")
print(f"    False Positives (FP): {fp}")
print(f"    True Negatives (TN):  {tn}")
print(f"    False Negatives (FN): {fn}")

# Distribution analysis
print(f"  Distribution Analysis:")
print(f"    True Labels:  Positive={num_true_positive}, Negative={num_true_negative}")
print(f"    Predictions:  Positive={num_pred_positive}, Negative={num_pred_negative}")

# Probability statistics
print(f"    Avg Probability: {all_probs_np.mean():.4f}")
print(f"    Prob Range: [{all_probs_np.min():.4f}, {all_probs_np.max():.4f}]")

# Warning when issue detected
if num_pred_positive == 0:
    print(f"  ⚠️  CRITICAL WARNING: Model is predicting ALL pairs as negative!")
```

### 2. Added Summary Section

After all models are evaluated, the code now prints:

```python
print("DETAILED DIAGNOSIS - CONFUSION MATRIX & PREDICTIONS")
for model_name, metrics in all_results.items():
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Predictions: Positive={pos}, Negative={neg}")
    print(f"  Probability Stats: Mean={mean}, Range=[min, max]")
    print(f"  Best Threshold: {thresh} (Acc={acc})")
```

### 3. Fixed Variable Ordering Bug

There was a minor bug where variables were referenced before being calculated. This has been fixed by reordering the code to:
1. Calculate all metrics and statistics
2. Store them in the metrics dictionary
3. Print them

## Recommendations to Fix the Model

1. **Check Class Balance in Training Data**
   ```python
   # In PairwiseGroupDataset.__init__
   pos_samples = sum(1 for _, _, _, label in self.samples if label == 1.0)
   print(f"Training data: {pos_samples} positive, {len(self.samples)-pos_samples} negative")
   ```

2. **Add Class Weighting to Loss Function**
   ```python
   # In train_scorer_model
   pos_weight = torch.tensor([neg_count / pos_count]).to(device)
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

3. **Use Focal Loss for Imbalanced Data**
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma
       
       def forward(self, logits, targets):
           bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
           pt = torch.exp(-bce)
           focal_loss = self.alpha * (1-pt)**self.gamma * bce
           return focal_loss.mean()
   ```

4. **Verify Features Are Informative**
   - Try training WITHOUT masking contour to see if it improves
   - Visualize embeddings to check if same-group pairs cluster together

5. **Increase Training**
   - More epochs (100-200)
   - Lower learning rate (1e-4)
   - Add learning rate scheduler

6. **Use Optimal Threshold**
   - The diagnostic code finds the best threshold
   - Use that instead of 0.5 for final predictions

## Next Steps

1. **Run the updated code** - you'll see full diagnostics
2. **Check the probability distribution** - are all probs really < 0.5?
3. **Inspect training data balance** - is it 50/50 or skewed?
4. **Try the recommended fixes** above
5. **Monitor training more carefully** - plot loss curves, check gradients

## Files Modified

- `src/elvis_exp/grp_comparison.py` - Added comprehensive diagnostics

## Files Created

- `BUG_ANALYSIS.md` - Detailed explanation of the bug
- `test_bug_simulation.py` - Test confirming the analysis
- `SUMMARY.md` - This file
