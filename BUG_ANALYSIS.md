# Bug Analysis: Why Test Accuracy is Non-Zero but F1 Score is Zero

## The Problem

Your evaluation results show:
- **Test Accuracy: 0.3459 (34.59%)**
- **Precision: 0.0000**
- **Recall: 0.0000**
- **F1 Score: 0.0000**

This appears contradictory - how can accuracy be non-zero while F1 is zero?

## Root Cause

**The model is predicting ALL pairs as negative (different groups)**, meaning all output probabilities are below the 0.5 threshold.

### Why Metrics Behave This Way:

1. **Accuracy** = (TP + TN) / Total
   - Accuracy counts BOTH correct positives AND correct negatives
   - If 34.59% of pairs are truly negative (different groups), and the model predicts everything as negative
   - Then accuracy = TN / Total ≈ 0.3459
   - This is essentially the **negative class proportion** in your test set

2. **Precision** = TP / (TP + FP)
   - If model never predicts positive → TP = 0
   - Precision = 0 / X = 0

3. **Recall** = TP / (TP + FN)
   - If model never predicts positive → TP = 0
   - Recall = 0 / Y = 0

4. **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - F1 = 2 × (0 × 0) / (0 + 0) = 0

## What This Means

The model has **completely failed to learn** how to identify pairs that belong to the same group. It's outputting all probabilities < 0.5, resulting in:

```
Confusion Matrix:
  TP (True Positives):   0     ← Never correctly identifies same-group pairs
  FP (False Positives):  0     ← Never predicts positive class
  TN (True Negatives):   XXX   ← Correctly predicts different-group pairs by default
  FN (False Negatives):  YYY   ← Misses ALL same-group pairs
```

## Possible Causes

1. **Training Issue**: Model hasn't actually learned the task
   - Check if training loss is decreasing
   - Training accuracy might be misleading if also just predicting negatives

2. **Class Imbalance**: If training data is heavily skewed toward negative pairs
   - Model learns to always predict negative as a safe strategy
   - Solution: Balance training data or use class weights

3. **Threshold Too High**: All probabilities are in range (0, 0.5)
   - Check probability distribution statistics
   - May need to adjust threshold or calibrate model

4. **Train/Test Mismatch**: Distribution shift between training and test data
   - Test data might have different characteristics
   - Model doesn't generalize

5. **Model Collapse**: Outputs are saturated or collapsed
   - Check if all predictions are very similar
   - May indicate gradient issues or architecture problems

## Diagnosis Added

The updated code now prints:
- **Confusion matrix** (TP, FP, TN, FN)
- **Prediction distribution** (how many positive/negative predictions)
- **Probability statistics** (mean, std, range)
- **Best threshold** found by grid search
- **Warning flags** when issues are detected

Run the script again to see these diagnostics and identify the specific cause.

## Next Steps

1. **Run the updated script** to see detailed diagnostics
2. **Check training logs** - is training accuracy also ~34%?
3. **Inspect probability distributions** - are all outputs < 0.5?
4. **Verify data balance** - what % of pairs are same-group in train vs test?
5. **Consider solutions**:
   - Increase training epochs
   - Add class weighting: `BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]))`
   - Use focal loss for class imbalance
   - Lower threshold or use optimal threshold from diagnostics
   - Check if features are actually informative for the task
