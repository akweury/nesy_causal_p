# Group Scorer Models Comparison

This document explains how to use and compare the two group scorer models: **NN-based** and **Transformer-based**.

## Model Architectures

### 1. SimplifiedPositionScorer (NN-based)
- Location: `mbg/scorer/simplified_position_scorer.py`
- Architecture: Multi-layer perceptron with mean pooling for context objects
- Context aggregation: Mean pooling over all context objects
- Faster training and inference

### 2. TransformerPositionScorer (Transformer-based)
- Location: `mbg/scorer/transformer_position_scorer.py`
- Architecture: Transformer encoder with cross-attention mechanism
- Context aggregation: Attention mechanism that dynamically weights context objects
- Better at capturing complex relationships between objects
- Slower but potentially more accurate

## Shared Features

Both models support:
- **Dimension masking**: Selectively disable position, color, or shape features
- **Same interface**: Both use identical input/output signatures
- **9-dimensional input**: (x, y, r, g, b, shape_0, shape_1, shape_2, shape_3)

## Usage Examples

### Run with NN model (default)
```bash
python src/clevr_exp/grm_clevr.py --model_type nn
```

### Run with Transformer model
```bash
python src/clevr_exp/grm_clevr.py --model_type transformer
```

### Compare both models automatically
```bash
python src/clevr_exp/grm_clevr.py --compare_models
```

### Mask specific dimensions
```bash
# Mask position features (x, y)
python src/clevr_exp/grm_clevr.py --model_type transformer --mask_dims position

# Mask color features (r, g, b)
python src/clevr_exp/grm_clevr.py --model_type nn --mask_dims color

# Compare models with masked shape features
python src/clevr_exp/grm_clevr.py --compare_models --mask_dims shape
```

### Adjust training parameters
```bash
python src/clevr_exp/grm_clevr.py \
    --model_type transformer \
    --gd_epochs 200 \
    --gd_lr 0.0005
```

## Dimension Masking

The `mask_dims` parameter allows you to disable specific feature dimensions during training and testing:

- **'position'**: Masks indices 0-1 (x, y coordinates)
- **'color'**: Masks indices 2-4 (r, g, b values)
- **'shape'**: Masks indices 5-8 (one-hot shape encoding)

Examples in code:
```python
# Mask position only
args.mask_dims = ['position']

# Mask multiple dimensions
args.mask_dims = ['position', 'color']
```

## Comparison Output

When using `--compare_models`, the system will:

1. Train both NN and Transformer models on the same data
2. Evaluate both on the test set
3. Print a comparison table:

```
Model Comparison Results
============================================================
Metric                    NN              Transformer     Winner    
-----------------------------------------------------------------
mAP                       0.8234          0.8567          Transformer
precision                 0.8123          0.8445          Transformer
recall                    0.7956          0.8234          Transformer
f1                        0.8038          0.8338          Transformer
binary_accuracy           0.8456          0.8678          Transformer
group_count_accuracy      0.7234          0.7567          Transformer
```

4. Automatically select the better model (based on F1 score) for downstream tasks

## Model Parameters

### NN Model (SimplifiedPositionScorer)
- `position_dim`: 9 (fixed)
- `hidden_dim`: 64
- `context_embed_dim`: 32

### Transformer Model (TransformerPositionScorer)
- `position_dim`: 9 (fixed)
- `hidden_dim`: 64
- `context_embed_dim`: 32
- `num_heads`: 4 (attention heads)
- `num_layers`: 2 (transformer layers)

## Performance Considerations

**NN Model:**
- ✓ Faster training (2-3x)
- ✓ Lower memory usage
- ✓ Good for proximity-based grouping
- ✗ Limited context modeling

**Transformer Model:**
- ✓ Better context understanding
- ✓ Captures complex relationships
- ✓ Better for similarity-based grouping
- ✗ Slower training
- ✗ Higher memory usage

## Troubleshooting

### Out of Memory (Transformer)
Reduce batch size or use fewer transformer layers:
```python
model = TransformerPositionScorer(
    num_heads=2,  # Reduce from 4
    num_layers=1  # Reduce from 2
)
```

### Poor Performance
1. Increase training epochs: `--gd_epochs 200`
2. Adjust learning rate: `--gd_lr 0.0005`
3. Try different dimension masking combinations
4. Check if the right model type matches your task (proximity vs similarity)

## Code Integration

To use the models in your own code:

```python
from mbg.scorer.simplified_position_scorer import SimplifiedPositionScorer
from mbg.scorer.transformer_position_scorer import TransformerPositionScorer

# Initialize models
nn_model = SimplifiedPositionScorer(
    position_dim=9,
    hidden_dim=64,
    context_embed_dim=32,
    mask_dims=None  # or ['position'], ['color'], ['shape']
)

transformer_model = TransformerPositionScorer(
    position_dim=9,
    hidden_dim=64,
    context_embed_dim=32,
    mask_dims=None,
    num_heads=4,
    num_layers=2
)

# Both models have the same forward signature
logits = model(pos_i, pos_j, context_positions)
```
