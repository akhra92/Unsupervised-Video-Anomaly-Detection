# Ablation Study Guide

This document explains how to run systematic ablation studies to measure the contribution of each improvement component.

## Overview

Ablation studies systematically remove or add components to understand their individual and combined contributions to model performance. This helps:

- **Validate improvements**: Verify that each component actually helps
- **Identify key components**: Find which improvements matter most
- **Optimize resource allocation**: Focus on high-impact, low-cost improvements
- **Understand interactions**: See how components work together

---

## Available Ablation Configurations

### 1. Individual Components

These test each improvement in isolation:

| Configuration | Components | Expected Gain |
|---------------|------------|---------------|
| `baseline` | Original model (no improvements) | 0% (reference) |
| `temporal_attention` | + Temporal Attention only | +2-4% |
| `memory` | + Memory Module only | +3-5% |
| `cbam` | + CBAM Attention only | +1-2% |
| `perceptual` | + Perceptual Loss only | +1-2% |
| `augmentation` | + Data Augmentation only | +2-3% |
| `lr_scheduler` | + Cosine LR Scheduler only | +1% |

### 2. Component Groups

These test combinations of related improvements:

| Configuration | Components | Expected Gain |
|---------------|------------|---------------|
| `temporal_memory` | Temporal Attention + Memory | +4-6% |
| `attention_modules` | Temporal Attention + CBAM | +3-5% |
| `loss_improvements` | Perceptual Loss + enhanced multi-loss | +2-3% |
| `training_improvements` | Augmentation + Cosine LR | +2-3% |

### 3. Full Model

| Configuration | Components | Expected Gain |
|---------------|------------|---------------|
| `full_model` | All improvements combined | +8-15% |

---

## Quick Start

### Run All Ablation Studies

```bash
# Run all ablations on ShanghaiTech (50 epochs for quick results)
python ablation_study.py --cfg config/shanghaitech_wresnet.yaml \
                         --ablation all \
                         --epochs 50

# Run all ablations on Ped2
python ablation_study.py --cfg config/ped2_wresnet.yaml \
                         --ablation all \
                         --epochs 50
```

### Run Specific Ablation

```bash
# Test only temporal attention
python ablation_study.py --cfg config/shanghaitech_wresnet.yaml \
                         --ablation temporal_attention \
                         --epochs 50

# Test memory module
python ablation_study.py --ablation memory --epochs 50

# Test full model
python ablation_study.py --ablation full_model --epochs 100
```

---

## Visualizing Results

After running ablation studies, visualize the results:

```bash
# Generate all plots and summary
python visualize_ablation.py --results ablation_results/ablation_summary.json \
                             --output ablation_plots
```

This creates:
- `auc_comparison.png` - Bar chart comparing all configurations
- `component_contributions.png` - Individual component contributions
- `cumulative_effect.png` - Progressive improvement curve
- `ablation_summary.txt` - Detailed text summary

---

## Understanding Results

### Output Files

After running ablation studies, you'll find:

```
ablation_results/
├── ablation_summary.json              # Overall summary
├── baseline_results.json              # Baseline experiment results
├── temporal_attention_results.json    # Temporal attention results
├── memory_results.json                # Memory module results
├── ...
└── Baseline_(Original_Model)/         # Model checkpoints
    ├── epoch_10.pth
    ├── epoch_20.pth
    └── final_model.pth
```

### Summary JSON Format

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "dataset": "shanghaitech",
  "epochs": 50,
  "results": [
    {
      "ablation_name": "Baseline (Original Model)",
      "description": "Original model without any improvements",
      "auc": 72.5,
      "training_time": 3600.0,
      "model_path": "ablation_results/Baseline_(Original_Model)/final_model.pth"
    },
    {
      "ablation_name": "Baseline + Temporal Attention",
      "auc": 75.8,
      "training_time": 3750.0,
      ...
    }
  ]
}
```

---

## Expected Results

### UCSD Ped2 Dataset

| Configuration | Expected AUC | Improvement | Training Time |
|---------------|--------------|-------------|---------------|
| Baseline | ~95.0% | - | 2h |
| + Temporal Attention | ~96.5% | +1.5% | 2h 10m |
| + Memory Module | ~97.0% | +2.0% | 2h 20m |
| + CBAM | ~95.8% | +0.8% | 2h 5m |
| + Perceptual Loss | ~95.7% | +0.7% | 2h 15m |
| + Augmentation | ~96.2% | +1.2% | 2h 30m |
| Full Model | ~97.5-98.0% | +2.5-3.0% | 2h 40m |

### ShanghaiTech Dataset

| Configuration | Expected AUC | Improvement | Training Time |
|---------------|--------------|-------------|---------------|
| Baseline | ~72.0% | - | 6h |
| + Temporal Attention | ~75.5% | +3.5% | 6h 20m |
| + Memory Module | ~76.8% | +4.8% | 6h 40m |
| + CBAM | ~73.2% | +1.2% | 6h 10m |
| + Perceptual Loss | ~73.8% | +1.8% | 6h 30m |
| + Augmentation | ~74.5% | +2.5% | 7h |
| Full Model | ~80.0-82.0% | +8.0-10.0% | 7h 20m |

---

## Interpreting Plots

### 1. AUC Comparison Chart

**What it shows**: Direct comparison of AUC scores across all configurations

**How to read**:
- Longer bars = better performance
- Red bar = baseline reference
- Blue bar = full model
- Gray bars = individual/partial improvements

**What to look for**:
- Which components give biggest gains?
- Are gains consistent or variable?
- Does full model achieve expected performance?

### 2. Component Contributions

**What it shows**: Individual contribution of each component

**How to read**:
- Bar height = AUC improvement over baseline
- Scatter plot shows performance vs training cost

**What to look for**:
- Best performance/cost tradeoff
- Diminishing returns on expensive components
- Surprisingly effective cheap improvements

### 3. Cumulative Effect

**What it shows**: Progressive improvement as components are added

**How to read**:
- Line shows AUC as components accumulate
- Shaded area represents total improvement
- Annotations show incremental gains

**What to look for**:
- Is improvement steady or does it plateau?
- Do later additions still help?
- Are components complementary?

---

## Advanced Usage

### Custom Ablation Configuration

Edit `ablation_study.py` to add custom configurations:

```python
ABLATION_CONFIGS['my_custom'] = {
    'name': 'My Custom Configuration',
    'use_memory': True,
    'use_temporal_attn': False,
    'use_perceptual': True,
    'use_cbam': False,
    'use_augmentation': True,
    'use_cosine_lr': False,
    'description': 'Memory + Perceptual + Augmentation'
}
```

Then run:
```bash
python ablation_study.py --ablation my_custom --epochs 50
```

### Comparing Datasets

Run ablation studies on multiple datasets and compare:

```bash
# Run on all datasets
python ablation_study.py --cfg config/ped2_wresnet.yaml \
                         --ablation all \
                         --output-dir ablation_results_ped2

python ablation_study.py --cfg config/shanghaitech_wresnet.yaml \
                         --ablation all \
                         --output-dir ablation_results_shanghai

python ablation_study.py --cfg config/avenue_wresnet.yaml \
                         --ablation all \
                         --output-dir ablation_results_avenue

# Compare results
python compare_datasets.py ablation_results_*/ablation_summary.json
```

### Statistical Significance

Run multiple trials to test statistical significance:

```bash
# Run 3 trials with different random seeds
for seed in 42 123 456; do
    python ablation_study.py --ablation all \
                             --output-dir ablation_results_seed${seed} \
                             --seed $seed
done

# Compute mean and std
python compute_statistics.py ablation_results_seed*/ablation_summary.json
```

---

## Best Practices

### 1. Training Duration

- **Quick validation** (50 epochs): Good for rapid iteration and debugging
- **Standard evaluation** (100 epochs): Recommended for paper results
- **Full training** (200+ epochs): For final performance numbers

### 2. Multiple Runs

- Run each ablation 3 times with different seeds
- Report mean ± standard deviation
- Use t-tests to verify significance

### 3. Resource Management

- Run expensive ablations first (full model, memory module)
- Use smaller epochs for initial validation
- Scale up once you identify promising combinations

### 4. Documentation

- Save all configs and random seeds
- Keep detailed logs of changes
- Document unexpected results

---

## Troubleshooting

### Out of Memory

**Problem**: GPU runs out of memory during ablation

**Solutions**:
- Reduce batch size in config
- Disable memory module for quick tests
- Use gradient checkpointing

### Inconsistent Results

**Problem**: AUC varies significantly between runs

**Solutions**:
- Fix random seeds
- Run multiple trials and average
- Check for data leakage

### Slow Training

**Problem**: Ablation studies take too long

**Solutions**:
- Reduce epochs to 50 for validation
- Use smaller subset of training data
- Run ablations in parallel on multiple GPUs

---

## FAQ

**Q: How many epochs should I use?**
A: 50 epochs for quick validation, 100 for paper results, 200+ for final performance.

**Q: Should I run all ablations or just key ones?**
A: Start with key ones (temporal_attention, memory, full_model), then add others if time permits.

**Q: How do I know if a component really helps?**
A: Run 3-5 trials and check if improvement is consistent and statistically significant.

**Q: Can I combine specific components?**
A: Yes! Edit `ABLATION_CONFIGS` to create custom combinations.

**Q: What if full model doesn't achieve expected gain?**
A: Check for bugs, verify all components are enabled, ensure sufficient training epochs.

---

## Contributing

To add new ablation configurations:

1. Edit `ABLATION_CONFIGS` in `ablation_study.py`
2. Add visualization logic to `visualize_ablation.py`
3. Update this documentation
4. Run tests to verify functionality

---

## Citation

If you use these ablation studies in your research, please cite:

```bibtex
@article{rakhmonov2024aonet,
  title={AONet: Attention network with optional activation for unsupervised video anomaly detection},
  author={Rakhmonov, Akhrorjon Akhmadjon Ugli and Subramanian, Barathi and Amirian Varnousefaderani, Bahar and Kim, Jeonghong},
  journal={ETRI Journal},
  year={2024},
  note={With systematic ablation studies validating component contributions}
}
```
