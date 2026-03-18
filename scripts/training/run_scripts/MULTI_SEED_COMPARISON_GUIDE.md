# Multi-Seed Comparison Guide

This guide explains how to run fair comparisons between FiLM PE and LoRA models across **multiple datasets and seeds** using **shared train/test splits**.

## Quick Start

### For Any Dataset (Generic)

```bash
# For RIC dataset
bash scripts/training/run_scripts/run_comparison_multi_seed.sh RIC

# For bressin19 dataset
bash scripts/training/run_scripts/run_comparison_multi_seed.sh bressin19

# For hydra_s2 dataset
bash scripts/training/run_scripts/run_comparison_multi_seed.sh hydra_s2
```

### For RIC Dataset Only (Specific Script)

```bash
bash scripts/training/run_scripts/run_RIC_comparison_multi_seed.sh
```

## Available Scripts

### 1. Generic Scripts (Recommended)

#### [generate_shared_splits_any.sh](scripts/data_util/generate_shared_splits_any.sh)
Generate splits for any dataset:
```bash
bash scripts/data_util/generate_shared_splits_any.sh <DATASET_NAME> [SEED]

# Examples:
bash scripts/data_util/generate_shared_splits_any.sh RIC 2023
bash scripts/data_util/generate_shared_splits_any.sh bressin19 12345
bash scripts/data_util/generate_shared_splits_any.sh hydra_s2 42
```

#### [run_comparison_multi_seed.sh](scripts/training/run_scripts/run_comparison_multi_seed.sh)
Run complete multi-seed experiments for any dataset:
```bash
bash scripts/training/run_scripts/run_comparison_multi_seed.sh <DATASET_NAME>

# Examples:
bash scripts/training/run_scripts/run_comparison_multi_seed.sh RIC
bash scripts/training/run_scripts/run_comparison_multi_seed.sh bressin19
```

### 2. Dataset-Specific Scripts (Legacy)

#### RIC Dataset
- [generate_ric_shared_splits.sh](scripts/data_util/generate_ric_shared_splits.sh) - Single seed
- [run_RIC_comparison_multi_seed.sh](scripts/training/run_scripts/run_RIC_comparison_multi_seed.sh) - Multi-seed

## What These Scripts Do

### Job Structure

When you run the multi-seed comparison script, it submits **21 jobs** (for 10 seeds):

1. **Job 1** (CPU, 1h): Generate all splits
2. **Jobs 2-11** (GPU, per seed): FiLM PE experiments
3. **Jobs 12-21** (A100 80GB, per seed): LoRA experiments

### Experiment Breakdown

**Per Dataset:**
- 10 seeds (12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 10234)
- 4 FiLM PE models × 10 seeds = **40 FiLM PE experiments**
- 2 LoRA models × 10 seeds = **20 LoRA experiments**
- **Total: 60 experiments**

### Models Tested

**FiLM PE** (4 models):
- esm2_t33_650M_UR50D
- esm2_t36_3B_UR50D
- esm2_t48_15B_UR50D
- protT5_xl_uniref50

**LoRA** (2 models):
- esm2_t33_650M_UR50D
- protT5_xl_uniref50

## Expected File Structure

After running the split generation:

```
data/splits/
├── RIC_human_fine-tuning_film_pe_seed_12345_esm2_t33_650M_UR50D__ft_train.tsv
├── RIC_human_fine-tuning_film_pe_seed_12345_esm2_t33_650M_UR50D__ft_val.tsv
├── RIC_human_fine-tuning_lora_seed_12345_esm2_t33_650M_UR50D__ft_train.tsv
├── RIC_human_fine-tuning_lora_seed_12345_esm2_t33_650M_UR50D__ft_val.tsv
├── bressin19_human_fine-tuning_film_pe_seed_12345_esm2_t33_650M_UR50D__ft_train.tsv
└── ... (and so on for all seeds and models)
```

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### Check Logs
```bash
# For RIC dataset
tail -f $HOME/RBP_IG/scripts/sbatch_logs/RIC_*

# For bressin19 dataset
tail -f $HOME/RBP_IG/scripts/sbatch_logs/bressin19_*

# For a specific seed
tail -f $HOME/RBP_IG/scripts/sbatch_logs/RIC_FiLM_PE_seed_12345_*
```

### Cancel Jobs
```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <JOB_ID>
```

## Running Multiple Datasets

To run experiments on all datasets:

```bash
# Submit all three datasets
bash scripts/training/run_scripts/run_comparison_multi_seed.sh RIC
bash scripts/training/run_scripts/run_comparison_multi_seed.sh bressin19
bash scripts/training/run_scripts/run_comparison_multi_seed.sh hydra_s2
```

This will submit **63 jobs total** (21 jobs × 3 datasets).

## Advanced Usage

### Custom Seeds

Edit the `SEEDS` array in the script:

```bash
# Default (10 seeds)
SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10234)

# For quick testing (3 seeds)
SEEDS=(12345 23456 34567)

# For more robust analysis (20 seeds)
SEEDS=(12345 23456 34567 ... add more)
```

### Custom Parameters

Modify these variables at the top of the script:

```bash
LORA_EPOCHS=20      # LoRA training epochs
FILM_EPOCHS=30      # FiLM PE training epochs
BS=256              # Batch size
LR_FILM=0.0005      # FiLM PE learning rate
LR_LORA=0.0003      # LoRA learning rate
PE_DIM=512          # Positional encoding dimension
```

### Generate Splits Only

If you just want to generate splits without running training:

```bash
# Generate splits for all 10 seeds
for seed in 12345 23456 34567 45678 56789 67890 78901 89012 90123 10234; do
    bash scripts/data_util/generate_shared_splits_any.sh RIC $seed
done
```

## Paired Statistical Analysis

These experiments use **fixed seeds** across all models, enabling paired comparison:

### Example Analysis in Python

```python
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

# Load results for two models
film_results = [...]  # Performance for 10 seeds
lora_results = [...]  # Performance for same 10 seeds

# Paired t-test
statistic, pvalue = ttest_rel(film_results, lora_results)

# Wilcoxon signed-rank test (non-parametric)
statistic, pvalue = wilcoxon(film_results, lora_results)
```

## Troubleshooting

### "Split files not found"
→ Make sure split generation job completed successfully
→ Check logs: `cat $HOME/RBP_IG/scripts/sbatch_logs/<DATASET>_generate_splits_*.txt`

### "No common genes across all models"
→ Check that all embedding folders exist:
  - `data/embeddings/esm2_t33_650M_UR50D/RIC/`
  - `data/embeddings/esm2_t36_3B_UR50D/RIC/`
  - etc.

### Jobs stuck in queue
→ Check partition availability: `sinfo -p gpu_p`
→ Check your priority: `sprio -u $USER`

### Out of memory errors
→ Increase `--mem` in the job submission
→ Use larger models only with A100 80GB constraint

## File Overview

| Script | Purpose | GPU | Time | Runs |
|--------|---------|-----|------|------|
| `generate_shared_splits_any.sh` | Generate splits for any dataset | No | ~10 min | Once per dataset per seed |
| `run_comparison_multi_seed.sh` | Full multi-seed comparison | Yes | ~24h per job | 21 jobs total |
| `run_FiLM_PE_RIC_shared.sh` | Single FiLM PE run | Yes | ~1-2h | Called by master script |
| `run_LoRA_RIC_shared.sh` | Single LoRA run | A100 | ~2-4h | Called by master script |

## Summary

✅ **Fixed seeds** (12345, ..., 10234) for paired statistical comparison
✅ **Shared splits** - identical train/test genes across all models
✅ **Generic** - works with RIC, bressin19, hydra_s2, or any dataset
✅ **Automated** - one command submits all jobs
✅ **Efficient** - parallel execution per seed
✅ **Comprehensive** - 60 experiments per dataset
