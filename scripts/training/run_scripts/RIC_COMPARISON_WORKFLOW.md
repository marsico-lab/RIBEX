# RIC Dataset Comparison Workflow

This guide explains how to run fair comparisons between FiLM PE and LoRA models on the RIC dataset using **shared train/test splits**.

## Problem Solved

Previously, different models had different train/test splits because:
- Different models have different embeddings available
- `generate_splits.py` filtered genes based on embedding availability
- Same seed → different filtered datasets → different splits

**Solution**: Pre-generate splits using only genes that have embeddings in ALL models.

## The Solution

### Step 1: Generate Shared Splits (RUN ONCE)

Generate the shared splits for all model configurations:

```bash
# For seed 2023 (default)
bash scripts/data_util/generate_ric_shared_splits.sh

# For a different seed
bash scripts/data_util/generate_ric_shared_splits.sh 2024
```

This creates split files for:
- FiLM PE models: esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D, protT5_xl_uniref50
- LoRA models: esm2_t33_650M_UR50D, protT5_xl_uniref50

**Output files** (in `data/splits/`):
```
RIC_human_fine-tuning_film_pe_seed_2023_esm2_t33_650M_UR50D__ft_train.tsv
RIC_human_fine-tuning_film_pe_seed_2023_esm2_t33_650M_UR50D__ft_val.tsv
RIC_human_fine-tuning_film_pe_seed_2023_esm2_t36_3B_UR50D__ft_train.tsv
RIC_human_fine-tuning_film_pe_seed_2023_esm2_t36_3B_UR50D__ft_val.tsv
... (and so on for all models)
RIC_human_fine-tuning_lora_seed_2023_esm2_t33_650M_UR50D__ft_train.tsv
RIC_human_fine-tuning_lora_seed_2023_esm2_t33_650M_UR50D__ft_val.tsv
... (and so on for LoRA models)
```

### Step 2: Run Training Experiments

Now you can run experiments knowing all models use the **exact same train/test genes**:

#### FiLM PE Training

```bash
# ESM2-650M
SEED=2023 LM_NAME=esm2_t33_650M_UR50D bash scripts/training/run_scripts/run_FiLM_PE_RIC_shared.sh

# ESM2-3B
SEED=2023 LM_NAME=esm2_t36_3B_UR50D bash scripts/training/run_scripts/run_FiLM_PE_RIC_shared.sh

# ESM2-15B
SEED=2023 LM_NAME=esm2_t48_15B_UR50D bash scripts/training/run_scripts/run_FiLM_PE_RIC_shared.sh

# ProtT5-XL
SEED=2023 LM_NAME=protT5_xl_uniref50 bash scripts/training/run_scripts/run_FiLM_PE_RIC_shared.sh
```

#### LoRA Training

```bash
# ESM2-650M
SEED=2023 LM_NAME=esm2_t33_650M_UR50D bash scripts/training/run_scripts/run_LoRA_RIC_shared.sh

# ProtT5-XL
SEED=2023 LM_NAME=protT5_xl_uniref50 bash scripts/training/run_scripts/run_LoRA_RIC_shared.sh
```

### Step 3: Run with Multiple Seeds

For robust comparison, run with multiple seeds:

```bash
# Generate splits for multiple seeds
bash scripts/data_util/generate_ric_shared_splits.sh 2023
bash scripts/data_util/generate_ric_shared_splits.sh 2024
bash scripts/data_util/generate_ric_shared_splits.sh 2025

# Then run experiments for each seed
for SEED in 2023 2024 2025; do
    SEED=$SEED LM_NAME=esm2_t33_650M_UR50D bash scripts/training/run_scripts/run_FiLM_PE_RIC_shared.sh
    SEED=$SEED LM_NAME=esm2_t33_650M_UR50D bash scripts/training/run_scripts/run_LoRA_RIC_shared.sh
done
```

## File Structure

```
RBP_IG/
├── scripts/
│   ├── data_util/
│   │   ├── generate_shared_splits.py          # Core logic for shared splits
│   │   └── generate_ric_shared_splits.sh      # Convenience wrapper (RUN THIS FIRST)
│   └── training/
│       └── run_scripts/
│           ├── run_FiLM_PE_RIC_shared.sh      # FiLM PE training (uses pre-gen splits)
│           ├── run_LoRA_RIC_shared.sh         # LoRA training (uses pre-gen splits)
│           └── RIC_COMPARISON_WORKFLOW.md     # This file
└── data/
    └── splits/
        └── RIC_human_fine-tuning_*__ft_*.tsv  # Pre-generated split files
```

## Key Benefits

✅ **Fair Comparison**: All models trained on identical samples
✅ **Reproducible**: Same seed → same splits across all models
✅ **Efficient**: Generate splits once, reuse for all experiments
✅ **Clear Provenance**: Split files explicitly show which genes were used

## Verification

To verify all models use the same splits:

```bash
# Check that train sets are identical across models for the same seed
cd data/splits/
wc -l RIC_human_fine-tuning_*_seed_2023_esm2_t33_650M_UR50D__ft_train.tsv

# Compare Gene IDs
diff <(cut -f2 RIC_human_fine-tuning_film_pe_seed_2023_esm2_t33_650M_UR50D__ft_train.tsv | sort) \
     <(cut -f2 RIC_human_fine-tuning_lora_seed_2023_esm2_t33_650M_UR50D__ft_train.tsv | sort)

# Should show no differences!
```

## Troubleshooting

### "Split files not found"
→ Run `bash scripts/data_util/generate_ric_shared_splits.sh` first

### "No common genes across all models"
→ Check that embeddings exist for all specified models in `EMBEDDINGS/<lm_name>/RIC/`

### Different number of samples across models
→ This shouldn't happen with shared splits. Check that you're using the correct seed.

## Advanced Usage

### Custom Model Combinations

To generate splits for a different set of models:

```bash
python scripts/data_util/generate_shared_splits.py \
    --dataset "RIC_human_fine-tuning.pkl" \
    --seed 2023 \
    --models "esm2_t33_650M_UR50D:RIC,protT5_xl_uniref50:RIC" \
    --model_names "FiLM_PE,Lora"
```

### Force Specific Genes into Validation

```bash
python scripts/data_util/generate_shared_splits.py \
    --dataset "RIC_human_fine-tuning.pkl" \
    --seed 2023 \
    --models "esm2_t33_650M_UR50D:RIC,protT5_xl_uniref50:RIC" \
    --model_names "FiLM_PE,Lora" \
    --forced_val_genes "O75808,Q9Y2W1" \
    --forced_val_genes_only
```
