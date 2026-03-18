#!/bin/bash
# Master script to run complete RIC dataset comparison across multiple seeds
#
# This submits:
# 1. One job to generate all splits (no GPU)
# 2. One job PER SEED for FiLM PE (all FiLM PE models for that seed, regular GPU)
# 3. One job PER SEED for LoRA (all LoRA models for that seed, A100 80GB)
#
# IMPORTANT: Uses FIXED seeds for paired comparison across models

# ============================================================================
# CONFIGURATION
# ============================================================================

# FIXED seeds for paired comparison (10 seeds, 5 digits each)
SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10234)

# Training parameters
LORA_EPOCHS=20
FILM_EPOCHS=30
BS=256
LR_FILM=0.0005
LR_LORA=0.0003
PE_DIM=512
DEVICES=0
LORA_BS=2

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "================================================================================"
echo "RIC DATASET COMPARISON - MULTI-SEED PAIRED EXPERIMENT"
echo "================================================================================"
echo ""
echo "FIXED Seeds: ${SEEDS[@]}"
echo "Number of seeds: ${#SEEDS[@]}"
echo ""
echo "This will submit:"
echo "  - 1 job to generate splits (no GPU)"
echo "  - ${#SEEDS[@]} jobs for FiLM PE (one per seed, all FiLM PE models, regular GPU)"
echo "  - ${#SEEDS[@]} jobs for LoRA (one per seed, all LoRA models, A100 80GB)"
echo ""
echo "Total: $((1 + ${#SEEDS[@]} * 2)) jobs"
echo ""
echo "================================================================================"

# ============================================================================
# JOB 1: GENERATE ALL SPLITS
# ============================================================================

echo ""
echo "Submitting JOB 1: Generate all splits..."

SPLIT_JOB_ID=$(sbatch --parsable << 'EOF'
#!/bin/bash

#SBATCH -J RIC_generate_splits
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_generate_splits_%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_generate_splits_%j.txt
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10234)

echo "================================================================================"
echo "GENERATING SHARED SPLITS FOR ALL SEEDS"
echo "================================================================================"
echo ""
echo "Seeds: ${SEEDS[@]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Generating splits for seed: $SEED"
    echo "----------------------------------------"

    bash scripts/data_util/generate_ric_shared_splits.sh $SEED

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate splits for seed $SEED"
        exit 1
    fi

    echo ""
done

echo "✓ All shared splits generated successfully!"
echo ""
echo "================================================================================"

EOF
)

echo "  Job ID: $SPLIT_JOB_ID"

# ============================================================================
# JOBS 2-11: ONE FiLM PE JOB PER SEED
# ============================================================================

echo ""
echo "Submitting FiLM PE jobs (one per seed)..."

FILM_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    FILM_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID << EOF
#!/bin/bash

#SBATCH -J RIC_FiLM_seed_${SEED}
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_FiLM_PE_seed_${SEED}_%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_FiLM_PE_seed_${SEED}_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

SEED=${SEED}
DEVICES=0
BS=256
LR=0.0005
FILM_EPOCHS=30
PE_DIM=512
EMB="RIC"
FILM_PE_MODELS=(esm2_t33_650M_UR50D esm2_t36_3B_UR50D esm2_t48_15B_UR50D protT5_xl_uniref50)

echo "================================================================================"
echo "FiLM PE EXPERIMENTS - SEED ${SEED}"
echo "================================================================================"
echo ""
echo "Models: \${FILM_PE_MODELS[@]}"
echo ""

COMPLETED=0
FAILED=0

for LM_NAME in "\${FILM_PE_MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Model: \$LM_NAME"
    echo "Progress: \$((COMPLETED + FAILED + 1))/\${#FILM_PE_MODELS[@]}"
    echo "----------------------------------------"

    # Verify splits exist
    SPLIT_TRAIN="/path/to/RBP_IG_storage/data/splits/RIC_human_fine-tuning_film_pe_seed_\${SEED}_\${LM_NAME}__ft_train.tsv"
    SPLIT_VAL="/path/to/RBP_IG_storage/data/splits/RIC_human_fine-tuning_film_pe_seed_\${SEED}_\${LM_NAME}__ft_val.tsv"

    if [ ! -f "\$SPLIT_TRAIN" ] || [ ! -f "\$SPLIT_VAL" ]; then
        echo "ERROR: Split files not found!"
        FAILED=\$((FAILED + 1))
        continue
    fi

    echo "✓ Split files found"

    # Run FiLM PE training
    python3 scripts/training/train.py \\
        -M FiLM_PE \\
        -D \$DEVICES \\
        -DS RIC_human_fine-tuning.pkl \\
        -lm \$LM_NAME \\
        -ef \$EMB \\
        -S \$SEED \\
        -e \$FILM_EPOCHS \\
        -bs \$BS \\
        -lr \$LR \\
        --pe_dim \$PE_DIM \\
        --patience 5 \\
        --keep_small_val_batches

    if [ \$? -eq 0 ]; then
        echo "✓ SUCCESS: \$LM_NAME"
        COMPLETED=\$((COMPLETED + 1))
    else
        echo "✗ FAILED: \$LM_NAME"
        FAILED=\$((FAILED + 1))
    fi

    echo ""
done

echo "================================================================================"
echo "SEED ${SEED} COMPLETE"
echo "  Completed: \$COMPLETED / \${#FILM_PE_MODELS[@]}"
echo "  Failed: \$FAILED / \${#FILM_PE_MODELS[@]}"
echo "================================================================================"

if [ \$FAILED -gt 0 ]; then
    exit 1
fi

EOF
)

    FILM_JOB_IDS+=($FILM_JOB_ID)
    echo "  FiLM PE Seed $SEED: Job $FILM_JOB_ID"
done

# ============================================================================
# JOBS 12-21: ONE LoRA JOB PER SEED
# ============================================================================

echo ""
echo "Submitting LoRA jobs (one per seed)..."

LORA_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    LORA_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID << EOF
#!/bin/bash

#SBATCH -J RIC_LoRA_seed_${SEED}
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_LoRA_seed_${SEED}_%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_LoRA_seed_${SEED}_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

SEED=${SEED}
DEVICES=0
BS=256
LORA_BS=2
LR=0.0003
LORA_EPOCHS=20
PE_DIM=512
EMB="RIC"
LORA_MODELS=(esm2_t33_650M_UR50D protT5_xl_uniref50)

echo "================================================================================"
echo "LoRA EXPERIMENTS - SEED ${SEED}"
echo "================================================================================"
echo ""
echo "Models: \${LORA_MODELS[@]}"
echo ""

COMPLETED=0
FAILED=0

for LM_NAME in "\${LORA_MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Model: \$LM_NAME"
    echo "Progress: \$((COMPLETED + FAILED + 1))/\${#LORA_MODELS[@]}"
    echo "----------------------------------------"

    # Verify splits exist
    SPLIT_TRAIN="/path/to/RBP_IG_storage/data/splits/RIC_human_fine-tuning_lora_seed_\${SEED}_\${LM_NAME}__ft_train.tsv"
    SPLIT_VAL="/path/to/RBP_IG_storage/data/splits/RIC_human_fine-tuning_lora_seed_\${SEED}_\${LM_NAME}__ft_val.tsv"

    if [ ! -f "\$SPLIT_TRAIN" ] || [ ! -f "\$SPLIT_VAL" ]; then
        echo "ERROR: Split files not found!"
        FAILED=\$((FAILED + 1))
        continue
    fi

    echo "✓ Split files found"

    # Run LoRA training
    python3 scripts/training/train.py \\
        -M Lora \\
        -D \$DEVICES \\
        -DS RIC_human_fine-tuning.pkl \\
        -lm \$LM_NAME \\
        -ef \$EMB \\
        -S \$SEED \\
        -bs \$BS \\
        --lora_per_device_bs \$LORA_BS \\
        --lora_learning_rate \$LR \\
        --lora_num_train_epochs \$LORA_EPOCHS \\
        --pe_dim \$PE_DIM \\
        --patience 10

    if [ \$? -eq 0 ]; then
        echo "✓ SUCCESS: \$LM_NAME"
        COMPLETED=\$((COMPLETED + 1))
    else
        echo "✗ FAILED: \$LM_NAME"
        FAILED=\$((FAILED + 1))
    fi

    echo ""
done

echo "================================================================================"
echo "SEED ${SEED} COMPLETE"
echo "  Completed: \$COMPLETED / \${#LORA_MODELS[@]}"
echo "  Failed: \$FAILED / \${#LORA_MODELS[@]}"
echo "================================================================================"

if [ \$FAILED -gt 0 ]; then
    exit 1
fi

EOF
)

    LORA_JOB_IDS+=($LORA_JOB_ID)
    echo "  LoRA Seed $SEED: Job $LORA_JOB_ID"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Split Generation Job: $SPLIT_JOB_ID"
echo ""
echo "FiLM PE Jobs (${#FILM_JOB_IDS[@]}):"
for i in "${!SEEDS[@]}"; do
    echo "  Seed ${SEEDS[$i]}: ${FILM_JOB_IDS[$i]}"
done
echo ""
echo "LoRA Jobs (${#LORA_JOB_IDS[@]}):"
for i in "${!SEEDS[@]}"; do
    echo "  Seed ${SEEDS[$i]}: ${LORA_JOB_IDS[$i]}"
done
echo ""
echo "Total jobs submitted: $((1 + ${#FILM_JOB_IDS[@]} + ${#LORA_JOB_IDS[@]}))"
echo ""
echo "Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  \$HOME/RBP_IG/scripts/sbatch_logs/"
echo ""
echo "Total experiments:"
echo "  Seeds: ${#SEEDS[@]}"
echo "  FiLM PE: ${#SEEDS[@]} × 4 models = 40 experiments"
echo "  LoRA:    ${#SEEDS[@]} × 2 models = 20 experiments"
echo "  TOTAL:   60 experiments across $((1 + ${#SEEDS[@]} * 2)) jobs"
echo ""
echo "================================================================================"
