#!/bin/bash
# Run LoRA ESM2 650M with PE dim=2 (FiLM disabled) across all 10 comparison seeds.
#
# Reuses the train/val splits already generated for the LoRA ESM2 650M RIC runs.
# No split-generation job is needed.
#
# IMPORTANT: Uses the same FIXED seeds as run_RIC_comparison_multi_seed.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10234)

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "================================================================================"
echo "LoRA ESM2 650M  |  PE_DIM=2 (FiLM disabled)  |  MULTI-SEED"
echo "================================================================================"
echo ""
echo "FIXED Seeds: ${SEEDS[@]}"
echo "Number of seeds: ${#SEEDS[@]}"
echo ""
echo "This will submit:"
echo "  - ${#SEEDS[@]} LoRA jobs (one per seed, A100 80GB)"
echo ""
echo "================================================================================"

# ============================================================================
# VERIFY SPLITS EXIST FOR ALL SEEDS BEFORE SUBMITTING
# ============================================================================

echo ""
echo "Verifying split files exist for all seeds..."
ALL_SPLITS_OK=1
SPLIT_BASE="/path/to/RBP_IG_storage/data/splits"

for SEED in "${SEEDS[@]}"; do
    SPLIT_TRAIN="${SPLIT_BASE}/RIC_human_fine-tuning_lora_seed_${SEED}_esm2_t33_650M_UR50D__ft_train.tsv"
    SPLIT_VAL="${SPLIT_BASE}/RIC_human_fine-tuning_lora_seed_${SEED}_esm2_t33_650M_UR50D__ft_val.tsv"
    if [ ! -f "$SPLIT_TRAIN" ] || [ ! -f "$SPLIT_VAL" ]; then
        echo "  MISSING splits for seed ${SEED}:"
        [ ! -f "$SPLIT_TRAIN" ] && echo "    $SPLIT_TRAIN"
        [ ! -f "$SPLIT_VAL"   ] && echo "    $SPLIT_VAL"
        ALL_SPLITS_OK=0
    else
        echo "  OK seed ${SEED}"
    fi
done

if [ "$ALL_SPLITS_OK" -eq 0 ]; then
    echo ""
    echo "ERROR: Some split files are missing. Aborting."
    exit 1
fi

echo "All splits found."
echo ""

# ============================================================================
# SUBMIT ONE JOB PER SEED
# ============================================================================

echo "Submitting LoRA PE2 jobs (one per seed)..."

LORA_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    LORA_JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash

#SBATCH -J RIC_LoRA_PE2_seed_${SEED}
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_LoRA_PE2_seed_${SEED}_%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_LoRA_PE2_seed_${SEED}_%j.txt
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
DEVICE_ID=0
BS=256
LORA_BS=2
LR=0.0003
LORA_EPOCHS=20
PE_DIM=2
EMB="RIC"
LM_NAME="esm2_t33_650M_UR50D"
SPLIT_BASE="/path/to/RBP_IG_storage/data/splits"

echo "================================================================================"
echo "LoRA PE2 EXPERIMENT - SEED \${SEED}"
echo "================================================================================"
echo ""
echo "Model:   \${LM_NAME}"
echo "PE dim:  \${PE_DIM}  (FiLM disabled)"
echo ""

# Verify splits
SPLIT_TRAIN="\${SPLIT_BASE}/RIC_human_fine-tuning_lora_seed_\${SEED}_\${LM_NAME}__ft_train.tsv"
SPLIT_VAL="\${SPLIT_BASE}/RIC_human_fine-tuning_lora_seed_\${SEED}_\${LM_NAME}__ft_val.tsv"

if [ ! -f "\$SPLIT_TRAIN" ] || [ ! -f "\$SPLIT_VAL" ]; then
    echo "ERROR: Split files not found!"
    echo "  \$SPLIT_TRAIN"
    echo "  \$SPLIT_VAL"
    exit 1
fi

echo "Split train: \$SPLIT_TRAIN"
echo "Split val:   \$SPLIT_VAL"
echo ""

if ! [[ "\$DEVICE_ID" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: DEVICE_ID must be an integer, got: '\$DEVICE_ID'"
    exit 1
fi

TRAIN_CMD=(
    python3 scripts/training/train.py
    -M Lora
    -D "\$DEVICE_ID"
    -DS RIC_human_fine-tuning.pkl
    -lm "\$LM_NAME"
    -ef "\$EMB"
    -S "\$SEED"
    -bs "\$BS"
    --lora_per_device_bs "\$LORA_BS"
    --lora_learning_rate "\$LR"
    --lora_num_train_epochs "\$LORA_EPOCHS"
    --pe_dim "\$PE_DIM"
    --patience 10
)

echo "Running command:"
printf '  %q' "\${TRAIN_CMD[@]}"
echo
"\${TRAIN_CMD[@]}"

EXIT_CODE=\$?

if [ \$EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "SUCCESS: SEED \${SEED}"
    echo "================================================================================"
else
    echo "================================================================================"
    echo "FAILED: SEED \${SEED}  (exit code \${EXIT_CODE})"
    echo "================================================================================"
    exit 1
fi

EOF
)

    LORA_JOB_IDS+=($LORA_JOB_ID)
    echo "  Seed ${SEED}: Job ${LORA_JOB_ID}"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "LoRA PE2 Jobs (${#LORA_JOB_IDS[@]}):"
for i in "${!SEEDS[@]}"; do
    echo "  Seed ${SEEDS[$i]}: ${LORA_JOB_IDS[$i]}"
done
echo ""
echo "Total jobs submitted: ${#LORA_JOB_IDS[@]}"
echo ""
echo "Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  \$HOME/RBP_IG/scripts/sbatch_logs/"
echo ""
echo "================================================================================"
