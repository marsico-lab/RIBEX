#!/bin/bash
# Run Peng model comparison across multiple seeds using shared splits
#
# Reuses the same train/test splits as FiLM_PE (symlinked with _peng_ naming)
# so that Peng results are directly comparable to FiLM_PE and LoRA.
#
# USAGE:
#   bash scripts/training/run_scripts/run_Peng_comparison_multi_seed.sh <DATASET_NAME>
#
# Examples:
#   bash scripts/training/run_scripts/run_Peng_comparison_multi_seed.sh RIC
#   bash scripts/training/run_scripts/run_Peng_comparison_multi_seed.sh bressin19

# ============================================================================
# CONFIGURATION
# ============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Dataset name required!"
    echo "Usage: $0 <DATASET_NAME>"
    echo "Examples:"
    echo "  $0 RIC"
    echo "  $0 bressin19"
    exit 1
fi

DATASET_NAME="$1"

# FIXED seeds (same as FiLM_PE and LoRA experiments)
SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 10234)

# Peng models (subset of FiLM_PE models)
PENG_MODELS=(esm2_t33_650M_UR50D protT5_xl_uniref50)

# Training parameters
FT_EPOCHS=50
BS=256
LR=0.005
DEVICES=0
EMB="RIC"

SPLIT_DIR="/path/to/RBP_IG_storage/data/splits"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "================================================================================"
echo "Peng MODEL COMPARISON - MULTI-SEED (${DATASET_NAME})"
echo "================================================================================"
echo ""
echo "Dataset: ${DATASET_NAME}_human_fine-tuning.pkl"
echo "FIXED Seeds: ${SEEDS[@]}"
echo "Models: ${PENG_MODELS[@]}"
echo ""
echo "Training Parameters:"
echo "  Epochs: $FT_EPOCHS"
echo "  Batch Size: $BS"
echo "  Learning Rate: $LR"
echo ""
echo "Total: ${#SEEDS[@]} seeds × ${#PENG_MODELS[@]} models = $((${#SEEDS[@]} * ${#PENG_MODELS[@]})) experiments"
echo ""
echo "================================================================================"

# ============================================================================
# STEP 1: CREATE PENG-NAMED SYMLINKS FROM EXISTING FiLM_PE SPLITS
# ============================================================================

echo ""
echo "Creating Peng-named split symlinks from FiLM_PE splits..."

MISSING=0
CREATED=0
EXISTED=0

for SEED in "${SEEDS[@]}"; do
    for LM_NAME in "${PENG_MODELS[@]}"; do
        for SUFFIX in ft_train ft_val; do
            SRC="${SPLIT_DIR}/${DATASET_NAME}_human_fine-tuning_film_pe_seed_${SEED}_${LM_NAME}__${SUFFIX}.tsv"
            DST="${SPLIT_DIR}/${DATASET_NAME}_human_fine-tuning_peng_seed_${SEED}_${LM_NAME}__${SUFFIX}.tsv"

            if [ ! -f "$SRC" ]; then
                echo "  ERROR: Source not found: $SRC"
                MISSING=$((MISSING + 1))
                continue
            fi

            if [ -f "$DST" ] || [ -L "$DST" ]; then
                EXISTED=$((EXISTED + 1))
            else
                ln -s "$SRC" "$DST"
                CREATED=$((CREATED + 1))
            fi
        done
    done
done

echo "  Created: $CREATED symlinks"
echo "  Already existed: $EXISTED"
if [ $MISSING -gt 0 ]; then
    echo "  MISSING SOURCE FILES: $MISSING"
    echo "  ERROR: Cannot proceed. Run FiLM_PE split generation first."
    exit 1
fi
echo "  ✓ All Peng splits ready"
echo ""

# ============================================================================
# STEP 2: SUBMIT ONE JOB PER SEED
# ============================================================================

echo "Submitting Peng jobs (one per seed)..."

PENG_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    PENG_JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash

#SBATCH -J ${DATASET_NAME}_Peng_seed_${SEED}
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/${DATASET_NAME}_Peng_seed_${SEED}_%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/${DATASET_NAME}_Peng_seed_${SEED}_%j.txt
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
DATASET_NAME="${DATASET_NAME}"
DEVICES=${DEVICES}
BS=${BS}
LR=${LR}
FT_EPOCHS=${FT_EPOCHS}
EMB="${EMB}"
PENG_MODELS=(esm2_t33_650M_UR50D protT5_xl_uniref50)

echo "================================================================================"
echo "Peng EXPERIMENTS - \${DATASET_NAME} - SEED \${SEED}"
echo "================================================================================"
echo ""
echo "Models: \${PENG_MODELS[@]}"
echo ""

COMPLETED=0
FAILED=0

for LM_NAME in "\${PENG_MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Model: \$LM_NAME"
    echo "Progress: \$((COMPLETED + FAILED + 1))/\${#PENG_MODELS[@]}"
    echo "----------------------------------------"

    # Verify splits exist
    SPLIT_TRAIN="${SPLIT_DIR}/\${DATASET_NAME}_human_fine-tuning_peng_seed_\${SEED}_\${LM_NAME}__ft_train.tsv"
    SPLIT_VAL="${SPLIT_DIR}/\${DATASET_NAME}_human_fine-tuning_peng_seed_\${SEED}_\${LM_NAME}__ft_val.tsv"

    if [ ! -f "\$SPLIT_TRAIN" ] || [ ! -f "\$SPLIT_VAL" ]; then
        echo "ERROR: Split files not found!"
        echo "  \$SPLIT_TRAIN"
        echo "  \$SPLIT_VAL"
        FAILED=\$((FAILED + 1))
        continue
    fi

    echo "✓ Split files found"

    # Run Peng training (fine-tuning only, no pre-training)
    python3 scripts/training/train.py \
        -M Peng \
        -D \$DEVICES \
        -DS "\${DATASET_NAME}_human_fine-tuning.pkl" \
        -lm "\$LM_NAME" \
        -ef "\$EMB" \
        -S "\$SEED" \
        -e "\$FT_EPOCHS" \
        -bs "\$BS" \
        -lr "\$LR" \
        --patience 5

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
echo "SEED \${SEED} COMPLETE"
echo "  Completed: \$COMPLETED / \${#PENG_MODELS[@]}"
echo "  Failed: \$FAILED / \${#PENG_MODELS[@]}"
echo "================================================================================"

if [ \$FAILED -gt 0 ]; then
    exit 1
fi

EOF
)

    PENG_JOB_IDS+=($PENG_JOB_ID)
    echo "  Peng Seed $SEED: Job $PENG_JOB_ID"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "ALL Peng JOBS SUBMITTED - ${DATASET_NAME}"
echo "================================================================================"
echo ""
echo "Peng Jobs (${#PENG_JOB_IDS[@]}):"
for i in "${!SEEDS[@]}"; do
    echo "  Seed ${SEEDS[$i]}: ${PENG_JOB_IDS[$i]}"
done
echo ""
echo "Total jobs: ${#PENG_JOB_IDS[@]}"
echo "Total experiments: $((${#SEEDS[@]} * ${#PENG_MODELS[@]}))"
echo ""
echo "Check status: squeue -u \$USER"
echo "Check logs:   ls \$HOME/RBP_IG/scripts/sbatch_logs/${DATASET_NAME}_Peng_*"
echo ""
echo "================================================================================"
