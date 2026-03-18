#!/usr/bin/env bash

# Parameters for Forced Validation Run — IDR-structured proteins (high pLDDT in IDR regions)
: "${SEED:=6666}"       # Unique seed to avoid overwriting standard benchmarks
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}" # bressin19 or RIC
: "${DATASET:=RIC_2_human}" # Using RIC or bressin19 human dataset RIC_2 for hydra dataset with forced validation genes
: "${DEVICES:=0}"
: "${LORA_BS:=2}"
: "${PATIENCE:=10}"
: "${FT_EPOCHS:=10}"
: "${FORCED_GENES:=O75683,P49761,Q9HBL0,Q13573,P15311,P46777,P27824,Q68D10,P26583,Q8N9T8,Q9NQZ2,Q9H8Y5,Q9BRD0,P27797,Q14692,Q96GA3,Q6Y7W6,Q9P015,Q96AG4,Q16540,O00566,Q96CT7,P62750,O95819,Q96A33,O75937,Q9BVP2,P62847,P82979,Q01105,Q9Y266,O00488,Q14247,Q14244,Q9UHB7,Q9NXH9,P17861,P49756,Q05682,O15213,Q9P013,O95391,Q9NVU7,Q8N3X1,Q9Y2S6,O15372}"
: "${EXCLUSIVE_VAL:=True}" # If true, validation set will contain ONLY forced genes

# Map LM_NAME to the short token used in run folders
case "${LM_NAME}" in
  esm1b_t33_650M_UR50S) LM_SHORT="ESM1b_650M" ;;
  esm2_t6_8M_UR50D)    LM_SHORT="ESM2_8M" ;;
  esm2_t12_35M_UR50D)  LM_SHORT="ESM2_35M" ;;
  esm2_t30_150M_UR50D) LM_SHORT="ESM2_150M" ;;
  esm2_t33_650M_UR50D) LM_SHORT="ESM2_650M" ;;
  esm2_t36_3B_UR50D)   LM_SHORT="ESM2_3B" ;;
  esm2_t48_15B_UR50D)  LM_SHORT="ESM2_15B" ;;
  protT5_xl_uniref50)  LM_SHORT="ProtT5_XL" ;;
  *) LM_SHORT="NA" ;;
esac

# Set model-specific hyperparameters
# ProtT5 uses optimized hyperparameters from random search
# ESM models use the original hyperparameters
if [ "${LM_NAME}" = "protT5_xl_uniref50" ]; then
    echo "Using ProtT5-optimized hyperparameters..."
    : "${BS:=392}"
    : "${LR:=0.000335}"
    : "${PE_DIM:=466}" # 466 for protT5
    : "${LORA_R:=10}"
    : "${LORA_ALPHA:=1.58}"
    : "${LORA_DROPOUT:=0.38}"
    : "${LORA_WEIGHT_DECAY:=0.000164}"
else
    echo "Using ESM-optimized hyperparameters..."
    : "${BS:=415}"
    : "${LR:=0.000912}"
    : "${PE_DIM:=418}"
    : "${LORA_R:=6}"
    : "${LORA_ALPHA:=1.66}"
    : "${LORA_DROPOUT:=0.38}"
    : "${LORA_WEIGHT_DECAY:=0.000367}"
fi

LR_FMT=$(printf '%.6f' "${LR}")
PE_SUFFIX=""
if [ "${PE_DIM}" -gt 0 ]; then
    PE_SUFFIX="-PE=${PE_DIM}"
else
    PE_SUFFIX="-PE=No"
fi

sbatch --wait << EOF
#!/bin/bash

#SBATCH -J LoRA_IDRstruct_${SEED}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_IDRstruct_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_IDRstruct_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

# Activate Environment
source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

# Generate splits with FORCED VALIDATION GENES
echo "Generating splits for seed ${SEED} with forced genes: ${FORCED_GENES}..."

# Build the command arguments
SPLIT_ARGS="--dataset ${DATASET}_fine-tuning.pkl --model Lora --seed ${SEED} --lm_name ${LM_NAME} --emb_name ${EMB} --forced_val_genes ${FORCED_GENES}"

# Check if exclusive mode (case-insensitive)
if [ "${EXCLUSIVE_VAL,,}" = "true" ]; then
    echo "  Using EXCLUSIVE mode: validation set will contain ONLY forced genes"
    SPLIT_ARGS="\${SPLIT_ARGS} --forced_val_genes_only"
else
    echo "  Using INCLUSIVE mode: forced genes will be added to validation set"
fi

# Run the split generation
python3 scripts/data_util/generate_splits.py \${SPLIT_ARGS}

# Check if split generation succeeded
if [ \$? -ne 0 ]; then
    echo "ERROR: Split generation failed!"
    exit 1
fi

echo "Starting LoRA Fine-Tuning..."

python3 scripts/training/train.py \
  -M Lora \
  -D ${DEVICES} \
  -DS ${DATASET}_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef "${EMB}" \
  -S "${SEED}" \
  -bs ${BS} \
  --lora_per_device_bs ${LORA_BS} \
  --lora_learning_rate ${LR} \
  --lora_num_train_epochs ${FT_EPOCHS} \
  --pe_dim ${PE_DIM} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lora_weight_decay ${LORA_WEIGHT_DECAY} \
  --patience ${PATIENCE}

EOF
