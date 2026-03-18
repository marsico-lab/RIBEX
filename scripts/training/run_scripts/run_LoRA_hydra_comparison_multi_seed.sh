#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}"
: "${DATASET:=hydra_s2_comparison}" #hydra_s2_comparison hydra_comparison
: "${DEVICES:=0}"
: "${LORA_BS:=2}"
: "${PATIENCE:=10}"
: "${FT_EPOCHS:=15}"

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
    : "${PE_DIM:=466}"
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

# Multi-seed loop
#SEEDS=(2023 2842 5923 8347 1290 4751 7634 9821 1234 4567 8901 1357 2468 3579 1347 1618 1914 1889 3201)
# use different seeds from
SEEDS=(1618 2947 9258 5113 4321 6789 2413 7304 3963 4856 2704 4737) # 3957 9584 3948 2937)
for SEED in "${SEEDS[@]}"; do
    echo "Submitting job for SEED: ${SEED}"

    sbatch << EOF
#!/bin/bash

#SBATCH -J LoRA_${DATASET}_${SEED}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${DATASET}_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${DATASET}_${LM_SHORT}_${SEED}_%j.txt
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

# Fine-tuning ONLY logic

# Generate splits for this specific seed
echo "Generating splits for seed ${SEED}..."
python3 scripts/data_util/generate_splits.py \
    --dataset "${DATASET}_fine-tuning.pkl" \
    --model "Lora" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}"

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

    sleep 1
done
