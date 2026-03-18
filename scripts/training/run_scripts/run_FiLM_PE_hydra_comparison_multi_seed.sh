#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}"
: "${DATASET:=hydra_s2_comparison}" # hydra_s2_comparison hydra_comparison
: "${FT_EPOCHS:=40}"
: "${BS:=256}" #512}"
: "${LR:=0.000093}" #0.000093}"
: "${DEVICES:=0}"
: "${PE_DIM:=512}"

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

LR_FMT=$(printf '%.6f' "${LR}")
PE_DIM_PRE_TRAINING=2

if [ "${PE_DIM_PRE_TRAINING}" -gt 2 ]; then
    PE_SUFFIX="-PE=${PE_DIM_PRE_TRAINING}"
else
    PE_SUFFIX="-PE=No"
fi

# Converting to sbatch structure for multiple seeds...
#SEEDS=(2023 2842 5923 8347 1290 4751 7634 9821 1234 4567 8901 1357 2468 3579 1347 1618 1914 1889 3201)
SEEDS=(1618 2947 9258 5113 4321 6789 2413 7304 3963 4856 2704 4737) # 3957 9584 3948 2937)

for SEED in "${SEEDS[@]}"; do
    echo "Submitting job for SEED: ${SEED}"

    sbatch << EOF
#!/bin/bash

#SBATCH -J FiLM_PE_${DATASET}_${SEED}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_${DATASET}_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_${DATASET}_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

# Activate Environment
source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

# Fine-tuning ONLY logic

# Generate splits for this specific seed (mimicking pre-training behavior)
echo "Generating splits for seed ${SEED}..."
python3 scripts/data_util/generate_splits.py \
    --dataset "${DATASET}_fine-tuning.pkl" \
    --model "FiLM_PE" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}"

echo "Starting Fine-Tuning..."

python3 scripts/training/train.py \
  -M FiLM_PE \
  -D ${DEVICES} \
  -DS ${DATASET}_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef "${EMB}" \
  -S "${SEED}" \
  -e ${FT_EPOCHS} \
  -bs ${BS} \
  -lr ${LR} \
  --pe_dim ${PE_DIM} \
  --patience 5

EOF

    # Small delay between submissions to be nice to the scheduler
    sleep 1

done
