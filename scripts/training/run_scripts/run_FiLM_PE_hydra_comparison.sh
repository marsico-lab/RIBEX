#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${SEED:=2023}"
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}"
: "${DATASET:=hydra_comparison}"
: "${FT_EPOCHS:=50}"
: "${BS:=512}"
: "${LR:=0.000093}"
: "${DEVICES:=0}"
: "${PE_DIM:=1062}"

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

# Output log inside sbatch logic will be handled by SLURM, but we can also use specific log files if needed.
# Converting to sbatch structure...

sbatch --wait << EOF
#!/bin/bash

#SBATCH -J FiLM_PE_${DATASET}
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

# Note: pre-training would normally generate splits. We bypassed this by generating them manually.
# The filename logic in utils.py will search for splits based on "fine-tuning.pkl" name and seed.

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
