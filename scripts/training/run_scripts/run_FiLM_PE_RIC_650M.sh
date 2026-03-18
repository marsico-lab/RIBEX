#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${SEED:=2222}"
: "${LM_NAME:=esm2_t33_650M_UR50D}" # protT5_xl_uniref50 or esm2_t33_650M_UR50D or esm2_t48_15B_UR50D or esm2_t36_3B_UR50D
: "${EMB:=RIC}"
: "${DATASET:=RIC_2}"   # RIC or bressin19 or RIC_2
: "${PRE_EPOCHS:=10}"
: "${FT_EPOCHS:=30}"
: "${BS:=256}"
: "${LR:=0.0005}"
: "${DEVICES:=0}"
: "${PE_DIM:=512}"
: "${FORCED_GENES:=O75808}"
: "${EXCLUSIVE_VAL:=True}"

# Map LM_NAME to the short token used in Peng run folders
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
# Note: FiLM_PE might need to include PE_DIM in the folder name if utils.py uses it for naming.
# Checking utils.py: params['model_file_name'] = f"{params['model_name']}_{LM_name_short}-E={params['embeddingSubfolder']}-S={params['seed']}-E={params['epochs']}-BS={params['bs']}-LR={params['lr']:.6f}"
# It does NOT include PE_DIM in the standard naming (only for Lora).
# So the folder name logic remains the same as Peng.
# set PE_DIM_PRE_TRAINING=2
PE_DIM_PRE_TRAINING=2

if [ "${PE_DIM_PRE_TRAINING}" -gt 2 ]; then
    PE_SUFFIX="-PE=${PE_DIM_PRE_TRAINING}"
else
    PE_SUFFIX="-PE=No"
fi

CKPT_FOLDER="FiLM_PE_${LM_SHORT}-E=${EMB}-S=${SEED}-E=${PRE_EPOCHS}-BS=${BS}-LR=${LR_FMT}${PE_SUFFIX}"

sbatch --wait << EOF
#!/bin/bash

#SBATCH -J RBP_fine_tuning_650M_FiLM_PE
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_${EMB}_650M_fine_tuning%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_${EMB}_650M_fine_tuning%j.txt
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

SEED="${SEED}"

# # # Pre-training
# # # Note: FiLM_PE will ignore PE during pre-training (dataset returns None), behaving like a pooled MLP.
# python3 scripts/training/train.py \
#   -M FiLM_PE \
#   -D ${DEVICES} \
#   -DS ${EMB}_human_pre-training.pkl \
#   -lm ${LM_NAME} \
#   -ef "${EMB}" \
#   -S "${SEED}" \
#   -e ${PRE_EPOCHS} \
#   -bs ${BS} \
#   -lr ${LR} \
#   --pe_dim ${PE_DIM_PRE_TRAINING} \
#   --patience 10


# Fine-tuning ONLY logic

# Note: pre-training would normally generate splits. We bypassed this by generating them manually.
# The filename logic in utils.py will search for splits based on "fine-tuning.pkl" name and seed.

# Generate splits for this specific seed (mimicking pre-training behavior)
echo "Generating splits for seed ${SEED}..."
python3 scripts/data_util/generate_splits.py \
    --dataset "${DATASET}_human_fine-tuning.pkl" \
    --model "FiLM_PE" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}" \
    --forced_val_genes "${FORCED_GENES}" \
    --forced_val_genes_only

# Fine-tuning (load from the exact pre-training run folder name)
# Note: FiLM_PE will use PE during fine-tuning.
python3 scripts/training/train.py \
  -M FiLM_PE \
  -D ${DEVICES} \
  -DS ${DATASET}_human_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef "${EMB}" \
  -S "${SEED}" \
  -e ${FT_EPOCHS} \
  -bs ${BS} \
  -lr ${LR} \
  --pe_dim ${PE_DIM} \
  --patience 5 \
  --keep_small_val_batches \
##  --checkpoint-folder "${CKPT_FOLDER}" \

EOF
