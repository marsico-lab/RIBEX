#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${SEED:=2023}"
: "${LM_NAME:=protT5_xl_uniref50}" # protT5_xl_uniref50 or esm2_t33_650M_UR50D
: "${EMB:=RIC}" # RIC or bressin19
: "${PRE_EPOCHS:=40}"
: "${FT_EPOCHS:=50}"
: "${BS:=256}"
: "${LR:=0.005}"
: "${DEVICES:=0}"

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
CKPT_FOLDER="Peng_${LM_SHORT}-E=${EMB}-S=${SEED}-E=${PRE_EPOCHS}-BS=${BS}-LR=${LR_FMT}"

sbatch --wait << EOF
#!/bin/bash

#SBATCH -J RBP_fine_tuning_650M
#SBATCH --output=
#SBATCH --output=
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/Peng_${EMB}_650M_fine_tuning%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/Peng_${EMB}_650M_fine_tuning%j.txt
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

# # Pre-training
# python3 scripts/training/train.py \
#   -M Peng \
#   -D ${DEVICES} \
#   -DS ${EMB}_human_pre-training.pkl \
#   -lm ${LM_NAME} \
#   -ef "${EMB}" \
#   -S "${SEED}" \
#   -e ${PRE_EPOCHS} \
#   -bs ${BS} \
#   -lr ${LR} \
#   --patience 10

# Fine-tuning (load from the exact pre-training run folder name)
python3 scripts/training/train.py \
  -M Peng \
  -D ${DEVICES} \
  -DS ${EMB}_human_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef "${EMB}" \
  -S "${SEED}" \
  -e ${FT_EPOCHS} \
  -bs ${BS} \
  -lr ${LR} \
  --patience 5 \
  ##--checkpoint-folder "${CKPT_FOLDER}"

EOF
