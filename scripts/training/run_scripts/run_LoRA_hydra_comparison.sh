#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${SEED:=2023}"
: "${LM_NAME:=esm2_t36_3B_UR50D}"
: "${EMB:=RIC}"
: "${DATASET:=hydra_comparison}"
: "${FT_EPOCHS:=20}"
: "${BS:=256}"         # Global batch size
: "${LORA_BS:=2}"       # Per-device batch size (keep low for large models!)
: "${LR:=0.0003}"       # LoRA learning rate (often higher than full FT)
: "${DEVICES:=0}"       # GPU device ID
: "${PE_DIM:=512}"      # Positional Encoding dimension (0 to disable)

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
PE_SUFFIX=""
if [ "${PE_DIM}" -gt 0 ]; then
    PE_SUFFIX="-PE=${PE_DIM}"
else
    PE_SUFFIX="-PE=No"
fi

# Output log inside sbatch logic will be handled by SLURM
# Converting to sbatch structure...

sbatch --wait << EOF
#!/bin/bash

#SBATCH -J LoRA_${DATASET}
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

# Generate splits for this specific seed if needed
# Note: we use model name "Lora" so the split generator knows we might need a tokenizer (though Generate splits usually just needs embeddings/groups)
# However, generate_splits might be model-agnostic regarding the output file if we name it consistently.
# The original script passed --model "FiLM_PE". We pass "Lora" here.

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
  --patience 10

EOF
