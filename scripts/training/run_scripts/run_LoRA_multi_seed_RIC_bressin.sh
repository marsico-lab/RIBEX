#!/usr/bin/env bash

# Parameters (customize via environment overrides when calling this script)
: "${LM_NAME:=esm2_t33_650M_UR50D}" # Default to 3B as in run_LoRA_hydra_comparison.sh
: "${FT_EPOCHS:=20}"
: "${BS:=256}"         # Global batch size
: "${LORA_BS:=2}"       # Per-device batch size
: "${LR:=0.0003}"       # LoRA learning rate
: "${DEVICES:=0}"       # GPU device ID
: "${PE_DIM:=512}"      # Positional Encoding dimension

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

# Seeds from run_FiLM_PE_hydra_comparison_multi_seed.sh
SEEDS=(
    2842
    5923
    8347
    1290
    4751
    7634
    9821
    1234
    4567
    8901
    1357
    2468
    3579
)
# Embeddings to iterate over
EMBEDDINGS=("RIC" "bressin19")

for EMB in "${EMBEDDINGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Submitting LoRA job for Embedding: ${EMB}, Seed: ${SEED}"

        sbatch << EOF
#!/bin/bash

#SBATCH -J LoRA_${EMB}_${SEED}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${EMB}_${LM_SHORT}_${SEED}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${EMB}_${LM_SHORT}_${SEED}_%j.txt
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

# Generate splits for this specific seed
echo "Generating splits for seed ${SEED} (Emb: ${EMB})..."
python3 scripts/data_util/generate_splits.py \
    --dataset "${EMB}_human_fine-tuning.pkl" \
    --model "Lora" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}"

echo "Starting LoRA Fine-Tuning..."
python3 scripts/training/train.py \
  -M Lora \
  -D ${DEVICES} \
  -DS ${EMB}_human_fine-tuning.pkl \
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

        # Sleep to be nice to scheduler
        sleep 1
    done
done
