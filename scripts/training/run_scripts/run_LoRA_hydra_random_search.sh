#!/usr/bin/env bash

# Parameters (can be overridden)
: "${DATASET:=hydra_comparison}"
: "${SEED:=8888}"
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}"
: "${FT_EPOCHS:=15}"
: "${DEVICES:=0}"
: "${NUM_TRIALS:=30}" # Number of random experiments to run
: "${LORA_BS:=2}"     # Per-device batch size (keep low for large models!)

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

echo "Starting LoRA Random Search with ${NUM_TRIALS} trials..."

for ((i=1; i<=NUM_TRIALS; i++)); do

    # 1. Random Learning Rate (Centered around 6.12e-4)
    # Range ~ 3e-4 to 1e-3
    LR=$(python3 -c "import random; print(f'{10**random.uniform(-3.52, -3.0):.6f}')")

    # 2. Random Batch Size (Centered around 422)
    # Range 380 to 460
    BS=$(python3 -c "import random; print(random.randint(380, 460))")

    # 3. Random PE_DIM (Centered around 450)
    # Range 400 to 500
    PE_DIM=$(python3 -c "import random; print(random.randint(400, 500))")

    # 4. Random LoRA Rank (Centered around 7)
    # Range 4 to 10
    LORA_R=$(python3 -c "import random; print(random.randint(4, 10))")

    # 5. Random LoRA Alpha (Centered around 1.70)
    # Range 1.5 to 2.0
    LORA_ALPHA=$(python3 -c "import random; print(f'{random.uniform(1.5, 2.0):.2f}')")

    # 6. Random LoRA Dropout (Centered around 0.42)
    # Range 0.35 to 0.50
    LORA_DROPOUT=$(python3 -c "import random; print(f'{random.uniform(0.35, 0.50):.2f}')")

    # 7. Random LoRA Weight Decay (Centered around 2.72e-4)
    # Range 1e-4 to 5e-4
    LORA_WD=$(python3 -c "import random; print(f'{10**random.uniform(-4.0, -3.3):.6f}')")


    echo "Trial $i/$NUM_TRIALS: LR=$LR, BS=$BS, PE_DIM=$PE_DIM, R=$LORA_R, Alpha=$LORA_ALPHA, Drop=$LORA_DROPOUT, WD=$LORA_WD"

    sbatch << EOF
#!/bin/bash

#SBATCH -J LoRA_${LM_SHORT}_Rand_${i}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${DATASET}_${LM_SHORT}_RandSearch_${i}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/LoRA_${DATASET}_${LM_SHORT}_RandSearch_${i}_%j.txt
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

# Generate splits (if needed, usually seed dependent)
python3 scripts/data_util/generate_splits.py \
    --dataset "${DATASET}_fine-tuning.pkl" \
    --model "Lora" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}"

echo "Starting LoRA Fine-Tuning Trial $i..."

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
  --lora_weight_decay ${LORA_WD} \
  --patience 10

EOF

    # Small sleep
    sleep 1
done

echo "Submitted all ${NUM_TRIALS} trials."
