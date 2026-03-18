#!/usr/bin/env bash

# Parameters (can be overridden)
: "${DATASET:=hydra_comparison}" 
: "${SEED:=2023}"
: "${LM_NAME:=protT5_xl_uniref50}"
: "${EMB:=RIC}" 
: "${FT_EPOCHS:=30}"
: "${DEVICES:=0}"
: "${NUM_TRIALS:=30}" # Number of random experiments to run

# Map LM_NAME to the short token
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

echo "Starting Random Search with ${NUM_TRIALS} trials..."

for ((i=1; i<=NUM_TRIALS; i++)); do
    
    # 1. Random Learning Rate (Broad range for ESM 3B/15B)
    # Range 1e-5 to 1e-3
    # 0.000225
    LR=$(python3 -c "import random; print(f'{10**random.uniform(-5, -3):.6f}')")
    
    # 2. Random Batch Size (Narrower range around 429)
    # Range 380 to 480
    # 410
    BS=$(python3 -c "import random; print(random.randint(100, 500))")
    
    # 3. Random PE_DIM (Narrower range around 812)
    # Range 700 to 924
    # 912
    PE_DIM=$(python3 -c "import random; print(random.randint(128, 1224))")
    
    echo "Trial $i/$NUM_TRIALS: LR=$LR, BS=$BS, PE_DIM=$PE_DIM"
    
    # Submit Job
    # Using 'sbatch' directly (no --wait) so they run in parallel
    sbatch << EOF
#!/bin/bash

#SBATCH -J FiLM_${LM_SHORT}_Rand_${i}
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_RandSearch_${i}_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/FiLM_PE_RandSearch_${i}_%j.txt
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

# Activate Environment
source ~/.bashrc
source $HOME/miniconda3/bin/activate rbp_ig_lustre
cd $HOME/RBP_IG/

# Generate Splits (Just in case, though usually seed is fixed to 2023 for all trials to be comparable on data)
# Note: Should we vary seed? User's request implies varying hyperparameters. Keeping seed fixed allows fair comparison.
python3 scripts/data_util/generate_splits.py \
    --dataset "${DATASET}_fine-tuning.pkl" \
    --model "FiLM_PE" \
    --seed "${SEED}" \
    --lm_name "${LM_NAME}" \
    --emb_name "${EMB}"

# Run Training
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

    # Small sleep to avoid overwhelming scheduler momentarily
    sleep 1
done

echo "Submitted all ${NUM_TRIALS} trials."
