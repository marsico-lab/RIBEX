#!/usr/bin/env bash

# Sweep LoRA rank values during pre-training in a single sbatch job.
sbatch --wait <<'EOF'
#!/bin/bash

#SBATCH -J RBP_pre-training_lora_r
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_lora_r_pre-training%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_lora_r_pre-training%j.txt
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd /path/to/RBP_IG/

for RANK in 2 3 4 5 6 8; do
    echo "=== LoRA rank: ${RANK} ==="
    python scripts/training/train.py \
        -D 1 \
        -M Lora \
        -DS RIC_human_pre-training.pkl \
        -lm esm2_t30_150M_UR50D \
        --lm_provider synthyra \
        -ef RIC \
        -S 2023 \
        -e 10 \
        -bs 1024 \
        --patience 20 \
        --lora-r "${RANK}" \
        --lora-num-train-epochs 10 \
        --pe-dim 128
done
EOF
