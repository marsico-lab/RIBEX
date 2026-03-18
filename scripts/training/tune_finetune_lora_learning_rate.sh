#!/usr/bin/env bash

# Sweep LoRA learning rates during fine-tuning of the 150M model.
sbatch --wait <<'EOF'
#!/bin/bash

#SBATCH -J RBP_fine_tuning_lora_lr
#SBATCH --output=/path/to/RBP_IG/scripts/sbatch_logs/RIC_lora_lr_fine-tuning%j.txt
#SBATCH --error=/path/to/RBP_IG/scripts/sbatch_logs/RIC_lora_lr_fine-tuning%j.txt
#SBATCH --time=15:00:00
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

for LR in 5.0e-04 7.5e-04 9.0e-04 1.0e-03 1.1e-03 1.3e-03; do
    echo "=== LoRA learning rate: ${LR} ==="
    python scripts/training/train.py \
        -D 1 \
        -M Lora \
        -DS RIC_human_fine-tuning.pkl \
        -lm esm2_t33_650M_UR50D \
        --lm_provider synthyra \
        -ef RIC \
        -S 2023 \
        -e 10 \
        -bs 1024 \
        --patience 20 \
        --lora-learning-rate "${LR}" \
        --lora-num-train-epochs 5 \
        --lora-alpha 0.2 \
        --lora-dropout 0.6 \
        --lora-r 2 \
        --lora-weight-decay 5.0e-5 \
        --lora-target-modules key value \
        --pe-dim 192
done
EOF
