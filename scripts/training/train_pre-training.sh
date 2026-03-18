sbatch --wait << EOF
#!/bin/bash

#SBATCH -J RBP_pre-training_650M
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/bressin19_650M_pre-training%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/bressin19_650M_pre-training%j.txt
#SBATCH --time=03:00:00
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

SEED="${SEED:-2023}"

# esm2_t33_650M_UR50D esm2_t12_35M_UR50D esm2_t36_3B_UR50D  esm2_t30_150M_UR50D bressin19 RIC
python /path/to/RBP_IG/scripts/training/train.py \
    -D 1 \
    -M Lora \
    -DS bressin19_human_pre-training.pkl \
    -lm esm2_t33_650M_UR50D \
    --lm_provider synthyra \
    -ef bressin19 \
    -S "${SEED}" \
    -e 100 \
    -bs 1024 \
    --patience 20 \
    --lora-num-train-epochs 3 \
    --lora-alpha 0.8 \
    --lora-dropout 0.45 \
    --lora-learning-rate 3.5e-4 \
    --lora-r 3 \
    --lora-target-modules key value \
    --lora-weight-decay 2.0e-4 \
    --pe-dim 2
EOF
