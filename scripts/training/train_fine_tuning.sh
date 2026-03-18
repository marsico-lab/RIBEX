sbatch --wait << EOF
#!/bin/bash

#SBATCH -J RBP_fine_tuning_3B
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/bressin19_3B_fine_tuning%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/bressin19_3B_fine_tuning%j.txt
#SBATCH --time=10:00:00
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
    -DS bressin19_human_fine-tuning.pkl \
    -lm esm2_t36_3B_UR50D \
    --lm_provider synthyra \
    -ef bressin19 \
    -S "${SEED}" \
    -e 100 \
    -bs 1024 \
    --patience 20 \
    --lora-num-train-epochs 6 \
    --lora-alpha 0.12 \
    --lora-dropout 0.50 \
    --lora-learning-rate 1.3e-3 \
    --lora-r 6 \
    --lora-weight-decay 0 \
    --pe-dim 512
EOF
