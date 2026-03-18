sbatch --wait << EOF
#!/bin/bash

#SBATCH -J RBP_HPO_pre_train_650M
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/HPO_Bressin_650M_pre_train%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/HPO_Bressin_650M_pre_train%j.txt
#SBATCH --time=1-00:00:00
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
# esm2_t33_650M_UR50D esm2_t12_35M_UR50D
python /path/to/RBP_IG/scripts/training/train.py -D 1 -M Lora -DS bressin19_human_pre-training.pkl -lm esm2_t33_650M_UR50D --lm_provider synthyra -ef bressin19 -S 2023 -e 100 -bs 1024 --patience 20 --is_hpo
EOF