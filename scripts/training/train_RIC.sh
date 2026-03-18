sbatch --wait << EOF
#!/bin/bash

#SBATCH -J train_rbp
#SBATCH --output=$HOME/RBP_IG/scripts/sbatch_logs/training_logs_RIC_%j.txt
#SBATCH --error=$HOME/RBP_IG/scripts/sbatch_logs/training_logs_RIC_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig
cd /path/to/RBP_IG/

# esm2_t12_35M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D esm2_t6_8M_UR50D esm1b_t33_650M_UR50S esm1b_t12_35M_UR50S
# best params so far Trial 12 finished with value: 0.6111751794815063 and parameters: {'learning_rate': 0.0002835623769003636, 'lr_scheduler_type': 'cosine', 'warmup_steps': 4, 'weight_decay': 0.0009275695811762735, 'r': 4, 'lora_alpha': 0.61036561806247, 'lora_dropout': 0.07399531292334416, 'target_modules': ['query', 'key', 'value']}.
python3 /path/to/RBP_IG/scripts/training/train.py -M Lora -DS RIC_human_pre-training.pkl -lm esm2_t6_8M_UR50D -ef "RIC" -S 2023 -e 50 -bs 512 --patience 10

#python3 /path/to/RBP_IG/scripts/training/train.py -M Lora -DS InterPro_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "InterPro" -S 2023 -e 30 -bs 512 --patience 5

EOF