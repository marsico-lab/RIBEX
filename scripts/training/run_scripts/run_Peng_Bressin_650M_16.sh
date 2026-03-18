#!/bin/sh


#SBATCH --account=hai_ml4rg_rbp
#SBATCH --job-name=TRAIN_P_B_650M
##SBATCH --mail-type=END, FAIL
##SBATCH --mail-user=your.email@example.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/TRAIN/%x_%j-out.log
##SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/TRAIN/%x_%j_%N-out.log
#SBATCH --error=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/TRAIN/%x_%j-err.log
#SBATCH --time=00:45:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1


jutil env activate -p hai_ml4rg_rbp
module load Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 NCCL Python SciPy-Stack PyTorch PyTorch/1.12.0-CUDA-11.7 tqdm Biopython
module load nano

cd /p/project/hai_ml4rg_rbp/Project_ml4rg
pwd

# Peng & esm1b: 100 epochs ~ 20min

#Pre training
#python3 scripts/training/train.py -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -e 100 -bs 512 --patience 20
python3 scripts/training/train.py -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -e 100 -bs 16 --patience 20
python3 scripts/training/train.py -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2024 -e 100 -bs 16 --patience 20

#Fine tune
#python3 scripts/training/train.py -M Peng -DS bressin19_human_fine-tuning.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" --checkpoint-folder "LM=esm1b_t33_650M_UR50S-E=bressin19-S=2023-E=100-BS=512" -S 2024 -e 300 -bs 512 --patience 20
