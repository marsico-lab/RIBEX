#!/bin/sh


#SBATCH --account=hai_ml4rg_rbp
#SBATCH --job-name=GET_IP
##SBATCH --mail-type=END, FAIL
##SBATCH --mail-user=your.email@example.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/DATA_RAW/%x_%j-out.log
##SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/DATA_RAW/%x_%j_%N-out.log
#SBATCH --error=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/DATA_RAW/%x_%j-err.log
#SBATCH --time=00:05:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1


jutil env activate -p hai_ml4rg_rbp
module load Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 NCCL Python SciPy-Stack PyTorch PyTorch/1.12.0-CUDA-11.7 tqdm Biopython
module load nano

cd /p/project/hai_ml4rg_rbp/Project_ml4rg
pwd

#Login node: 1.66it/s
#Pre training
python3 scripts/data_raw/generate_InterPro.py
