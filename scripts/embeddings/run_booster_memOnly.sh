#!/bin/sh


#SBATCH --account=hai_ml4rg_rbp
#SBATCH --job-name=EMB
##SBATCH --mail-type=END, FAIL
##SBATCH --mail-user=your.email@example.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/EMB/%x_%j-out.log
#SBATCH --error=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/EMB/%x_%j-err.log
#SBATCH --time=2:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:0


jutil env activate -p hai_ml4rg_rbp
module load Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 Python SciPy-Stack PyTorch PyTorch/1.12.0-CUDA-11.7 tqdm Biopython
module load nano

cd /p/project/hai_ml4rg_rbp/Project_ml4rg
pwd
#python3 generateEmbeddings.py --device cpu --languageModel esm1b_t33_650M_UR50S --maxSeqLen 1024
#python3 generateEmbeddings.py --device cpu --languageModel esm2_t33_650M_UR50D --maxSeqLen 1024
#python3 generateEmbeddings.py --device cpu --languageModel esm2_t36_3B_UR50D --maxSeqLen 1024
python3 generateEmbeddings.py --device cpu --languageModel esm2_t48_15B_UR50D --maxSeqLen 1024

