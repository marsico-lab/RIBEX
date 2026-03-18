#!/bin/sh


#SBATCH --account=hai_ml4rg_rbp
#SBATCH --job-name=EMB
##SBATCH --mail-type=END, FAIL
##SBATCH --mail-user=your.email@example.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/EMB/%x_%j-out.log
##SBATCH --output=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/EMB/%x_%j_%N-out.log
#SBATCH --error=/p/project/hai_ml4rg_rbp/Project_ml4rg/logs/cluster/EMB/%x_%j-err.log
#SBATCH --time=00:10:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1


jutil env activate -p hai_ml4rg_rbp
module load Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 NCCL Python SciPy-Stack PyTorch PyTorch/1.12.0-CUDA-11.7 tqdm Biopython
module load nano

cd /p/project/hai_ml4rg_rbp/Project_ml4rg
pwd

#esm1b_t33_650M_UR50S
python3 scripts/embeddings/generate.py --device cuda --languageModel esm1b_t33_650M_UR50S --precision f32

#esm2_t33_650M_UR50D
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t33_650M_UR50D --precision f16 --fsdp

# esm2_t36_3B_UR50D on 40 VRAM requires half precision
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t36_3B_UR50D --precision f16

# esm2_t48_15B_UR50D on 40 VRAM does not even work with half precision for 1024
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t48_15B_UR50D --precision f16
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t48_15B_UR50D --precision f16 --fsdp

