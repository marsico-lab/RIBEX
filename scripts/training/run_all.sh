#!/bin/sh

#Peng 650M
echo Starting TRAIN_P_B_650M && sbatch run_scripts/run_Peng_Bressin_650M.sh
echo Starting TRAIN_P_R_650M && sbatch run_scripts/run_Peng_RIC_650M.sh
echo Starting TRAIN_P_I_650M && sbatch run_scripts/run_Peng_InterPro_650M.sh

