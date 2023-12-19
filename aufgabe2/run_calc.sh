#!/bin/bash

#SBATCH --job-name=SemesterprojectKIGruppeB
#SBATCH --nodes=4
#SBATCH --gres=gpu:rtx6000

module load cuda

TORCH_CUDA_ARCH_LIST="7.0" CUDA_LAUNCH_BLOCKING=1 python3 versuch2.3.py
