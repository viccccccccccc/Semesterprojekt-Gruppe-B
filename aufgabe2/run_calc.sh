#!/bin/bash

#SBATCH --job-name=SemesterprojectKIGruppeB
#SBATCH --nodes=1
#SBATCH --gres=gpu

module load cuda

TORCH_CUDA_ARCH_LIST="7.0" CUDA_LAUNCH_BLOCKING=1 python3 versuch2.3.py
