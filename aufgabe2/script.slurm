#!/bin/bash

#SBATCH --job-name=SemesterprojectKIGruppeB
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu

module load cuda

CUDA_LAUNCH_BLOCKING=1 python3 versuch2.3.py
