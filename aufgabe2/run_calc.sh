#!/bin/bash

#SBATCH --job-name=SemesterprojectKI
#SBATCH --nodes=1
#SBATCH --gres=gpu

module load cuda

python3 versuch2.3.py
