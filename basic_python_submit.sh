#!/bin/bash
#SBATCH -o train_model.out
#SBATCH -e train_model.err
#SBATCH -J train_DDPM 
#SBATCH -n 1
#SBATCH -p scavenger

srun python train_2_cluster.py