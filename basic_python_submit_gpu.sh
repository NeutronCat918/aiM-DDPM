#!/bin/bash
#SBATCH -o train_model_gpu.out
#SBATCH -e train_model_gpu.err
#SBATCH -J train_GPU 
#SBATCH -n 1
#SBATCH --mem=200G
#SBATCH -p scavenger-gpu 
#SBATCH --gres=gpu:1 
#SBATCH --exclusive

srun python train_2_cluster.py