#!/bin/bash
#SBATCH -n 1
#SBATCH --partition=gpu-mono
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint='gpu_16g|gpu_22g|gpu_32g|gpu_v100'
#SBATCH --output=/tmp-network/user/amafla/slurm/out.out
# a file for errors from the job
#SBATCH --error=/tmp-network/user/amafla/slurm/error.err

python train.py context --ocr yolo_phoc --embedding fisher --fusion mlb --data_path /tmp-network/user/amafla/data/ --optim radam --model RMAC_Full --epsilon 0

