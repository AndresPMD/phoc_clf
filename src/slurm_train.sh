#!/bin/bash
#SBATCH -n 1
#SBATCH --partition=gpu-mono
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint='gpu_16g|gpu_22g|gpu_32g|gpu_v100'
#SBATCH --output=/home/amafla/phoc_clf/slurm/out%j
# a file for errors from the job
#SBATCH --error=/home/amafla/phoc_clf/slurm/error%j

EP=$1
FUSION=$2

python train.py context --ocr yolo_phoc --embedding fisher --data_path /tmp-network/user/amafla/data/ --optim radam --batch_size 128 --model RMAC_Full --epsilon $EP --fusion $FUSION --save /home/amafla/phoc_clf/backup/
