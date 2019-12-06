#!/bin/bash
#SBATCH -n 1
#SBATCH --partition=gpu-mono
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint='gpu_16g|gpu_22g|gpu_32g|gpu_v100'
#SBATCH --output=/tmp-network/user/%u/slurm/%j-out.out
# a file for errors from the job
#SBATCH --error=/tmp-network/user/%u/slurm/%j-error.err

EP=$1
FUSION=$2
REG=$3

python train.py context --ocr yolo_phoc --embedding fisher --fusion $FUSION --data_path /tmp-network/user/amafla/data/ --optim radam --model RMAC_Full --epsilon $EP --regularization $REG --save /tmp-network/user/rsampaio/models/finegrained-classif/
