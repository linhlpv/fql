#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/s222493549/my-vast/code/offline2online/fql/exps/slurms/slurm_%j.out 
#SBATCH --qos=batch-long
#SBATCH --constraint=nvidia-driver-565
#SBATCH --exclude=v100l-f-04

module load Anaconda3
source activate
conda activate fql
cd /home/s222493549/my-vast/code/offline2online/fql
nvidia-smi


for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py \
    --env_name=puzzle-4x4-play-singletask-v0 \
    --agent.discount=0.995 \
    --agent.alpha=30 \
    --project_name=FQL_reproduce \
    --seed="$seed"
done


for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py \
    --env_name=antmaze-giant-navigate-singletask-v0 \
    --agent.discount=0.995 \
    --agent.alpha=30 \
    --project_name=FQL_reproduce \
    --seed="$seed"
done


for seed in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py \
    --env_name=antmaze-large-navigate-singletask-v0 \
    --agent.discount=0.995 \
    --agent.alpha=30 \
    --project_name=FQL_reproduce \
    --seed="$seed"
done