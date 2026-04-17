#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=9
#SBATCH --mem=64G
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


# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 \
#     --agent.alpha=100 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 \
#     --agent.alpha=100 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 \
#     --agent.alpha=100 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 \
#     --agent.alpha=100 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 \
#     --agent.alpha=100 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done


# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-large-navigate-singletask-task1-v0 \
#     --agent.alpha=30 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-large-navigate-singletask-task2-v0 \
#     --agent.alpha=30 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-large-navigate-singletask-task3-v0 \
#     --agent.alpha=30 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-large-navigate-singletask-task4-v0 \
#     --agent.alpha=30 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=humanoidmaze-large-navigate-singletask-task5-v0 \
#     --agent.alpha=30 --agent.discount=0.995
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done



# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-large-navigate-singletask-task1-v0 \
#     --agent.alpha=10 --agent.q_agg=min \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-large-navigate-singletask-task2-v0 \
#     --agent.alpha=10 --agent.q_agg=min \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-large-navigate-singletask-task3-v0 \
#     --agent.alpha=10 --agent.q_agg=min \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-large-navigate-singletask-task4-v0 \
#     --agent.alpha=10 --agent.q_agg=min \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-large-navigate-singletask-task5-v0 \
#     --agent.alpha=10 --agent.q_agg=min \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done


# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-giant-navigate-singletask-task1-v0 \
#     --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-giant-navigate-singletask-task2-v0 \
#     --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-giant-navigate-singletask-task3-v0 \
#     --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-giant-navigate-singletask-task4-v0 \
#     --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#   CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antmaze-giant-navigate-singletask-task5-v0 \
#     --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done


# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antsoccer-arena-navigate-singletask-task1-v0 \
#     --agent.alpha=30 --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antsoccer-arena-navigate-singletask-task2-v0 \
#     --agent.alpha=30 --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antsoccer-arena-navigate-singletask-task3-v0 \
#     --agent.alpha=30 --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antsoccer-arena-navigate-singletask-task4-v0 \
#     --agent.alpha=30 --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=antsoccer-arena-navigate-singletask-task5-v0 \
#     --agent.alpha=30 --agent.discount=0.995 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=cube-double-play-singletask-task1-v0 \
#     --agent.alpha=300 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=cube-double-play-singletask-task2-v0 \
#     --agent.alpha=300 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=cube-double-play-singletask-task3-v0 \
#     --agent.alpha=300 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=cube-double-play-singletask-task4-v0 \
#     --agent.alpha=300 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=cube-double-play-singletask-task5-v0 \
#     --agent.alpha=300 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done


# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=puzzle-4x4-play-singletask-task1-v0 \
#     --agent.alpha=1000 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=puzzle-4x4-play-singletask-task2-v0 \
#     --agent.alpha=1000 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=puzzle-4x4-play-singletask-task3-v0 \
#     --agent.alpha=1000 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=puzzle-4x4-play-singletask-task4-v0 \
#     --agent.alpha=1000 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done

# for seed in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 /home/s222493549/my-vast/.conda/envs/fql/bin/python main.py --online_steps=500000 --env_name=puzzle-4x4-play-singletask-task5-v0 \
#     --agent.alpha=1000 \
#     --project_name=FQL_reproduce \
#     --seed="$seed"
# done
