#!/usr/bin/env bash

for seed in 0 1 2; do
  python main.py \
    --env_name=antmaze-giant-navigate-singletask-v0 \
    --agent.discount=0.995 \
    --agent.alpha=30 \
    --project_name=FQL_reproduce \
    --seed="$seed"
done