#!/bin/bash

# This is a general purpose script that is designed to be a run a cluster system and it setups an experiment on one of my on creations.
# This script needs to be passed a task id as an argument as well as the algorithm to run as another arguement

OUTPUTDIR=experiments

task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
num_seeds=10
num_algos=3
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "HumanoidStandup-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

# Get current date/time in yy/m/d_h-m-s format
current_time=$(date +"%y-%m-%d_%H-%M-%S")
exp_name="${selected_env}_${seed}_${current_time}"

exp_dir="${PWD}/${OUTPUTDIR}/${exp_name}"

mkdir -p "$exp_dir"
log_file="$exp_dir/logs.txt"

echo " Running ${selected_env} with seed ${seed}"

training_script=OpenAIGym_SAC/examples/dsunrise.py

echo "==Running ${training_script}=="

poetry run python ${training_script} --seed=${seed} --exp-dir="$exp_dir" --env=${selected_env} --exp-name="${exp_name}" "${@:3}" > "$log_file" 2>&1 &
echo "==${training_script} submitted and running as name ${exp_name} with PID ${!}=="