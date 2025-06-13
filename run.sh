#!/bin/bash

# This is a general purpose script that is designed to be a run a cluster system and it setups an experiment on one of my on creations.
# This script needs to be passed a task id as an argument as well as the algorithm to run as another arguement

cd ~/code/sunrise


task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
algorithm=$2

num_seeds=10
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "HumanoidStandup-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

# Get current date/time in yy/m/d_h-m-s format
current_time=$(date +"%y-%m-%d_%H-%M-%S")
exp_name="${selected_env}_${seed}_${current_time}"

echo " Running ${selected_env} with seed ${seed}"


training_script=OpenAIGym_SAC/examples/$algorithm.py

# Pass on any arbitarary kwargs to the training script


echo "==Running ${training_script}=="

# Check if --no-redirect is passed
if [[ " $@ " =~ " --no-redirect " ]]; then
    exp_dir=${OUTPUTDIR}/${selected_env}_${seed}
    # Run without redirecting logs
    echo "==Running ${training_script} without log redirection=="
    poetry run python ${training_script} --seed=${seed} --exp_dir="$exp_dir" --env=${selected_env} --exp_name="${exp_name}" "${@:4}"
else
    OUTPUTDIR=experiments
    exp_dir="${PWD}/${OUTPUTDIR}/${algorithm}/${exp_name}"
    mkdir -p "$exp_dir"
    log_file="$exp_dir/logs.txt"
    # Run with log redirection
    echo "==Running ${training_script} with log redirection=="
    poetry run python ${training_script} --seed=${seed} --exp_dir="$exp_dir" --env=${selected_env} --exp_name="${exp_name}" "${@:3}" > "$log_file" 2>&1 &
    echo "==${training_script} submitted and running as name ${exp_name} with PID ${!}=="
fi