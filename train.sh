#!/bin/bash

#SBATCH --account hc86
#SBATCH --partition commons
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal
#SBATCH --job-name="train_prompt_tuning"
#SBATCH --output="/home/zf28/align/log/train_peft-%j.txt"

# hf download meta-llama/Meta-Llama-3.1-8B  --exclude "original/*" --token xxxx
export HF_HOME="/scratch/${USER}/huggingface"

model=$1
dataset=$2
lr=$3
peft=$4
rank=$5
prompt_length=$6

log_name="/home/zf28/align/COMP576_PEFT/logs/${model}_${dataset}_lr${lr}_${peft}_rank${rank}_len${prompt_length}.txt"

module purge
module load CUDA/12.8.0

srun python -u -m train \
        --model "$model" \
        --dataset "$dataset" \
        --lr "$lr" \
        --peft "$peft" \
        --rank "$rank" \
        --prompt_length "$prompt_length" \
        > "$log_name" 2>&1