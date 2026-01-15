#!/bin/bash
#SBATCH --job-name=qwen3-mdlm-baseline
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot

# This template adapts the dllm/scripts/train.slurm.sh for Qwen3-0.6B conversion.
# Usage: sbatch scripts/train_qwen3_mdlm_slurm.sh

# User should ensure WANDB_API_KEY is in .env or environment
source .env 2>/dev/null

# Set the script path relative to the root
SCRIPT_PATH="scripts/train_qwen3_mdlm.py"
ACCELERATE_CONFIG="zero2" # or fsdp

# Launch via the repo's slurm helper
bash dllm/scripts/train.slurm.sh \
    --accelerate_config "$ACCELERATE_CONFIG" \
    --script_path "$SCRIPT_PATH" \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "data/sft/qwen3-0.6b/tulu-3" \
    --load_preprocessed_data True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --output_dir "outputs/qwen3-0.6b-mdlm-multinode" \
    --report_to wandb \
    --save_steps 0.1 \
    --eval_steps 0.1
