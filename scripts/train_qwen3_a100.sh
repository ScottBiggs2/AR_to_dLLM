#!/bin/bash
#SBATCH --job-name=qwen3-sft-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4       # Standard A100 allocation on most nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00        # Increase for real runs
#SBATCH --partition=gpu        # Per user request
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

# === CONFIGURATION ===
RUN_NAME="qwen3-mdlm-tulu3-sft-v1"
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="/scratch/$USER/data/sft/qwen3-0.6b/tulu-3"
OUTPUT_DIR="/scratch/$USER/outputs/$RUN_NAME"

# === ENVIRONMENT SETUP ===
# Cache redirections to avoid disk quota issues
export HF_HOME="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
export PYTHONPATH="$(pwd)/dllm:$PYTHONPATH"

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

# Activate environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

echo "Starting Qwen3 MDLM Training Run: $RUN_NAME"
echo "Output Directory: $OUTPUT_DIR"

# Launch Training
# Note: --num_processes should match GPUs (4 in this case)
accelerate launch \
    --config_file dllm/scripts/accelerate_configs/zero2.yaml \
    --num_processes 4 \
    scripts/train_qwen3_mdlm.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_args "$DATA_DIR" \
    --load_preprocessed_data True \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --report_to wandb \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 8 \
    --save_strategy steps \
    --save_steps 500

echo "Training finished."
