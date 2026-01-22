#!/bin/bash
#SBATCH --job-name=eval-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Defaults
CHECKPOINT_PATH=${1:-"/scratch/$USER/outputs/qwen3-mdlm-tulu3-sft-v1/checkpoint-final"}
TASKS=${2:-"gsm8k"}
LIMIT=${3:-10}  # Number of samples for "quick" eval
STEPS=${4:-32}  # Diffusion steps for evaluation

echo "Starting Evaluation for Qwen3-MDLM..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Steps: $STEPS"

# Environment Setup
source .env 2>/dev/null
export HF_HOME="/scratch/$USER/hf_cache"
export PYTHONPATH=".:./dllm:$PYTHONPATH"

# Load modules and activate environment
module load cuda/12.3.0
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

# Launch evaluation
accelerate launch \
    --num_processes 1 \
    dllm/dllm/pipelines/a2d/eval.py \
    --model mdlm \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --model_args "pretrained=$CHECKPOINT_PATH,max_new_tokens=256,steps=$STEPS,block_size=128" \
    --apply_chat_template

echo "Evaluation finished."
