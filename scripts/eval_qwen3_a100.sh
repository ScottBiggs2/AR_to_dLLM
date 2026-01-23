#!/bin/bash
#SBATCH --job-name=eval-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Defaults
CHECKPOINT_PATH=${1:-"/scratch/$USER/outputs/qwen3-mdlm-tulu3-sft-v1/checkpoint-6500"}
TASKS=${2:-"gsm8k"}
LIMIT=${3:--1}  # -1 for full evaluation
STEPS=${4:-32}  # Diffusion steps
BLOCK_SIZE=${5:-128}
BATCH_SIZE=${6:-32} # Reduced from 128 to avoid OOM on large vocabs

echo "Starting Evaluation for Qwen3-MDLM..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Steps: $STEPS"
echo "Batch Size: $BATCH_SIZE"

# Environment Setup
source .env 2>/dev/null
export HF_HOME="/scratch/$USER/hf_cache"
export PYTHONPATH=".:./dllm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Load modules and activate environment
module load cuda/12.3.0
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

# Handle optional limit
LIMIT_ARG=""
if [ "$LIMIT" != "-1" ] && [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Launch evaluation
accelerate launch \
    --num_processes 1 \
    --mixed_precision no \
    --dynamo_backend no \
    dllm/dllm/pipelines/a2d/eval.py \
    --model mdlm \
    --tasks "$TASKS" \
    $LIMIT_ARG \
    --batch_size "$BATCH_SIZE" \
    --model_args "pretrained=$CHECKPOINT_PATH,max_new_tokens=512,steps=$STEPS,block_size=$BLOCK_SIZE" \
    --apply_chat_template

echo "Evaluation finished."
