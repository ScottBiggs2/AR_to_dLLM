#!/bin/bash
#SBATCH --job-name=debug-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug-%j.out
#SBATCH --err=logs/debug-%j.err

# Defaults
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-false}
NUM_STEPS=${NUM_STEPS:-10}

echo "Starting Debug Run on H200..."
echo "Steps: $NUM_STEPS"
echo "Save Checkpoint: $SAVE_CHECKPOINT"

# Determine Save Strategy
if [ "$SAVE_CHECKPOINT" = true ]; then
    SAVE_STRATEGY="steps"
    SAVE_STEPS=5
else
    SAVE_STRATEGY="no"
    SAVE_STEPS=0
fi


# Environment Setup
source .env 2>/dev/null
export HF_HOME="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

# Add the dllm subdirectory to PYTHONPATH to allow importing 'examples'
export PYTHONPATH=$PYTHONPATH:$(pwd)/dllm

# Load modules and activate environment
module load cuda/12.3.0
# Initialize Conda from user's install (since module failed and caused fallback to base env)
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

# --- CHECK ACCELERATE CONFIGURATION (DIAGNOSTIC) ---
accelerate launch \
    --config_file dllm/scripts/accelerate_configs/zero2.yaml \
    scripts/check_accelerate.py

# --- REAL TRAINING RUN (COMMENTED OUT FOR DEBUGGING) ---
# accelerate launch \
#     --config_file dllm/scripts/accelerate_configs/zero2.yaml \
#     scripts/train_qwen3_mdlm.py \
#     --model_name_or_path "Qwen/Qwen3-0.6B" \
#     --dataset_args "data/sft/qwen3-0.6b/tulu-3" \
#     --load_preprocessed_data True \
#     --max_steps $NUM_STEPS \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1e-5 \
#     --warmup_ratio 0.1 \
#     --output_dir "outputs/debug-qwen3" \
#     --report_to wandb \
#     --logging_steps 1 \
#     --eval_strategy no \
#     --save_strategy $SAVE_STRATEGY \
#     --save_steps $SAVE_STEPS

echo "Debug run finished."
