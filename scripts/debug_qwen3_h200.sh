#!/bin/bash
#SBATCH --job-name=debug-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
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

source .env 2>/dev/null

accelerate launch \
    --config_file dllm/scripts/accelerate_configs/zero2.yaml \
    scripts/train_qwen3_mdlm.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "data/sft/qwen3-0.6b/tulu-3" \
    --load_preprocessed_data True \
    --max_steps $NUM_STEPS \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --output_dir "outputs/debug-qwen3" \
    --report_to wandb \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS

echo "Debug run finished."
