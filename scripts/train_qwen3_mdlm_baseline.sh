#!/bin/bash

# Default values
MODEL_PATH="models/a2d/Qwen3-0.6B"
DATA_PATH="data/sft/qwen3-0.6b/tulu-3"
OUTPUT_DIR="outputs/qwen3-0.6b-mdlm-baseline"

echo "Launching Qwen3-0.6B MDLM Baseline Training..."

accelerate launch \
    --config_file dllm/scripts/accelerate_configs/zero2.yaml \
    scripts/train_qwen3_mdlm.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_args "$DATA_PATH" \
    --load_preprocessed_data True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --report_to wandb \
    --logging_steps 10 \
    --save_steps 0.1 \
    --eval_steps 0.1 \
    --save_only_model True

echo "Training job launched."
