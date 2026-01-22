#!/bin/bash
#SBATCH --job-name=qwen3-sft-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00        # 4 hours is plenty for a 1-GPU debug/SFT run
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

# === CONFIGURATION ===
RUN_NAME="qwen3-mdlm-tulu3-sft-v1"
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="/scratch/$USER/data/sft/qwen3-0.6b/tulu-3"
OUTPUT_DIR="/scratch/$USER/outputs/$RUN_NAME"

# === ENVIRONMENT SETUP ===
export HF_HOME="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
export PYTHONPATH="$(pwd)/dllm:$(pwd):$PYTHONPATH"

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

# Activate environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

echo "Starting Qwen3 MDLM Multi-Node Training: $RUN_NAME"
echo "Nodes: $SLURM_NNODES"

# Launch using the slurm helper script which handles MASTER_ADDR, num_processes, etc.
bash dllm/scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "scripts/train_qwen3_mdlm.py" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_args "$DATA_DIR" \
    --load_preprocessed_data True \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 16 \
    --max_eval_samples 500 \
    --save_strategy steps \
    --save_steps 500

echo "Training finished."
