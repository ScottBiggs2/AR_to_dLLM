#!/bin/bash
#SBATCH --job-name=qwen3-scaling
#SBATCH --reservation=biggs.s_test
# #SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=18:00:00
#SBATCH --exclusive
#SBATCH --output=logs/scaling-%j.out
#SBATCH --err=logs/scaling-%j.err

# === CONFIGURATION ===
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="/scratch/$USER/data/sft/qwen3-0.6b/tulu-3"
COMMON_OUTPUT_DIR="/scratch/$USER/outputs/scaling_tests"
WANDB_PROJECT="qwen3-scaling-test"
MAX_STEPS=500
LOGGING_STEPS=1

# === ENVIRONMENT SETUP ===
export HF_HOME="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
export PYTHONPATH="$(pwd)/dllm:$(pwd):$PYTHONPATH"
export WANDB_PROJECT=$WANDB_PROJECT

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

# Activate environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

echo "Starting Scaling Efficiency Test Suite"
echo "Reservation: biggs.s_test"
echo "Node: $SLURMD_NODENAME"
echo "Project: $WANDB_PROJECT"

# Loop through GPU configurations
for NUM_GPUS in 1 2 4 8; do
    RUN_NAME="qwen3-scaling-${NUM_GPUS}gpu"
    OUTPUT_DIR="${COMMON_OUTPUT_DIR}/${RUN_NAME}"
    
    echo "=================================================="
    echo "Starting test with ${NUM_GPUS} GPUs..."
    echo "Run Name: ${RUN_NAME}"
    echo "=================================================="
    
    # Run training
    # We explicitly set --num_processes to control the number of GPUs used by accelerate
    accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_processes $NUM_GPUS \
        scripts/train_qwen3_mdlm.py \
        --model_name_or_path "$MODEL_NAME" \
        --dataset_args "$DATA_DIR" \
        --load_preprocessed_data True \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to wandb \
        --logging_steps $LOGGING_STEPS \
        --max_steps $MAX_STEPS \
        --save_strategy "no" \
        --eval_strategy "no" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.1 \
        --overwrite_output_dir
        
    echo "Completed test with ${NUM_GPUS} GPUs."
    echo ""
done

echo "All scaling tests completed."
