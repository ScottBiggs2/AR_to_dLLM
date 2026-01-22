#!/bin/bash
#SBATCH --job-name=eval-suite-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-suite-%j.out
#SBATCH --error=logs/eval-suite-%j.err

# Defaults
CHECKPOINT_PATH=${1:-"/scratch/biggs.s/outputs/qwen3-mdlm-tulu3-sft-v1/checkpoint-6500"}
TASKS=${2:-"gsm8k,minerva_math,bbh_cot_fewshot,mmlu_pro"}
LIMIT=${3:--1}  # -1 for full evaluation
STEPS=${4:-32}  # Diffusion steps
BLOCK_SIZE=${5:-128}

echo "============================================================"
echo "Starting Evaluation Suite for Qwen3-MDLM..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Steps: $STEPS"
echo "Block Size: $BLOCK_SIZE"
echo "============================================================"

# Environment Setup
export HF_HOME="/scratch/$USER/hf_cache"
export PYTHONPATH=".:./dllm:$PYTHONPATH"

# Load modules and activate environment
# Adjust these paths as per your cluster setup
module load cuda/12.3.0 2>/dev/null || echo "Warning: Could not load cuda module"
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null
conda activate /scratch/$USER/project_envs/qwen3_dllm 2>/dev/null || echo "Warning: Could not activate conda environment"

# Results Directory
RESULTS_DIR="eval_results/$(basename $CHECKPOINT_PATH)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Launch evaluation
# We use --output_path to save JSON results
# lm-eval will create a directory at output_path if it doesn't exist
accelerate launch \
    --num_processes 1 \
    dllm/dllm/pipelines/a2d/eval.py \
    --model mdlm \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --model_args "pretrained=$CHECKPOINT_PATH,max_new_tokens=512,steps=$STEPS,block_size=$BLOCK_SIZE" \
    --apply_chat_template \
    --output_path "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "Evaluation Finished."
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"

# Optional: Generate a quick summary text file
if [ -f "$RESULTS_DIR/results.json" ]; then
    echo "Creating summary.txt..."
    python3 -c "
import json, os
try:
    with open('$RESULTS_DIR/results.json', 'r') as f:
        data = json.load(f)
    with open('$RESULTS_DIR/summary.txt', 'w') as f:
        f.write('Evaluation Summary\n')
        f.write('==================\n')
        f.write(f'Checkpoint: $CHECKPOINT_PATH\n')
        f.write(f'Date: ' + data.get('date', 'N/A') + '\n\n')
        f.write('{:<20} {:<10}\n'.format('Task', 'Score'))
        f.write('-' * 30 + '\n')
        results = data.get('results', {})
        for task, metrics in results.items():
            # Try to find a primary metric (accuracy, exact_match, etc.)
            score = metrics.get('acc,none') or metrics.get('em,none') or metrics.get('acc_norm,none') or 'N/A'
            f.write('{:<20} {:<10}\n'.format(task, str(score)))
    print('Summary created at $RESULTS_DIR/summary.txt')
except Exception as e:
    print(f'Could not create summary: {e}')
"
fi
