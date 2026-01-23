#!/bin/bash
#SBATCH --job-name=eval-suite-qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval-suite-%j.out
#SBATCH --error=logs/eval-suite-%j.err

# Defaults
CHECKPOINT_PATH=${1:-"/scratch/biggs.s/outputs/qwen3-mdlm-tulu3-sft-v1/checkpoint-6500"}
TASKS=${2:-"gsm8k,minerva_math,bbh_cot_fewshot,mmlu_pro"}
LIMIT=${3:--1}  # -1 for full evaluation
STEPS=${4:-32}  # Diffusion steps
BLOCK_SIZE=${5:-128}
BATCH_SIZE=${6:-32} # Match MC_NUM for safer likelihood estimation
MC_NUM=${7:-32}     # 32 is a good balance for speed

echo "============================================================"
echo "Starting Evaluation Suite for Qwen3-MDLM..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Steps: $STEPS"
echo "Block Size: $BLOCK_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "MC Samples: $MC_NUM"
echo "============================================================"

# Environment Setup
export HF_HOME="/scratch/$USER/hf_cache"
export PYTHONPATH=".:./dllm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Load modules and activate environment
# Adjust these paths as per your cluster setup
module load cuda/12.3.0 2>/dev/null || echo "Warning: Could not load cuda module"
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null
conda activate /scratch/$USER/project_envs/qwen3_dllm 2>/dev/null || echo "Warning: Could not activate conda environment"

# Results Directory
RESULTS_DIR="eval_results/$(basename $CHECKPOINT_PATH)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Handle optional limit
LIMIT_ARG=""
if [ "$LIMIT" != "-1" ] && [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Launch evaluation per task for incremental saving
# We split the TASKS string by commas and loop
IFS=',' read -ra ADDR <<< "$TASKS"
for task in "${ADDR[@]}"; do
    echo "------------------------------------------------------------"
    echo "Evaluating Task: $task"
    echo "------------------------------------------------------------"
    
    TASK_DIR="$RESULTS_DIR/$task"
    mkdir -p "$TASK_DIR"

    accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        dllm/dllm/pipelines/a2d/eval.py \
        --model mdlm \
        --tasks "$task" \
        $LIMIT_ARG \
        --batch_size "$BATCH_SIZE" \
        --model_args "pretrained=$CHECKPOINT_PATH,max_new_tokens=512,steps=$STEPS,block_size=$BLOCK_SIZE,mc_num=$MC_NUM" \
        --apply_chat_template \
        --output_path "$TASK_DIR"
        
    echo "Finished $task. Partial results saved to $TASK_DIR"
done

echo ""
echo "============================================================"
echo "Evaluation Finished."
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"

# Optional: Generate a quick summary text file
echo "Creating summary.txt..."
python3 -c "
import json, os, glob
try:
    with open('$RESULTS_DIR/summary.txt', 'w') as summary_f:
        summary_f.write('Evaluation Summary\n')
        summary_f.write('==================\n')
        summary_f.write('Checkpoint: $CHECKPOINT_PATH\n\n')
        summary_f.write('{:<25} {:<10}\n'.format('Task', 'Score'))
        summary_f.write('-' * 35 + '\n')
        
        # Look for results.json in all subdirectories (incremental tasks)
        for result_file in sorted(glob.glob('$RESULTS_DIR/*/results.json')):
            with open(result_file, 'r') as f:
                data = json.load(f)
                results = data.get('results', {})
                for task, metrics in results.items():
                    score = metrics.get('acc,none') or metrics.get('em,none') or metrics.get('acc_norm,none') or 'N/A'
                    summary_f.write('{:<25} {:<10}\n'.format(task, str(score)))
                    
    print('Summary created at $RESULTS_DIR/summary.txt')
except Exception as e:
    print(f'Could not create summary: {e}')
"
