#!/bin/bash
# Sanity check for the evaluation environment on the login node.

echo "--- System Check ---"
echo "User: $USER"
echo "Working Dir: $(pwd)"

# 1. Environment Activation
echo -e "\n--- Activating Environment ---"
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null
conda activate /scratch/$USER/project_envs/qwen3_dllm 2>/dev/null

if [ $? -eq 0 ]; then
    echo "SUCCESS: Conda environment activated."
else
    echo "ERROR: Could not activate environment at /scratch/$USER/project_envs/qwen3_dllm"
    exit 1
fi

# 2. PYTHONPATH and DLLM setup
echo -e "\n--- Checking PYTHONPATH and DLLM ---"
export PYTHONPATH="$(pwd)/dllm:$(pwd):$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# 3. Package Verification
echo -e "\n--- Verifying Core Packages ---"
python3 -c "
import torch, transformers, accelerate, lm_eval, evaluate, dllm
print(f'torch:        {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'accelerate:   {accelerate.__version__}')
print(f'lm-eval:      {lm_eval.__version__}')
print(f'evaluate:     {evaluate.__version__}')
print('SUCCESS: All core imports working.')
" || { echo "ERROR: One or more core packages missing or failing to import."; exit 1; }

# 4. Checkpoint Verification
CHECKPOINT_DEFAULT="/scratch/$USER/outputs/qwen3-mdlm-tulu3-sft-v1/checkpoint-6000"
echo -e "\n--- Checking Checkpoint ---"
if [ -d "$CHECKPOINT_DEFAULT" ]; then
    echo "SUCCESS: Found default checkpoint at $CHECKPOINT_DEFAULT"
else
    echo "WARNING: Default checkpoint not found at $CHECKPOINT_DEFAULT"
    echo "Make sure to pass the correct path when running the eval suite."
fi

echo -e "\n--- All checks passed! ---"
echo "You are ready to submit the job with: sbatch scripts/eval_suite_qwen3.sh"
