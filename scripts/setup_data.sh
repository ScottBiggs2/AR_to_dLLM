#!/bin/bash
# Helper script to setup data for training/debugging
# Usage: ./scripts/setup_data.sh [--model path/to/model] [--output path/to/output]

# Ensure we are in the repo root
cd "$(dirname "$0")/.."

# Add the dllm subdirectory to PYTHONPATH to allow importing 'dllm' if not installed
export PYTHONPATH=$PYTHONPATH:$(pwd)/dllm

# Redirect Hugging Face cache to scratch to avoid home directory quotas
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$HF_HOME"

echo "Running data setup..."
# Default output to scratch if no arguments provided
if [ $# -eq 0 ]; then
    python scripts/setup_data.py --output "/scratch/$USER/data/sft/qwen3-0.6b/tulu-3"
else
    python scripts/setup_data.py "$@"
fi
