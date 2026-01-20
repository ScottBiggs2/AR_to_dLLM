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
python scripts/setup_data.py "$@"
