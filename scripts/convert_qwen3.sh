#!/bin/bash

# Exit on error
set -e

# Define directories
SOURCE_MODEL="Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR="models/a2d/Qwen3-0.6B"

echo "Starting conversion of $SOURCE_MODEL to A2D format..."

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the conversion script
# We use the python interpeter from the environment
python dllm/dllm/pipelines/a2d/convert.py \
    --model_name_or_path "$SOURCE_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo "Conversion complete. Model saved in $OUTPUT_DIR"

# Verification
echo "Verifying converted model config..."
grep -q "a2d-qwen3" "$OUTPUT_DIR/config.json" && echo "Success: model_type is a2d-qwen3" || echo "Error: model_type mismatch"
