#!/bin/bash
module load cuda/12.3.0   # Load available CUDA toolkit

echo "Cleaning up conflicting libraries..."
pip uninstall -y flash-attn torch torchvision torchaudio
# Nuke all nvidia pip packages to ensure no 12.4 leftovers cause linker errors
pip freeze | grep "nvidia-" | xargs pip uninstall -y

echo "Installing stable PyTorch..."
# 1. Downgrade PyTorch to match typical HPC CUDA drivers 
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing dependencies..."
# 2. Install simpler deps
pip install -r requirements.txt

echo "Installing Flash Attention..."
# 3. Install compatible flash-attn
pip install flash-attn==2.6.3 --no-build-isolation
