#!/bin/bash
module load cuda/12.3.0   # Load available CUDA toolkit

# 1. Downgrade PyTorch to match typical HPC CUDA drivers (12.1/12.3) instead of 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install simpler deps
pip install -r requirements.txt

# 3. Install compatible flash-attn
pip install flash-attn==2.6.3 --no-build-isolation
