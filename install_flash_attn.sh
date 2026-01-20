#!/bin/bash
module load cuda/12.4   # Load full CUDA toolkit for compilation
pip install flash-attn==2.6.3 --no-build-isolation
