#!/bin/bash
module load cuda/12.3.0   # Load available CUDA toolkit
pip install flash-attn==2.6.3 --no-build-isolation
