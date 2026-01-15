![Project Banner](assets/banner.png)

# Qwen3 0.6B AR to dLLM Conversion

A research project focused on converting the Qwen3-0.6B Auto-Regressive model into a high-performance Diffusion Language Model (dLLM) using recalibration techniques, advanced masking, and position-agnostic sampling.

## 🚀 Quick Start

### 1. Environment Setup
Create a conda environment and install dependencies:
```bash
conda create -n qwen3-dllm python=3.10
conda activate qwen3-dllm
pip install -r requirements.txt
```

### 2. Model Weight Initialization
Convert the base AR weights into the A2D-bidirectional format:
```bash
bash scripts/convert_qwen3.sh
```

### 3. Data Preprocessing
Pre-tokenize the training dataset (e.g., Tulu-3):
```bash
bash scripts/preprocess_data.sh
```

### 4. Launch Training
Start the baseline MDLM training (multinode Slurm example):
```bash
sbatch scripts/train_qwen3_mdlm_slurm.sh
```

## 🛠 Project Components

- **[Overview](docs/overview.md)**: Project goals and technical background.
- **[Repository Cleanup](docs/repository_cleanup.md)**: Details on the streamlined `dllm` sub-repo.
- **[Conversion Pipeline](scripts/convert_qwen3.sh)**: Logic for AR weight transfer.
- **[Advanced Masking](dllm/dllm/core/masking.py)**: CoDA-style curriculum masking.

## 📊 Training & Monitoring

Detailed metrics (NLL, PPL) and sample generations are logged to **Weights & Biases**. 
Ensure your `WANDB_API_KEY` is set in your environment or `.env` file.

- **Checkpointing**: Saved every 10% of training by default.
- **Evaluation**: Generations are logged to W&B tables for visual audit of progress.

## 📜 Acknowledgments

Based on the `ZHZisZZ/dLLM` codebase, with enhancements inspired by:
- **Salesforce CoDA** (Advanced Masking)
- **Apple DiffuCoder** (Block Diffusion)
- **DrOPE** (Recalibration techniques)
