# HPC Setup & Execution Guide (Northeastern Discovery/Explorer)

This guide details how to set up the Qwen3 conversion environment on the Northeastern HPC cluster, specifically targeting A100/H200 GPUs.

## 1. Environment Setup (Scratch Storage)

To avoid quota limits in your home directory, we will create the conda environment in `/scratch` and link it to your project.

### Step 1: Create Environment in Scratch
SSH into the cluster and run:
```bash
# 1. Go to your personal scratch space
cd /scratch/$USER

# 2. Create a directory for this project's environments
mkdir -p project_envs/qwen3_dllm

# 3. Create the conda environment here
module load anaconda3/2022.05  # Load typical anaconda module
conda create --prefix /scratch/$USER/project_envs/qwen3_dllm/venv python=3.10 -y
```

### Step 2: Link to Project Directory
Navigate to your project code repository (e.g., in `~/AR_to_dLLM`):
```bash
cd ~/AR_to_dLLM

# Link the scratch environment to a local .venv folder
ln -s /scratch/$USER/project_envs/qwen3_dllm/venv .venv

# Activate
source activate ./.venv 
# OR
conda activate /scratch/$USER/project_envs/qwen3_dllm/venv
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Job Submission (Slurm)

We have configured `scripts/train_qwen3_mdlm_slurm.sh` for reliable multinode training.

### Partition & GPU Selection

| Resource | Partition | Flag Example | Notes |
|----------|-----------|--------------|-------|
| 8x A100  | `multigpu`| `--gres=gpu:a100:8` | **Default**. Requires approval for >4 GPUs. |
| 4x A100  | `gpu`     | `--gres=gpu:a100:4` | Standard access. |
| 8x H200  | `multigpu`| `--gres=gpu:h200:8` | Requires access. Update script to uncomment H200 line. |

### Redundancy & Time Limits
*   **Max Walltime**: The script is set to **8 hours** (`#SBATCH --time=08:00:00`).
*   **Automatic Checkpointing**: A callback automatically triggers a checkpoint at **7h 45m** of runtime to ensure progress is saved before the job is killed.

### Launching a Job
```bash
sbatch scripts/train_qwen3_mdlm_slurm.sh
```

### Monitoring
*   **Logs**: Check `logs/qwen3-mdlm-<jobid>.out` and `logs/qwen3-mdlm-<jobid>.err`.
*   **W&B**: Metrics and generations are logged to your Weights & Biases project.

## 3. Testing & Debugging (Single H200)

Before launching full training, verifying the setup on a single H200 node is recommended. We have provided a debug script for this purpose.

```bash
# Runs for 10 steps, no checkpointing by default
sbatch scripts/debug_qwen3_h200.sh
```

**Customizing the Debug Run:**
You can override the defaults by exporting environment variables before sbatch:
```bash
# Example: Run for 50 steps and enable checkpointing
export NUM_STEPS=50
export SAVE_CHECKPOINT=true
sbatch scripts/debug_qwen3_h200.sh
```
