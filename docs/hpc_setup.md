# HPC Setup & Execution Guide (Northeastern Discovery/Explorer)

This guide details how to set up the Qwen3 conversion environment on the Northeastern HPC cluster, specifically targeting A100/H200 GPUs.

## 1. Environment Setup (Scratch Storage)

To avoid quota limits in your home directory, we will create the conda environment in `/scratch` and link it to your project.

### Step 1: Create Environment in Scratch
SSH into the cluster and run:
```bash
# 1. Go to your personal scratch space
# 2. Create a directory for this project's environments
mkdir -p project_envs/qwen3_dllm

# 3. Create the conda environment here
```bash
# 1. Load the Anaconda module provided by the cluster (if needed)
module load anaconda3/2022.05   # Adjust version as appropriate for your system

# 2. Create the conda environment *directly* in scratch (no extra "venv" sub‑folder)
conda create -p /scratch/$USER/project_envs/qwen3_dllm python=3.10 -y

# 3. Link the environment to a convenient ``.venv`` folder in your project repository
cd ~/AR_to_dLLM
ln -sfn /scratch/$USER/project_envs/qwen3_dllm .venv   # ``-sfn`` overwrites any existing link

# 4. Activate the environment
#   • Using conda’s prefix activation [Working]
conda activate /scratch/$USER/project_envs/qwen3_dllm

#   • Or, if you prefer the classic virtual‑env style [Ignore it]
source .venv/bin/activate
```

> **Note**: The ``conda`` command you see (`conda () { … }`) is a shell function provided by the cluster’s Anaconda module. There is no ``$HOME/miniconda3`` installation, so sourcing that path will fail. The ``(base)`` prompt indicates the module’s base environment is already active.

> **Why we drop the ``/venv`` suffix**: When you use ``--prefix /path/to/venv`` conda creates the environment *inside* the ``venv`` directory, making the actual environment root ``/path/to``. Activating ``/path/to/venv`` therefore fails with `EnvironmentLocationNotFound`. Creating the environment with ``-p /path/to`` (or ``--prefix`` without an extra sub‑folder) makes the root the directory you pass, which can be activated directly.

```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
# Manually install flash-attn with CUDA module loaded (required for compilation)
module load cuda/12.4
pip install flash-attn==2.6.3 --no-build-isolation
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
