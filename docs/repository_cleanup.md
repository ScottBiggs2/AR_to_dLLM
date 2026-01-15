# Repository Cleanup: Qwen3-Specific Focus

To streamline the development environment and focus exclusively on the Qwen3 conversion project, the following components were removed from the merged `dllm` repository.

## Removed Components

### 1. Unused Model Pipelines (`dllm/dllm/pipelines/`)
Removed implementations for architecture types that are not relevant to the Qwen3-0.6B conversion:
- `bert/`: Masked language model foundations (not used for this AR conversion).
- `dream/`: Specific LLaMa-based diffusion variant.
- `editflow/`: Discrete flow matching implementation.
- `llada/` & `llada2/`: Specific LLaDA model pipelines.

### 2. Unused Examples (`dllm/examples/`)
Removed training and inference examples for the above models:
- `bert/`
- `dream/`
- `editflow/`
- `llada/`
- `llada2/`

### 3. Non-Target A2D Models (`dllm/dllm/pipelines/a2d/models/`)
Focused the A2D (AR to Diffusion) pipeline by removing:
- `llama/`: Llama-specific bidirectional adaptations.
- `qwen2/`: Previous generation Qwen adaptations (keeping `qwen3/`).

## Rationale
- **Clarity**: Reduces cognitive load when navigating the codebase.
- **Maintenance**: Fewer files to update/keep compatible as we modify core trainers.
- **Build Speed**: Faster editable installation and lighter project footprint.
- **Focus**: Aligns the repository state with the "Qwen3 0.6B AR to dLLM" project objective.
