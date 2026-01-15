# dLLM Sub-Repository (Streamlined)

This directory contains a specialized fork of the `dllm` library, pruned to focus exclusively on the **Qwen3-0.6B AR to dLLM** conversion project.

## Project Scope
This sub-repository provides the core training and sampling logic:
- `dllm/core/`: Specialized `MDLMTrainer` and `BD3LMTrainer` with CoDA masking integration.
- `dllm/pipelines/a2d/`: Qwen3 model architecture adaptations (`modeling_qwen3.py`).
- `examples/a2d/`: Reference SFT scripts for MDLM and BD3LM.

## Cleanup Summary
Unused model architectures (Llama, BERT, LLaDA) and unrelated examples (EditFlow, Dream) have been removed to reduce complexity. For a full list of removals, see **[docs/repository_cleanup.md](../docs/repository_cleanup.md)**.

## Usage
It is recommended to use the top-level scripts in the project root for standard workflows:
- `../scripts/convert_qwen3.sh`
- `../scripts/train_qwen3_mdlm_slurm.sh`

---
For more details, see the **[Main Project README](../readme.md)**.
