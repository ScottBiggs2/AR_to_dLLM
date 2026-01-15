# Project Overview: Qwen3 0.6B AR to dLLM

This project explores the conversion of the **Qwen3-0.6B-Base** model from a Causal Auto-Regressive (AR) model into a Diffusion Language Model (dLLM). 

## Core Objectives

1. **Recalibration**: Initializing diffusion models with pre-trained AR weights.
2. **MDLM/BD3LM**: Implementing Masked Diffusion and Block Diffusion objectives.
3. **CoDA Masking**: Applying advanced masking strategies (Unmaskable Prefix, Truncated Suffix, Block Masking) with progressive curriculum.
4. **DrOPE**: Implementing position-agnostic diffusion by removing RoPE during recalibration.

## Project Structure

- `dllm/`: A streamlined fork of the dLLM repository focused on Qwen3.
- `scripts/`: Production-ready scripts for conversion, preprocessing, and training.
- `docs/`: Technical documentation and implementation details.
- `assets/`: Project assets including banners and diagrams.

## Key Technical Features

### Advanced Masking (CoDA Style)
We use a `MaskingCurriculum` to gradually shift from random masking to structured, task-aware masking patterns:
- **S1 (Unmaskable Prefix)**: Protections instruction/prompt tokens.
- **S2 (Truncated Suffix)**: Simulates variable-length generation.
- **S3 (Block Masking)**: contiguous spans for better infilling performance.

### Positional Embedding Recalibration (DrOPE)
The model supports disabling Rotary Positional Embeddings (RoPE) via a `use_rope` flag, allowing the diffusion process to learn position-independent semantic relationships.

---

For setup and execution details, see the [Main README](../readme.md).
