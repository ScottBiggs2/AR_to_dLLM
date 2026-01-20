
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

import datasets
import transformers
from huggingface_hub import login

# Add repo root to path to allow importing dllm
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# from dllm.dllm.utils.data import default_sft_map_fn

def default_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.

    Args:
        row: a dataset row with `messages`
        tokenizer: a HF tokenizer
        mask_prompt_loss: whether to mask prompt tokens (set their labels to -100)

    Returns:
        dict with keys: input_ids, labels, and optionally prompt_len
    """
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}

load_dotenv(REPO_ROOT / ".env")

def setup_data(model_path: str, output_dir: str, dataset_name: str = "allenai/tulu-3-sft-mixture"):
    print(f"Loading environment...")
    token = os.getenv("HF_TOKEN")
    if token and token.startswith("hf_"):
        print(f"Logging in to Hugging Face...")
        login(token=token)
    else:
        print("Warning: No valid HF_TOKEN found in .env. Ensure public access or add token.")

    print(f"Loading dataset: {dataset_name}")
    try:
        ds = datasets.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Tulu-3 specific splitting as per dllm logic
    if "allenai/tulu-3-sft-mixture" in dataset_name:
        if "train" in ds and "test" not in ds:
            print("Splitting dataset (train/test)...")
            ds = ds["train"].train_test_split(test_size=0.05, seed=42)
    
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        print(f"Could not load tokenizer from {model_path}. Trying fallback 'Qwen/Qwen2.5-0.5B-Instruct'...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Mapping dataset to SFT format (tokenizing)...")
    # Apply default_sft_map_fn
    # Note: default_sft_map_fn requires 'mask_prompt_loss' kwarg.
    # We'll use the default True as typically desired for SFT.
    def map_wrapper(row):
        return default_sft_map_fn(row, tokenizer=tokenizer, mask_prompt_loss=True)

    processed_ds = ds.map(
        map_wrapper,
        num_proc=os.cpu_count(),
        desc="Tokenizing",
        remove_columns=ds["train"].column_names # Remove raw text columns to save space/avoid errors
    )

    print(f"Saving processed dataset to: {output_dir}")
    processed_ds.save_to_disk(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/a2d/Qwen3-0.6B", help="Path to model for tokenizer")
    parser.add_argument("--output", type=str, default="data/sft/qwen3-0.6b/tulu-3", help="Output directory")
    args = parser.parse_args()

    setup_data(args.model, args.output)
