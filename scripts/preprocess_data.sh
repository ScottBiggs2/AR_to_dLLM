python dllm/dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/qwen3-0.6b/tulu-3" \
    --num_proc 64
