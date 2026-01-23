import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.configs import ModelArguments, TrainingArguments
from dllm.utils.utils import disable_caching_allocator_warmup, load_peft, print_main


def get_model(
    model_args,
    config: transformers.PretrainedConfig | None = None,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: An optional dataclass or namespace containing model parameters.
        model_name_or_path: Optional direct model path or name (overrides model_args.model_name_or_path).
        dtype: Dtype (string or torch.dtype).
        load_in_4bit: Whether to load using 4-bit quantization (can override model_args.load_in_4bit).

    Returns:
        transformers.PreTrainedModel
    """
    model_name_or_path = getattr(model_args, "model_name_or_path")
    dtype = getattr(model_args, "dtype", "bfloat16")
    load_in_4bit = getattr(model_args, "load_in_4bit", False)
    attn_implementation = getattr(model_args, "attn_implementation", None)

    # Device map: skip when ZeRO-3
    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Convert string dtype to torch.dtype
    if isinstance(dtype, str) and dtype != "auto":
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
    else:
        torch_dtype = dtype

    params = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": attn_implementation,
        "config": config,
    }

    if config is None:
        try:
            config = transformers.AutoConfig.from_pretrained(model_name_or_path, **params)
            
            # For Qwen3/Qwen2 models, we might need to cast to A2D versions for MaskedLM support
            if type(config) == transformers.Qwen2Config:
                 from dllm.pipelines.a2d.models.qwen2.modeling_qwen2 import A2DQwen2Config
                 config = A2DQwen2Config(**config.to_dict())
            elif type(config) == transformers.Qwen3Config:
                 from dllm.pipelines.a2d.models.qwen3.modeling_qwen3 import A2DQwen3Config
                 config = A2DQwen3Config(**config.to_dict())
            elif type(config) == transformers.LlamaConfig:
                 from dllm.pipelines.a2d.models.llama.modeling_llama import A2DLlamaConfig
                 config = A2DLlamaConfig(**config.to_dict())

            params["config"] = config
        except Exception:
            pass
        
    # Apply DrOPE if specified in model_args
    if config is not None and hasattr(model_args, "use_rope") and hasattr(config, "use_rope"):
        config.use_rope = model_args.use_rope
            
    model = None
    errors = []
    # Strategy: try ForMaskedLM -> ForCausalLM -> AutoModel
    for model_class in [
        transformers.AutoModelForMaskedLM,
        transformers.AutoModelForCausalLM,
        transformers.AutoModel,
    ]:
        try:
            model = model_class.from_pretrained(model_name_or_path, **params)
            break
        except Exception as e:
            errors.append(f"{model_class.__name__}: {str(e)}")
            continue
            
    if model is None:
        error_msg = "\n".join(errors)
        raise ValueError(f"Could not load model from {model_name_or_path}. Underlying errors:\n{error_msg}")

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Optionally train with lora
    model = load_peft(model, model_args)

    return model


def get_tokenizer(model_args) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Optional dataclass or namespace containing model parameters.
        model: Optional model instance to configure tokenizer behavior.
        model_name_or_path: Optional direct model name or path (overrides model_args.model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from transformers import (
        BertPreTrainedModel,
        ModernBertPreTrainedModel,
        RobertaPreTrainedModel,
    )

    # Safe imports for pipelines that might be missing in some environments
    try:
        from dllm.pipelines.a2d import (
            A2DLlamaLMHeadModel,
            A2DQwen2LMHeadModel,
            A2DQwen3LMHeadModel,
        )
    except ImportError:
        # If the models are missing from the __init__.py exposure
        try: from dllm.pipelines.a2d.models.qwen3.modeling_qwen3 import A2DQwen3LMHeadModel
        except ImportError: A2DQwen3LMHeadModel = type(None)
        
        try: from dllm.pipelines.a2d.models.llama.modeling_llama import A2DLlamaLMHeadModel
        except ImportError: A2DLlamaLMHeadModel = type(None)
        
        try: from dllm.pipelines.a2d.models.qwen2.modeling_qwen2 import A2DQwen2LMHeadModel
        except ImportError: A2DQwen2LMHeadModel = type(None)

    try: from dllm.pipelines.dream.models.modeling_dream import DreamModel
    except ImportError: DreamModel = type(None)
    
    try: from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM
    except ImportError: LLaDA2MoeModelLM = type(None)
    
    try: from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    except ImportError: LLaDAModelLM = type(None)
    
    try: from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM
    except ImportError: LLaDAMoEModelLM = type(None)

    model_name_or_path = getattr(model_args, "model_name_or_path")

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )

    assert tokenizer.eos_token is not None or tokenizer.pad_token is not None

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    # ---------------- Model-specific customization ----------------
    # Use config's model_type or class name to determine customization
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(model_cfg, "model_type", "")
    model_cls_name = type(model_cfg).__name__

    if "LLaDA" in model_cls_name or "llada" in model_type:
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        # fix bugs in chat template
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif "Dream" in model_cls_name or "dream" in model_type:
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif any(x in model_cls_name for x in ["Bert", "Roberta"]):
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    elif any(x in model_cls_name for x in ["Qwen3", "Qwen2", "Llama", "A2D"]):
        # Universal A2D/MDLM mask token
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>" if "Qwen" in model_cls_name else "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
        # Inject ChatML template if missing or generic
        if not tokenizer.chat_template or "start_header_id" not in tokenizer.chat_template:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
                "{% endif %}"
            )
    else:
        print_main(f"no specific tokenizer customization for model_type '{model_type}' or class '{model_cls_name}'")

    # Global fallback: ensure mask_token is always present for diffusion models
    if tokenizer.mask_token is None:
        # Avoid overriding if already set by customization above
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        
    return tokenizer
