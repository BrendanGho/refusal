import torch
MODEL_REGISTRY = None


def _get_model_registry():
    global MODEL_REGISTRY
    if MODEL_REGISTRY is None:
        from models import (
            Llama2_7b,
            Llama3_8b,
            Qwen2_32b, Qwen35_9b,
            Mistral7B_RR, Gemma3_12b, Phi35Mini, Llama32_3b, Mistral7b_Instruct, Mixtral8x7b_Instruct, Ministral3_14b, GPT20b, Olmo3_7b, Phi4_15b, DeepSeekR1_8b
        )

        # The "Single Source of Truth"
        MODEL_REGISTRY = {
            "llama2-7b": {
                "hf_id": "meta-llama/Llama-2-7b-chat-hf",
                "class": Llama2_7b
            },
            "llama3-8b": {
                "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "class": Llama3_8b
            },
            "qwen2-32b": {
                "hf_id": "Qwen/Qwen2.5-32B-Instruct",
                "class": Qwen2_32b
            },
            "qwen35-9b": {
                "hf_id": "Qwen/Qwen3.5-9B",
                "class": Qwen35_9b
            },
            "mistral-7b-rr": {
                "hf_id": "GraySwanAI/Mistral-7B-Instruct-RR",
                "class": Mistral7B_RR
            },
            "gemma3-12b": {
                "hf_id": "google/gemma-3-12b-it",
                "class": Gemma3_12b
            }, 
            "phi35mini": {
                "hf_id": "microsoft/Phi-3.5-Mini-instruct", 
                "class": Phi35Mini
            },
            "llama32-3b": {
                "hf_id": "meta-llama/Meta-Llama-3.2-3B-Instruct",
                "class": Llama32_3b
            },
            "mistral-7bv3": {
                "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "class": Mistral7b_Instruct
            },
            "mixtral-8x7b": {
                "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "class": Mixtral8x7b_Instruct
            },
            "ministral3-14b": {
                "hf_id": "mistralai/Ministral-3-14B-Instruct-2512",
                "class": Ministral3_14b
            },
            "gpt-oss-20b": {
                "hf_id": "openai/gpt-oss-20b",
                "class": GPT20b
            },
            "olmo3-7b": {
                "hf_id": "allenai/Olmo-3-7B-Instruct",
                "class": Olmo3_7b
            },
            "phi4-15b": {
                "hf_id": "microsoft/Phi-4-reasoning-vision-15B",
                "class": Phi4_15b
            },
            "deepseek-r1-8b": {
                "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "class": DeepSeekR1_8b
            }
        }
    return MODEL_REGISTRY

def get_model_info(model_name):
    model_registry = _get_model_registry()
    if model_name not in model_registry:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(model_registry.keys())}")
    return model_registry[model_name]


def get_transformer_layers(model):
    """
    Returns the model's transformer blocks through the model-specific implementation.
    """
    if hasattr(model, "_get_transformer_layers") and callable(model._get_transformer_layers):
        return model._get_transformer_layers()

    raise AttributeError(
        f"Model class '{model.__class__.__name__}' does not implement '_get_transformer_layers'."
    )

def get_ablated_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)
    proj = torch.einsum('...d,d->...', matrix, vec)  # shape: [...]
    return matrix - proj.unsqueeze(-1) * vec  # shape: [..., d_model]

def ablate_weights(model, direction: torch.Tensor):
    model.ablate_weights(direction)

def parse_layers(layer_arg):
    """Parses '10-20' into range(10, 20), '10-10' into [10], or '14,15,16' into [14, 15, 16]."""
    if '-' in layer_arg:
        start, end = map(int, layer_arg.split('-'))
        if start == end:
            return [start]
        return list(range(start, end))
    else:
        return [int(x) for x in layer_arg.split(',')]
    
