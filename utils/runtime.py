import gc
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset.load_dataset import load_dataset
from utils.eval_jailbreaks import evaluate_jailbreak
from utils.models_utils import get_model_info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(args):
    model_info = get_model_info(args.model_name)
    model_kwargs = {"device": args.device}
    return model_info["class"](**model_kwargs)


def load_prompts(args) -> tuple[List[str], List[str]]:
    try:
        dataset_items = load_dataset(args.dataset, instructions_only=False)
    except Exception as e:
        raise ValueError(f"Dataset loading failed for '{args.dataset}': {e}")

    if args.limit:
        dataset_items = dataset_items[: args.limit]
    prompts = [item["instruction"] if isinstance(item, dict) else item for item in dataset_items]
    categories = [item.get("category", "harmful") if isinstance(item, dict) else "harmful" for item in dataset_items]
    return prompts, categories


def validate_probe_dims(probes: Dict[int, Dict[str, torch.Tensor]], layer_indices: List[int], hidden_dim: int) -> None:
    for layer_idx in layer_indices:
        if probes[layer_idx]["w"].numel() != hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch in layer {layer_idx}: got {probes[layer_idx]['w'].numel()}, expected {hidden_dim}"
            )


def chunked(seq: List[str], chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield i, seq[i:i + chunk_size]


def cuda_visible_devices_for_device(device: str) -> str | None:
    device = device.strip().lower()
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return suffix
    return None


def evaluate(*, results: List[Dict[str, object]], eval_path: str) -> None:
    print("\n--- Starting Evaluation ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    evaluate_jailbreak(
        completions=results,
        methodologies=["harmbench"],
        evaluation_path=eval_path,
    )


def print_configuration(
    *,
    args,
    selected_layers: List[int],
    layer_margin_map: Dict[int, float],
    probe_reps_dir: Optional[str],
    output_path: str,
    prompt_count: int,
) -> None:
    print("--- Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Probe type: {args.probe_type}")
    print(f"SVM dir: {args.svm_dir}")
    if probe_reps_dir is not None:
        print(f"Probe reps dir: {probe_reps_dir}")
    print(f"Layers: {selected_layers}")
    print(f"Beta: {args.beta}")
    print(f"Margin: {args.margin}")
    if args.layer_margin is not None:
        print(f"Layer margins: {layer_margin_map}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset} | Prompts: {prompt_count}")
    print(f"Output: {output_path}")
