import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from utils.args import build_run_tag, parse_layer_margins, parse_layers_arg
from utils.hooks import projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import ProbeDict, load_probes
from utils.runtime import chunked, evaluate, load_model, load_prompts
from utils.runtime import print_configuration, set_seed, validate_probe_dims


def get_args():
    parser = argparse.ArgumentParser(
        description="Sequential multi-layer latent projection applied to every token during prefill and generation."
    )

    # Model config
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Probe / layer config
    parser.add_argument(
        "--svm_dir",
        type=str,
        default=None,
        help="Directory containing latent SVM checkpoints (svm_layerXX.pt).",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="svm",
        choices=["svm", "single_direction"],
        help=(
            "Probe artifact type. 'svm' loads svm_layerXX.pt with learned intercepts; "
            "'single_direction' loads sd_layerXX.pt from classifier/train_latent and "
            "derives the midpoint bias from class means."
        ),
    )
    parser.add_argument(
        "--probe_reps_dir",
        type=str,
        default=None,
        help=(
            "Directory containing HFx_train.pt and HLx_train.pt for deriving "
            "single-direction probe biases. Defaults to --svm_dir when available."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to intervene: 'all', '5-25' (end-exclusive), or '10,14,20'.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Scale in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin m in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--layer_margin",
        "--layer_margins",
        dest="layer_margin",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional per-layer margins aligned with --layers. Accepts either "
            "space-separated values or comma-separated values."
        ),
    )

    # Generation / dataset
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for projection and generation.")

    # Output / eval
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def register_projection_hooks(
    layers,
    selected_layers: List[int],
    probes: ProbeDict,
    layer_margin_map: Dict[int, float],
    args,
):
    handles = []
    for layer_idx in selected_layers:
        handles.append(
            layers[layer_idx].register_forward_hook(
                projection_hook(
                    probes[layer_idx]["w"],
                    probes[layer_idx]["b"],
                    args.beta,
                    layer_margin_map[layer_idx],
                )
            )
        )
    return handles


def generate_with_projection(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    probes: ProbeDict,
    layer_margin_map: Dict[int, float],
    args,
) -> List[str]:
    generation_handles = register_projection_hooks(layers, selected_layers, probes, layer_margin_map, args)
    try:
        return model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"Batch generation error. Falling back to single-prompt generation: {e}")
        remove_hooks(generation_handles)
        generation_handles = []
        responses = []
        for prompt in batch_prompts:
            single_handles = register_projection_hooks(layers, selected_layers, probes, layer_margin_map, args)
            try:
                responses.append(model.generate(prompt, max_new_tokens=args.max_new_tokens))
            except Exception as inner_e:
                print(f"Gen Error: {inner_e}")
                responses.append("")
            finally:
                remove_hooks(single_handles)
        return responses
    finally:
        remove_hooks(generation_handles)


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    hidden_dim = model.model.config.hidden_size

    selected_layers = parse_layers_arg(args.layers, n_layers)
    layer_margin_map = parse_layer_margins(args.layer_margin, selected_layers, args.margin)

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if not os.path.isdir(args.svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {args.svm_dir}")

    probes = load_probes(
        probe_type=args.probe_type,
        svm_dir=args.svm_dir,
        layer_indices=selected_layers,
        device=device,
        explicit_reps_dir=args.probe_reps_dir,
    )
    validate_probe_dims(probes, selected_layers, hidden_dim)

    prompts, categories = load_prompts(args)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    base_out_dir = args.out_dir if args.out_dir is not None else os.path.join("./completions", args.model_name, "projection")
    args.out_dir = base_out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = build_run_tag(args, selected_layers, layer_margin_map)
    output_path = os.path.join(args.out_dir, f"completions_{run_tag}.json")
    print_configuration(
        args=args,
        selected_layers=selected_layers,
        layer_margin_map=layer_margin_map,
        probe_reps_dir=args.probe_reps_dir if args.probe_type == "single_direction" else None,
        output_path=output_path,
        prompt_count=len(prompts),
    )

    results = []
    pbar = tqdm(total=len(prompts), desc="Projection + Generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[batch_start:batch_start + len(batch_prompts)]

        # CLE-P generation: projection hooks stay active during prefill and decoding.
        responses = generate_with_projection(
            batch_prompts=batch_prompts,
            layers=layers,
            selected_layers=selected_layers,
            model=model,
            probes=probes,
            layer_margin_map=layer_margin_map,
            args=args,
        )

        for prompt, response, category in zip(batch_prompts, responses, batch_categories):
            results.append({"category": category, "prompt": prompt, "response": response})
        pbar.update(len(batch_prompts))
    pbar.close()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved completions to {output_path}")

    if args.evaluate:
        del model
        evaluate(
            results=results,
            eval_path=os.path.join(args.out_dir, "evaluation", f"evaluation_{run_tag}.json"),
        )


if __name__ == "__main__":
    main()
