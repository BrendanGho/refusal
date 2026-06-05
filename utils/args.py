from typing import Dict, List, Optional

from utils.margin_utils import build_layer_margin_map, build_marginvec_tag
from utils.models_utils import parse_layers


def parse_layers_arg(layers_arg: str, n_layers: int) -> List[int]:
    if isinstance(layers_arg, str) and layers_arg.strip().lower() == "all":
        return list(range(n_layers))

    layers = parse_layers(layers_arg)
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of bounds [0, {n_layers - 1}]")
    return layers


def build_layers_arg(layer_start: int, layer_end: int) -> str:
    if layer_end <= layer_start:
        raise ValueError(f"Invalid layer range: start={layer_start}, end={layer_end}")
    if layer_end == layer_start + 1:
        return str(layer_start)
    return f"{layer_start}-{layer_end}"


def selected_layers_from_arg(layers_arg: str) -> List[int]:
    if "-" in layers_arg:
        start, end = map(int, layers_arg.split("-"))
        return list(range(start, end))
    return [int(x) for x in layers_arg.split(",")]


def parse_layer_margins(
    layer_margins_arg: Optional[List[str]],
    selected_layers: List[int],
    default_margin: float,
) -> Dict[int, float]:
    return build_layer_margin_map(selected_layers, default_margin, layer_margins_arg)


def build_margin_tag(args, selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    if args.layer_margin is None:
        return f"margin{args.margin}"

    return build_marginvec_tag(selected_layers, layer_margin_map)


def build_run_tag(args, selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    if isinstance(args.layers, str) and args.layers.strip().lower() == "all":
        layers_str = "all"
    elif len(selected_layers) == 1:
        layers_str = str(selected_layers[0])
    else:
        layers_str = args.layers.replace(",", "_").replace("-", "to")

    if args.limit:
        limit_str = f"limit{args.limit}"
    else:
        limit_str = "FULL"

    margin_tag = build_margin_tag(args, selected_layers, layer_margin_map)
    probe_tag = "" if args.probe_type == "svm" else f"_probe{args.probe_type}"
    return f"{args.dataset}_{limit_str}_layers{layers_str}{probe_tag}_beta{args.beta}_{margin_tag}_seed{args.seed}"
