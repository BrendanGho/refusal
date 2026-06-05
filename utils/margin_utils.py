import argparse
import hashlib
from decimal import Decimal
from typing import Dict, List, Optional


def flatten_margin_args(layer_margin_args: Optional[List[str]]) -> List[str]:
    if layer_margin_args is None:
        return []

    parts: List[str] = []
    for chunk in layer_margin_args:
        parts.extend(p.strip() for p in str(chunk).split(",") if p.strip() != "")
    return parts


def parse_layer_margin_values(layer_margin_args: Optional[List[str]]) -> List[float]:
    return [float(part) for part in flatten_margin_args(layer_margin_args)]


def decimal_places(step: float | None) -> int | None:
    if step is None:
        return None
    dec = Decimal(str(step)).normalize()
    return max(0, -dec.as_tuple().exponent)


def normalize_margin(value: float, step: float | None) -> float:
    digits = decimal_places(step)
    if digits is None:
        return float(value)
    return round(float(value), digits)


def build_layer_margin_map(
    selected_layers: List[int],
    default_margin: float,
    layer_margin_args: Optional[List[str]],
) -> Dict[int, float]:
    if layer_margin_args is None:
        return {layer_idx: float(default_margin) for layer_idx in selected_layers}

    margin_values = parse_layer_margin_values(layer_margin_args)
    if len(margin_values) != len(selected_layers):
        raise ValueError(
            f"--layer_margin provided {len(margin_values)} values, but --layers resolved to "
            f"{len(selected_layers)} layers: {selected_layers}"
        )

    return {
        layer_idx: margin_value
        for layer_idx, margin_value in zip(selected_layers, margin_values)
    }


def build_marginvec_payload(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return "|".join(f"{layer_idx}:{layer_margin_map[layer_idx]:.8g}" for layer_idx in selected_layers)


def build_marginvec_digest(
    selected_layers: List[int],
    layer_margin_map: Dict[int, float],
    digest_len: int = 12,
) -> str:
    payload = build_marginvec_payload(selected_layers, layer_margin_map)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:digest_len]


def build_marginvec_tag(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return f"marginvec{build_marginvec_digest(selected_layers, layer_margin_map)}"


def format_layer_margin_cli(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return " ".join(f"{layer_margin_map[layer_idx]:.8g}" for layer_idx in selected_layers)


def _parse_layers_arg(layers_arg: str) -> List[int]:
    if "-" in layers_arg:
        start, end = map(int, layers_arg.split("-"))
        return list(range(start, end))
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the canonical marginvec payload and digest used by pipeline.py. "
            "The SHA1 digest is one-way, so this tool helps you verify candidate per-layer margins."
        )
    )
    parser.add_argument(
        "--layers",
        required=True,
        type=str,
        help="Layer selection in the same format as pipeline.py, e.g. '8,12,14' or '4-7'.",
    )
    parser.add_argument(
        "--layer_margin",
        "--layer_margins",
        dest="layer_margin",
        nargs="+",
        required=True,
        type=str,
        help="Per-layer margins, either space-separated or comma-separated.",
    )
    args = parser.parse_args()

    selected_layers = _parse_layers_arg(args.layers)
    layer_margin_map = build_layer_margin_map(selected_layers, 0.0, args.layer_margin)

    print(f"layers={selected_layers}")
    print(f"layer_margin_map={layer_margin_map}")
    print(f"payload={build_marginvec_payload(selected_layers, layer_margin_map)}")
    print(f"digest={build_marginvec_digest(selected_layers, layer_margin_map)}")
    print(f"tag={build_marginvec_tag(selected_layers, layer_margin_map)}")
    print(f"cli=--layer_margin {format_layer_margin_cli(selected_layers, layer_margin_map)}")


if __name__ == "__main__":
    main()
