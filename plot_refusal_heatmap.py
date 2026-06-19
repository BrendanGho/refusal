"""Render (layer x turn) refusal-confidence heatmaps from refusal_matrix.py output.

Reads refusal_matrices.safetensors (key=<conversation_id> -> (num_layers, num_turns)
float32 of phi_l = w_l . z + b_l). Positive phi => refusal, negative => compliance.

Modes:
  --mean (default)        average across all conversations, aligned by turn index
                          -> the refusal "fingerprint": how refusal decays across
                          Crescendo turns at each layer.
  --conversation_id ID    single conversation's matrix.

Color is a diverging map centered at phi=0 (refusal vs compliance), symmetric range.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.numpy import load_file


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refusal_dir", type=str, required=True,
                   help="Directory containing refusal_matrices.safetensors (refusal_matrix.py --out_dir).")
    p.add_argument("--conversation_id", type=str, default=None,
                   help="Plot a single conversation. Default: mean across all conversations.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path. Default: <refusal_dir>/heatmap[...].png")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def mean_over_conversations(matrices: dict[str, np.ndarray]) -> np.ndarray:
    """Average (L, T) matrices aligned by turn index; conversations may have different T."""
    n_layers = next(iter(matrices.values())).shape[0]
    max_t = max(m.shape[1] for m in matrices.values())
    acc = np.zeros((n_layers, max_t), dtype=np.float64)
    cnt = np.zeros((n_layers, max_t), dtype=np.float64)
    for m in matrices.values():
        t = m.shape[1]
        acc[:, :t] += m
        cnt[:, :t] += 1.0
    return acc / np.maximum(cnt, 1.0)


def plot(matrix: np.ndarray, title: str, out_path: Path, cmap: str, dpi: int):
    n_layers, n_turns = matrix.shape
    vmax = float(np.abs(matrix).max()) or 1.0

    fig, ax = plt.subplots(figsize=(max(4, n_turns * 0.9), max(5, n_layers * 0.22)))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Crescendo turn")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_turns))
    ax.set_xticklabels(range(1, n_turns + 1))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("refusal confidence  phi  (+refusal / -compliance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved heatmap -> {out_path}")


def main():
    args = get_args()
    refusal_dir = Path(args.refusal_dir)
    matrices = load_file(str(refusal_dir / "refusal_matrices.safetensors"))

    if args.conversation_id is not None:
        if args.conversation_id not in matrices:
            raise KeyError(f"conversation_id {args.conversation_id!r} not in refusal_matrices.safetensors")
        matrix = matrices[args.conversation_id]
        title = f"Refusal (layer x turn) — {args.conversation_id}"
        default_name = f"heatmap_{args.conversation_id}.png"
    else:
        matrix = mean_over_conversations(matrices)
        title = f"Mean refusal (layer x turn) over {len(matrices)} conversations"
        default_name = "heatmap_mean.png"

    out_path = Path(args.out) if args.out else refusal_dir / default_name
    plot(matrix, title, out_path, args.cmap, args.dpi)


if __name__ == "__main__":
    main()
