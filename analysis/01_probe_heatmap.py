"""Sanity check: (layer x turn) refusal heatmaps grouped by jailbreak outcome.

Confirms the collected residual stream is meaningful and that a refusal signature exists
worth chasing — BEFORE any fancier exploration. Reads the probe-scored matrices written by
refusal_matrix.py plus the per-conversation per-turn `jailbroken` labels from the
collector's index.jsonl. Conversations are grouped by the turn at which they FIRST get
jailbroken (t1..tK) or "never", and each group's mean (layer x turn) heatmap is rendered
side by side on a shared, diverging, zero-centered color scale.

No alignment tricks — just subsets. This is the deliberately-simple committed analysis;
PCA / decomposition / head-attribution are left open (designed after seeing this).

Run from repo root:
  .venv/bin/python analysis/01_probe_heatmap.py --refusal_dir <activations_dir>/refusal
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
                   help="Directory with refusal_matrices.safetensors + meta.json (refusal_matrix.py --out_dir).")
    p.add_argument("--index", type=str, default=None,
                   help="index.jsonl from collect_crescendo.py. Default: read activations_dir from meta.json.")
    p.add_argument("--out", type=str, default=None, help="Output PNG. Default: <refusal_dir>/heatmap_by_outcome.png")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def resolve_index_path(refusal_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    meta_path = refusal_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cand = Path(meta.get("activations_dir", "")) / "index.jsonl"
        if cand.exists():
            return cand
    cand = refusal_dir.parent / "index.jsonl"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        "Could not locate index.jsonl; pass --index explicitly."
    )


def load_index(index_path: Path) -> dict[str, dict]:
    index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                index[e["conversation_id"]] = e
    return index


def first_jailbroken_turn(entry: dict) -> int | None:
    """The turn number of the first jailbroken=True turn, or None if never."""
    turns = entry.get("turns") or []
    jb = entry.get("jailbroken") or []
    for i, turn in enumerate(turns):
        if i < len(jb) and jb[i]:
            return int(turn)
    return None


def outcome_key(entry: dict | None) -> str:
    """Group label: 'unlabeled' if no turn carries a label, else t<first-jb> or 'never'.

    A conversation collected without --labeled has jailbroken=[None, ...]; that is NOT the
    same as a labeled conversation that never jailbroke, so it must not be pooled into the
    'never' panel.
    """
    if entry is None:
        return "unlabeled"
    jb = entry.get("jailbroken") or []
    if not jb or all(v is None for v in jb):
        return "unlabeled"
    fjt = first_jailbroken_turn(entry)
    return "never" if fjt is None else f"t{fjt}"


def mean_over_conversations(mats: list[np.ndarray]) -> np.ndarray:
    """Average (L, T) matrices aligned by turn index; conversations may have different T."""
    n_layers = mats[0].shape[0]
    max_t = max(m.shape[1] for m in mats)
    acc = np.zeros((n_layers, max_t), dtype=np.float64)
    cnt = np.zeros((n_layers, max_t), dtype=np.float64)
    for m in mats:
        t = m.shape[1]
        acc[:, :t] += m
        cnt[:, :t] += 1.0
    return acc / np.maximum(cnt, 1.0)


def main():
    args = get_args()
    refusal_dir = Path(args.refusal_dir)
    matrices = load_file(str(refusal_dir / "refusal_matrices.safetensors"))
    if not matrices:
        raise SystemExit(
            f"No refusal matrices found in {refusal_dir / 'refusal_matrices.safetensors'}. "
            "Run refusal_matrix.py over a non-empty collection first."
        )
    index = load_index(resolve_index_path(refusal_dir, args.index))

    # Group conversations by first-jailbroken turn (or "never" / "unlabeled").
    groups: dict[str, list[np.ndarray]] = {}
    for conv_id, mat in matrices.items():
        groups.setdefault(outcome_key(index.get(conv_id)), []).append(mat)

    # Stable, readable panel order: t1, t2, ... then never, then unlabeled.
    def sort_key(k: str):
        if k.startswith("t") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, {"never": 0, "unlabeled": 1}.get(k, 2))

    keys = sorted(groups, key=sort_key)
    means = {k: mean_over_conversations(groups[k]) for k in keys}

    vmax = max((float(np.abs(m).max()) for m in means.values()), default=1.0) or 1.0
    n_panels = len(keys)
    n_layers = next(iter(means.values())).shape[0]

    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(3, n_panels * 3.2), max(5, n_layers * 0.22)),
        squeeze=False, sharey=True,
    )
    im = None
    for ax, key in zip(axes[0], keys):
        m = means[key]
        n_turns = m.shape[1]
        im = ax.imshow(m, aspect="auto", origin="lower", cmap=args.cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{key}\n(n={len(groups[key])})")
        ax.set_xlabel("Crescendo turn")
        ax.set_xticks(range(n_turns))
        ax.set_xticklabels(range(1, n_turns + 1))
    axes[0][0].set_ylabel("Layer")

    fig.suptitle("Mean refusal phi (layer x turn), grouped by first-jailbroken turn")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
    cbar.set_label("phi  (+refusal / -compliance)")

    out_path = Path(args.out) if args.out else refusal_dir / "heatmap_by_outcome.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Groups: " + ", ".join(f"{k}={len(groups[k])}" for k in keys))
    print(f"Saved heatmap -> {out_path}")


if __name__ == "__main__":
    main()
