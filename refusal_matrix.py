"""Score collected CRESCENDO activations with per-layer SVM refusal probes.

This is a sanity/analysis tool over the exploratory collection from collect_crescendo.py:
it reads only the `resid` tensor (num_turns, num_layers, hidden_dim) from each
conversation's safetensors and applies every layer's probe:

    phi_l(z) = w_l . z + b_l        (positive => refusal, negative => compliance)

producing a (num_layers x num_turns) refusal-confidence matrix per conversation -- the
"refusal fingerprint" tracked across the Crescendo turns.

Probes are the LinearSVC checkpoints written by classifier/train_latent.py
(dataset/representations/<model_name>/train_svm/svm_layerXX.pt).

Outputs (under --out_dir):
  refusal_matrices.safetensors   key=<conversation_id> -> (num_layers, num_turns) float32
  refusal_long.csv               tidy rows: conversation_id, objective_index, sample_index,
                                  turn, layer, phi, jailbroken
  meta.json                      layer indices, model, source dirs
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from utils.probes import discover_available_layers, load_probes


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", type=str, default="llama32-3b")
    p.add_argument("--activations_dir", type=str, required=True,
                   help="Directory written by collect_crescendo.py (contains index.jsonl and acts/).")
    p.add_argument("--svm_dir", type=str, default=None,
                   help="Probe directory. Default: dataset/representations/<model_name>/train_svm")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write outputs. Default: <activations_dir>/refusal")
    p.add_argument("--layers", type=str, default="all",
                   help="'all' or comma-separated layer indices (e.g. '10,14,20').")
    return p.parse_args()


def select_layers(arg: str, available: list[int]) -> list[int]:
    if arg == "all":
        return available
    wanted = [int(x) for x in arg.split(",")]
    missing = [l for l in wanted if l not in available]
    if missing:
        raise ValueError(f"Requested layers not available as probes: {missing}")
    return wanted


def main():
    args = get_args()

    svm_dir = args.svm_dir or os.path.join("dataset", "representations", args.model_name, "train_svm")
    if not os.path.isdir(svm_dir):
        raise FileNotFoundError(
            f"SVM directory not found: {svm_dir}. Train probes first with "
            f"`classifier/train_latent.py --model_name {args.model_name}`."
        )

    acts_dir = Path(args.activations_dir)
    index_path = acts_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"index.jsonl not found in {acts_dir}; run collect_crescendo.py first.")

    out_dir = Path(args.out_dir) if args.out_dir else acts_dir / "refusal"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    layer_indices = select_layers(args.layers, discover_available_layers(svm_dir, "svm"))
    probes = load_probes(probe_type="svm", svm_dir=svm_dir, layer_indices=layer_indices, device=device)

    W = torch.stack([probes[l]["w"].float() for l in layer_indices])           # (L, D)
    b = torch.stack([probes[l]["b"].float().view(()) for l in layer_indices])   # (L,)

    with open(index_path, "r", encoding="utf-8") as f:
        index = [json.loads(line) for line in f if line.strip()]

    matrices: dict[str, torch.Tensor] = {}
    csv_path = out_dir / "refusal_long.csv"
    n_layers = len(layer_indices)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["conversation_id", "objective_index", "sample_index", "turn", "layer", "phi", "jailbroken"])

        max_layer = max(layer_indices)
        d_model = W.shape[1]
        for entry in tqdm(index, desc="Scoring"):
            conv_id = entry["conversation_id"]
            h = load_file(str(acts_dir / entry["file"]))["resid"].float()  # (T, L_all, D)
            if h.ndim != 3 or max_layer >= h.shape[1] or h.shape[2] != d_model:
                raise ValueError(
                    f"{conv_id}: stored resid {tuple(h.shape)} is incompatible with the "
                    f"probes (need >= {max_layer + 1} layers and hidden dim {d_model}). "
                    "The probe set and the collected activations were produced for "
                    "different models or layer conventions."
                )
            h = h[:, layer_indices, :]                                     # (T, L, D)

            # phi[t, l] = sum_d W[l,d] * h[t,l,d] + b[l]
            phi_tl = torch.einsum("tld,ld->tl", h, W) + b               # (T, L)
            phi_lt = phi_tl.transpose(0, 1).contiguous()                # (L, T)
            matrices[conv_id] = phi_lt

            turns = entry["turns"]
            jailbroken = entry.get("jailbroken") or [None] * len(turns)
            for ti, turn in enumerate(turns):
                jb = jailbroken[ti] if ti < len(jailbroken) else None
                for li, layer in enumerate(layer_indices):
                    writer.writerow([
                        conv_id, entry.get("objective_index"), entry.get("sample_index"),
                        turn, layer, f"{phi_lt[li, ti].item():.6f}",
                        "" if jb is None else int(bool(jb)),
                    ])

    save_file(matrices, str(out_dir / "refusal_matrices.safetensors"))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "svm_dir": svm_dir,
            "activations_dir": str(acts_dir),
            "layer_indices": layer_indices,
            "n_layers": n_layers,
            "n_conversations": len(matrices),
        }, f, indent=2)

    print(f"Scored {len(matrices)} conversations across {n_layers} layers.")
    print(f"  matrices -> {out_dir / 'refusal_matrices.safetensors'}")
    print(f"  long CSV -> {csv_path}")


if __name__ == "__main__":
    main()
