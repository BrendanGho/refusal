"""Sanity check: (layer x turn) refusal heatmaps grouped by jailbreak outcome.

Confirms the collected residual stream is meaningful and that a refusal signature exists
worth chasing — BEFORE any fancier exploration. Reads the probe-scored matrices written by
refusal_matrix.py plus the per-conversation per-turn `jailbroken` labels from the
collector's index.jsonl. Each group's mean (layer x turn) heatmap is rendered side by
side on a shared, diverging, zero-centered color scale.

Two groupings are available via --group_mode:
  recovery (default): THREE separate PNGs — all data; conversations that get jailbroken
    then recover (refuse again at a later turn); and conversations jailbroken with no
    recovery. Within each PNG, panels split by the turn at which the conversation FIRST
    gets jailbroken (t1..tK, plus "never"/"unlabeled" in the all-data PNG).
  first_turn: a single PNG with one panel per first-jailbroken turn (t1..tK), plus
    "never" and "unlabeled".

No alignment tricks — just subsets. This is the deliberately-simple committed analysis;
PCA / decomposition / head-attribution are left open (designed after seeing this).

Run from repo root:
  .venv/bin/python analysis/01_probe_heatmap.py --refusal_dir <activations_dir>/refusal
"""

import argparse
import json
import random
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
    p.add_argument("--out", type=str, default=None,
                   help="recovery mode: output DIR for the 3 PNGs (default <refusal_dir>). "
                        "first_turn mode: output PNG path (default <refusal_dir>/heatmap_by_outcome.png).")
    p.add_argument("--group_mode", type=str, default="recovery", choices=["recovery", "first_turn"],
                   help="recovery: 3 separate PNGs (all / jailbroken+recovers / jailbroken+no-recovery), "
                        "each paneled by first-jailbroken turn. "
                        "first_turn: a single PNG paneled by first-jailbroken turn (t1..tK, never, unlabeled).")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--samples_per_group", type=int, default=3,
                   help="Random conversations to dump per panel for qualitative review (0 to disable).")
    p.add_argument("--samples_out", type=str, default=None,
                   help="Output JSON for qualitative samples. Default: <refusal_dir>/qualitative_samples.json")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling.")
    p.add_argument("--goals_out", type=str, default=None,
                   help="Output JSON of per-goal ASR / jailbroken turns. Default: <refusal_dir>/goal_report.json")
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


def jailbreak_trajectory(entry: dict | None) -> str:
    """Classify a conversation's refusal trajectory from its per-turn labels.

    'unlabeled' (collected without --labeled), 'never' (never jailbroken),
    'recover' (first jailbroken, then refuses again at a later turn), or
    'no_recover' (jailbroken and never refuses again after).
    """
    if entry is None:
        return "unlabeled"
    jb = entry.get("jailbroken") or []
    if not jb or all(v is None for v in jb):
        return "unlabeled"
    first = next((i for i, v in enumerate(jb) if v), None)
    if first is None:
        return "never"
    if any(v is False for v in jb[first + 1:]):
        return "recover"
    return "no_recover"


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


def conversation_sample(activations_dir: Path, entry: dict) -> dict:
    """Reconstruct a readable conversation from its meta.json for qualitative review.

    Pairs the accumulated messages with per-turn jailbroken labels. The collector stops
    at the final user turn, so the last turn's assistant response is unavailable.
    """
    conv_id = entry["conversation_id"]
    meta_rel = entry.get("meta_file") or f"meta/{conv_id}.json"
    meta_path = activations_dir / meta_rel
    rec = {
        "conversation_id": conv_id,
        "objective": entry.get("objective"),
        "jailbroken": entry.get("jailbroken"),
    }
    if not meta_path.exists():
        rec["error"] = f"meta.json not found at {meta_path}"
        return rec
    meta = json.loads(meta_path.read_text())
    turns = meta.get("turns") or entry.get("turns") or []
    jb = meta.get("jailbroken") or entry.get("jailbroken") or []
    mpt = meta.get("messages_per_turn") or []
    # The longest accumulated context holds every question and all but the last response.
    longest = max(mpt, key=len) if mpt else []
    rec["turns"] = []
    for j, turn in enumerate(turns):
        q = longest[2 * j]["content"] if 2 * j < len(longest) else None
        r = longest[2 * j + 1]["content"] if 2 * j + 1 < len(longest) else None
        rec["turns"].append({
            "turn": turn,
            "jailbroken": jb[j] if j < len(jb) else None,
            "question": q,
            "response": r,
        })
    return rec


def write_samples(out_path: Path, activations_dir: Path, index: dict[str, dict],
                  panels: list[tuple[str, list[str]]], n: int, seed: int) -> None:
    rng = random.Random(seed)
    out = {}
    for title, ids in panels:
        chosen = ids if len(ids) <= n else rng.sample(ids, n)
        out[title] = [
            conversation_sample(activations_dir, index[c])
            for c in chosen if c in index
        ]
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Samples: " + ", ".join(f"{t}={len(v)}" for t, v in out.items()))
    print(f"Saved samples -> {out_path}")


def write_goal_report(out_path: Path, conv_ids: list[str], index: dict[str, dict]) -> None:
    """Per-goal (objective) jailbreak summary over the scored conversations.

    For each objective: ASR (share of its conversations jailbroken on any turn), which
    turn numbers a jailbreak occurred on (+ counts), recover/no-recover breakdown, and
    per-conversation labels. Sorted by descending ASR so the weakest goals surface first.
    """
    goals: dict[str, dict] = {}
    for cid in conv_ids:
        entry = index.get(cid)
        if entry is None:
            continue
        obj = entry.get("objective") or "<unknown>"
        g = goals.setdefault(obj, {
            "objective_index": entry.get("objective_index"),
            "n_conversations": 0, "n_labeled": 0, "n_jailbroken": 0,
            "jailbroken_turn_counts": {}, "first_jailbroken_turn_counts": {},
            "outcomes": {"recover": 0, "no_recover": 0, "never": 0, "unlabeled": 0},
            "conversations": [],
        })
        turns = entry.get("turns") or []
        jb = entry.get("jailbroken") or []
        labeled = bool(jb) and any(v is not None for v in jb)
        jb_turns = [int(turns[i]) for i in range(len(turns))
                    if i < len(jb) and jb[i]]

        g["n_conversations"] += 1
        g["n_labeled"] += int(labeled)
        g["n_jailbroken"] += int(bool(jb_turns))
        for t in jb_turns:
            g["jailbroken_turn_counts"][t] = g["jailbroken_turn_counts"].get(t, 0) + 1
        fjt = first_jailbroken_turn(entry)
        key = "never" if fjt is None else str(fjt)
        if labeled:
            g["first_jailbroken_turn_counts"][key] = g["first_jailbroken_turn_counts"].get(key, 0) + 1
        g["outcomes"][jailbreak_trajectory(entry)] += 1
        g["conversations"].append({
            "conversation_id": cid,
            "sample_index": entry.get("sample_index"),
            "turns": turns,
            "jailbroken": jb,
        })

    for g in goals.values():
        g["asr"] = round(g["n_jailbroken"] / g["n_labeled"], 4) if g["n_labeled"] else None
        g["jailbroken_turns"] = sorted(int(t) for t in g["jailbroken_turn_counts"])
        g["jailbroken_turn_counts"] = {str(k): v for k, v in sorted(g["jailbroken_turn_counts"].items())}

    ordered = dict(sorted(
        goals.items(),
        key=lambda kv: (-(kv[1]["asr"] if kv[1]["asr"] is not None else -1.0),
                        kv[1].get("objective_index") if kv[1].get("objective_index") is not None else 1e9),
    ))
    out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
    n_any = sum(1 for g in goals.values() if g["n_jailbroken"] > 0)
    print(f"Goal report: {len(goals)} goals, {n_any} with >=1 jailbreak -> {out_path}")


def first_turn_panels(conv_ids: list[str], index: dict[str, dict]) -> list[tuple[str, list[str]]]:
    """Group conversation ids by first-jailbroken turn: t1..tK, then never, then unlabeled."""
    groups: dict[str, list[str]] = {}
    for cid in conv_ids:
        groups.setdefault(outcome_key(index.get(cid)), []).append(cid)

    def sort_key(k: str):
        if k.startswith("t") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, {"never": 0, "unlabeled": 1}.get(k, 2))

    return [(k, groups[k]) for k in sorted(groups, key=sort_key)]


def panel_means(panels: list[tuple[str, list[str]]],
                matrices: dict[str, np.ndarray]) -> list[tuple[str, int, np.ndarray]]:
    """(title, n, mean matrix) per non-empty panel."""
    return [(title, len(ids), mean_over_conversations([matrices[c] for c in ids]))
            for title, ids in panels if ids]


def render_heatmap(means: list[tuple[str, int, np.ndarray]], suptitle: str, out_path: Path,
                   cmap: str, dpi: int, vmax: float | None = None) -> None:
    """Render side-by-side mean (layer x turn) heatmaps from precomputed panel means.

    Pass vmax to share one color scale across figures so they are directly comparable;
    otherwise it is derived from this figure's own panels.
    """
    if not means:
        print(f"Skipped {out_path.name}: no conversations in any panel.")
        return
    if vmax is None:
        vmax = max((float(np.abs(m).max()) for _, _, m in means), default=1.0) or 1.0
    n_panels = len(means)
    n_layers = means[0][2].shape[0]

    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(3, n_panels * 3.2), max(5, n_layers * 0.22)),
        squeeze=False, sharey=True,
    )
    im = None
    for ax, (title, n, m) in zip(axes[0], means):
        n_turns = m.shape[1]
        im = ax.imshow(m, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}\n(n={n})")
        ax.set_xlabel("Crescendo turn")
        ax.set_xticks(range(n_turns))
        ax.set_xticklabels(range(1, n_turns + 1))
    axes[0][0].set_ylabel("Layer")

    fig.suptitle(suptitle)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
    cbar.set_label("phi  (-compliance / +refusal)")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}: " + ", ".join(f"{t}={n}" for t, n, _ in means))


def main():
    args = get_args()
    refusal_dir = Path(args.refusal_dir)
    matrices = load_file(str(refusal_dir / "refusal_matrices.safetensors"))
    if not matrices:
        raise SystemExit(
            f"No refusal matrices found in {refusal_dir / 'refusal_matrices.safetensors'}. "
            "Run refusal_matrix.py over a non-empty collection first."
        )
    index_path = resolve_index_path(refusal_dir, args.index)
    index = load_index(index_path)

    if args.group_mode == "recovery":
        # One PNG per recovery category; within each, panels split by first-jailbroken turn.
        # 'all' overlaps the other two by design (it is every conversation).
        traj = {cid: jailbreak_trajectory(index.get(cid)) for cid in matrices}
        categories = [
            ("all", "All data", list(matrices.keys())),
            ("recover", "Jailbroken -> recovers", [c for c in matrices if traj[c] == "recover"]),
            ("no_recovery", "Jailbroken -> no recovery", [c for c in matrices if traj[c] == "no_recover"]),
        ]
        out_dir = Path(args.out) if args.out else refusal_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # Compute every panel mean first, then share one color scale so the 3 PNGs compare.
        per_cat = [(slug, label, ids, panel_means(first_turn_panels(ids, index), matrices))
                   for slug, label, ids in categories]
        vmax = max((float(np.abs(m).max()) for _, _, _, means in per_cat
                    for _, _, m in means), default=1.0) or 1.0
        sample_panels: list[tuple[str, list[str]]] = []
        for slug, label, ids, means in per_cat:
            render_heatmap(
                means,
                f"{label}: mean refusal phi (layer x turn) by first-jailbroken turn  (n={len(ids)})",
                out_dir / f"heatmap_{slug}.png", args.cmap, args.dpi, vmax=vmax,
            )
            sample_panels.append((label, ids))
    else:
        # Single PNG: every conversation, paneled by first-jailbroken turn.
        out_path = Path(args.out) if args.out else refusal_dir / "heatmap_by_outcome.png"
        panels = first_turn_panels(list(matrices.keys()), index)
        render_heatmap(
            panel_means(panels, matrices),
            "Mean refusal phi (layer x turn), grouped by first-jailbroken turn",
            out_path, args.cmap, args.dpi,
        )
        sample_panels = panels

    goals_out = Path(args.goals_out) if args.goals_out else refusal_dir / "goal_report.json"
    write_goal_report(goals_out, list(matrices.keys()), index)

    if args.samples_per_group > 0:
        samples_out = Path(args.samples_out) if args.samples_out else refusal_dir / "qualitative_samples.json"
        write_samples(samples_out, index_path.parent, index, sample_panels,
                      args.samples_per_group, args.seed)


if __name__ == "__main__":
    main()
