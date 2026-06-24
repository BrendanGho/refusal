"""Collect rich, per-turn activations over CRESCENDO conversations (exploratory pass).

For every turn i of every conversation the model sees the accumulated chat context
[user_1, assistant_1, ..., user_i] (chat-templated with add_generation_prompt=True). One
forward pass per turn captures, at the post-instruction token (token_pos=-1), three
signals across all layers:

  resid     (num_turns, num_layers, d_model)            residual stream per layer
  head_z    (num_turns, num_layers, n_heads, head_dim)  pre-W_O per-head attention output
  attn_pattern_t<turn>  (num_layers, n_heads, seq_k)    last-token attention pattern row
                                                        (one *ragged* tensor per turn)

We do NOT capture MLP signals: attn_out / resid_mid / mlp_out are derivable from
resid + head_z + W_O at analysis time (no extra hook, no LayerNorm-space ambiguity).

This replays the pre-generated conversations (including the vLLM-produced assistant
responses) through HF — it does not regenerate responses — so the captured activations
correspond to the exact token sequence the judge labeled. The residual stream is captured
on the same tokenization path as classifier/train_latent.py, so it is probe-scorable
(see refusal_matrix.py).

A runtime token-alignment assertion runs before any GPU work; if the multi-turn template
does not end on the same generation-prompt scaffold the probes were trained on, we abort.

Output (under --out_dir, e.g. a USB mount):
  acts/<conversation_id>.safetensors   resid, head_z, attn_pattern_t<turn> ...
  meta/<conversation_id>.json          labels, objective, text, pattern_seq_lens, offsets
  index.jsonl                          one line per conversation (lightweight)
"""

import argparse
from pathlib import Path
import json, glob, os, torch

import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

from utils.models_utils import get_model_info


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", type=str, default="llama32-3b")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--input", type=str, required=True,
                   help="Conversation JSONL produced by `build_dataset.py generate`.")
    p.add_argument("--labeled", type=str, default=None,
                   help="Turn-level labeled JSONL from `build_dataset.py judge`; joins "
                        "per-turn `jailbroken` into metadata. Optional.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory (e.g. USB mount) for acts/, meta/, index.jsonl.")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--include_system_prompt", action="store_true",
                   help="Prepend each conversation's target_system_prompt. Default off to "
                        "match probe training (train_latent.py uses no system prompt).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N conversations.")
    p.add_argument("--overwrite", action="store_true", help="Recompute conversations that already have output files.")
    return p.parse_args()


def load_conversations(path: str) -> list[dict]:
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec or not rec.get("turns"):
                continue
            convs.append(rec)
    return convs


def load_turn_labels(path: str | None) -> dict[tuple[str, int], object]:
    """{(conversation_id, turn): jailbroken} from the judge's turn-level JSONL."""
    labels: dict[tuple[str, int], object] = {}
    if path is None:
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv_id = row.get("conversation_id")
            turn = row.get("turn")
            if conv_id is None or turn is None:
                continue
            labels[(conv_id, int(turn))] = row.get("jailbroken")
    return labels


def build_messages(turns_sorted: list[dict], upto_turn: int) -> list[dict]:
    """Accumulated context [user_1, asst_1, ..., user_i] up to and including user_{upto_turn}."""
    msgs: list[dict] = []
    for t in turns_sorted:
        if t["turn"] > upto_turn:
            break
        msgs.append({"role": "user", "content": t["question"]})
        if t["turn"] < upto_turn:
            msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def message_tags(turns_sorted: list[dict], upto_turn: int) -> list[tuple[str, int]]:
    """(role, turn) for each message in build_messages(turns_sorted, upto_turn), same order."""
    tags: list[tuple[str, int]] = []
    for t in turns_sorted:
        if t["turn"] > upto_turn:
            break
        tags.append(("user", t["turn"]))
        if t["turn"] < upto_turn:
            tags.append(("assistant", t["turn"]))
    return tags


def annotate_segments(offsets: list[dict], tags: list[tuple[str, int]], has_system: bool) -> list[dict]:
    """Attach user_t<turn> / assistant_t<turn> / system / generation_prompt labels to spans."""
    out = []
    for off in offsets:
        role = off["role"]
        if role == "system":
            seg, turn = "system", None
        elif role == "generation_prompt":
            seg, turn = "generation_prompt", None
        else:
            j = off["index"] - (1 if has_system else 0)
            trole, turn = tags[j]
            seg = f"{trole}_t{turn}"
        out.append({"segment": seg, "role": role, "turn": turn,
                    "start": off["start"], "end": off["end"]})
    return out


def load_index(index_path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                index[e["conversation_id"]] = e
    return index


def write_index(index_path: Path, index: dict[str, dict]) -> None:
    with open(index_path, "w", encoding="utf-8") as f:
        for conv_id in sorted(index):
            f.write(json.dumps(index[conv_id]) + "\n")


def refresh_turn_labels(
    conv_id: str, turn_labels: dict, index: dict[str, dict], meta_path: Path
) -> bool:
    """Re-join per-turn `jailbroken` from a (possibly newer) --labeled file onto an already
    collected conversation, updating both the index entry and its meta.json in place.

    Without this, the resume guard skips collected conversations entirely, so labels added
    on a later run never reach the index (which is what refusal_matrix.py / the heatmap read).
    Only turns with a NON-None label in `turn_labels` are overwritten; an existing label is
    never destroyed by a missing or null entry in the new file. Files are rewritten only if
    something actually changed. Returns True if the index entry was modified.
    """
    entry = index.get(conv_id)
    if entry is None:
        return False
    turns = entry.get("turns") or []
    old = entry.get("jailbroken") or [None] * len(turns)
    new = (list(old) + [None] * len(turns))[: len(turns)]  # align to turns, never longer
    changed = False
    for i, turn in enumerate(turns):
        label = turn_labels.get((conv_id, int(turn)))
        if label is not None and new[i] != label:
            new[i] = label
            changed = True
    if not changed:
        return False
    entry["jailbroken"] = new
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["jailbroken"] = new
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def main():
    args = get_args()
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model_info = get_model_info(args.model_name)
    model = model_info["class"](device=args.device)

    # Blocker, enforced in code: abort before the collection forward passes if the
    # multi-turn template does not end on the same generation-prompt scaffold the probes
    # were trained on. (The model weights are already loaded above; this guards the long
    # per-turn capture loop, not the one-time load.)
    model.assert_token_alignment()

    out_dir = Path(args.out_dir)
    acts_dir = out_dir / "acts"
    meta_dir = out_dir / "meta"
    acts_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    index = load_index(index_path)
    turn_labels = load_turn_labels(args.labeled)

    conversations = load_conversations(args.input)
    if args.limit is not None:
        conversations = conversations[: args.limit]

    print(f"Loaded {len(conversations)} conversations from {args.input}")
    print(f"Writing activations to {acts_dir} (dtype={args.dtype})")
    if args.labeled:
        print(f"Joined {len(turn_labels)} turn labels from {args.labeled}")

    n_done = n_turns_total = 0
    n_heads, head_dim = model._head_dims()
    for conv in tqdm(conversations, desc="Conversations"):
        conv_id = conv["id"]
        out_path = acts_dir / f"{conv_id}.safetensors"
        meta_path = meta_dir / f"{conv_id}.json"
        if out_path.exists() and meta_path.exists() and conv_id in index and not args.overwrite:
            # Already collected: don't recompute activations, but still backfill labels
            # from a (possibly newer) --labeled file so the index/meta stay current.
            if turn_labels and refresh_turn_labels(conv_id, turn_labels, index, meta_path):
                write_index(index_path, index)
            continue

        turns_sorted = sorted(conv["turns"], key=lambda t: t["turn"])
        system_prompt = conv.get("target_system_prompt") if args.include_system_prompt else None
        has_system = system_prompt is not None

        resid_list, embed_list, headz_list = [], [], []
        pattern_tensors: dict[str, torch.Tensor] = {}
        turn_numbers, seq_lens, jailbroken, seg_offsets, messages_per_turn = [], [], [], [], []

        for t in turns_sorted:
            turn = int(t["turn"])
            messages = build_messages(turns_sorted, turn)
            cap = model.get_capture_messages(messages, system_prompt=system_prompt)

            resid_list.append(cap["resid"].to(save_dtype))            # (L, D)
            embed_list.append(cap["embed"].to(save_dtype))            # (D,)
            headz_list.append(cap["head_z"].to(save_dtype))           # (L, nh, hd)
            pattern_tensors[f"attn_pattern_t{turn}"] = cap["attn_row"].to(save_dtype)  # (L, nh, seq_k)

            offs, full_len = model.segment_offsets(messages, system_prompt)
            if full_len != cap["seq_len"]:
                raise RuntimeError(
                    f"{conv_id} turn {turn}: segment offset length {full_len} != captured "
                    f"sequence length {cap['seq_len']}; offsets would be misaligned."
                )

            turn_numbers.append(turn)
            seq_lens.append(int(cap["seq_len"]))
            jailbroken.append(turn_labels.get((conv_id, turn)))
            seg_offsets.append(annotate_segments(offs, message_tags(turns_sorted, turn), has_system))
            messages_per_turn.append(messages)

        resid = torch.stack(resid_list, dim=0).contiguous()   # (T, L, D)
        resid_embed = torch.stack(embed_list, dim=0).contiguous()  # (T, D)
        head_z = torch.stack(headz_list, dim=0).contiguous()  # (T, L, nh, hd)
        tensors = {"resid": resid, "resid_embed": resid_embed, "head_z": head_z, **pattern_tensors}
        save_file(tensors, str(out_path), metadata={"conversation_id": conv_id})

        meta = {
            "conversation_id": conv_id,
            "objective": conv.get("objective"),
            "objective_index": conv.get("objective_index"),
            "sample_index": conv.get("sample_index"),
            "include_system_prompt": has_system,
            "turns": turn_numbers,
            "jailbroken": jailbroken,
            "pattern_seq_lens": seq_lens,
            "pattern_keys": [f"attn_pattern_t{tn}" for tn in turn_numbers],
            "segment_offsets": seg_offsets,
            "messages_per_turn": messages_per_turn,
            "shapes": {"resid": list(resid.shape), "resid_embed": list(resid_embed.shape),
                       "head_z": list(head_z.shape)},
            "n_layers": int(resid.shape[1]),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "file": f"acts/{conv_id}.safetensors",
            "meta_file": f"meta/{conv_id}.json",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        index[conv_id] = {
            "conversation_id": conv_id,
            "file": f"acts/{conv_id}.safetensors",
            "meta_file": f"meta/{conv_id}.json",
            "objective": conv.get("objective"),
            "objective_index": conv.get("objective_index"),
            "sample_index": conv.get("sample_index"),
            "num_turns": len(turn_numbers),
            "turns": turn_numbers,
            "jailbroken": jailbroken,
        }
        write_index(index_path, index)

        n_done += 1
        n_turns_total += len(turn_numbers)

    print(f"Done. {n_done} conversations processed this run, {n_turns_total} turns. "
          f"Index has {len(index)} conversations total at {index_path}.")


    # root = os.path.join(os.environ["USB"], "crescendo_acts_smoke")
    # e = [json.loads(l) for l in open(f"{root}/index.jsonl") if l.strip()][0]
    # print("turns:", e["turns"], "jailbroken:", e.get("jailbroken"))

    # t = load_file(glob.glob(f"{root}/acts/*.safetensors")[0])
    # for k, v in t.items():
    #     print(f"{k:18s} {tuple(v.shape)} {v.dtype}")

    # # attention rows must sum to ~1 over key positions (softmax probs)
    # for k in sorted(t):
    #     if k.startswith("attn_pattern_t"):
    #         s = t[k].float().sum(-1)        # (L, n_heads)
    #         print(f"{k}: row-sum min {s.min():.4f} max {s.max():.4f}")


if __name__ == "__main__":
    main()
