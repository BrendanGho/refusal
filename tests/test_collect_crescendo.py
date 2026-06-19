"""GPU-free unit tests for the CRESCENDO collection + scoring pipeline.

No pytest in the venv, so this is a plain script: each check raises on failure and
prints a PASS line. Run from repo root:

    .venv/bin/python tests/test_collect_crescendo.py

Covers (no GPU, no real model):
  - collect_crescendo helper functions (build_messages, message_tags, annotate_segments,
    load_turn_labels, load_index/write_index round-trip)
  - end-to-end collect_crescendo.main() with a fake capture model -> safetensors keys/shapes
    (resid, resid_embed, head_z, ragged attn_pattern_t<turn>), meta.json round-trip, index join
  - refusal_matrix.main() reads the new container and emits a jailbroken column
  - analysis/01_probe_heatmap.py renders a PNG (subprocess; module name starts with a digit)
  - attn_out / resid_mid / mlp_out reconstruction vs synthetic ground truth (embed = resid[-1])
  - LanguageModel.assert_token_alignment positive + negative via a fake tokenizer
  - LanguageModel.segment_offsets round-trip via a fake tokenizer
"""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import collect_crescendo as cc
import refusal_matrix as rm
from models.language_models import LanguageModel

# analysis/01_probe_heatmap.py can't be imported normally (module name starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "probe_heatmap", REPO_ROOT / "analysis" / "01_probe_heatmap.py"
)
heatmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heatmap)


PASSES = []


def ok(name: str) -> None:
    PASSES.append(name)
    print(f"PASS  {name}")


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------
class FakeCaptureModel:
    """Deterministic stand-in for the HF model wrapper used by collect_crescendo.main()."""

    N_LAYERS, D, NH, HD = 4, 6, 2, 3
    n_capture_calls = 0  # class-level counter to detect resume-skips

    def __init__(self, device="cpu"):
        self.device = device

    def assert_token_alignment(self):
        return None

    def enable_eager_attention(self):
        return None

    def _head_dims(self):
        return (self.NH, self.HD)

    def _lens(self, messages, system_prompt):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt is not None else []) + list(messages)
        total = 1  # BOS
        cum = []
        for m in msgs:
            total += 4 + len(str(m["content"]).split())  # header + content tokens
            cum.append(total)
        full = (cum[-1] if cum else 1) + 3  # generation prompt scaffold
        return msgs, cum, full

    def segment_offsets(self, messages, system_prompt=None):
        msgs, cum, full = self._lens(messages, system_prompt)
        offsets = []
        prev = 0
        for i, (m, end) in enumerate(zip(msgs, cum)):
            offsets.append({"role": m["role"], "index": i, "start": prev, "end": end})
            prev = end
        offsets.append({"role": "generation_prompt", "index": len(msgs), "start": prev, "end": full})
        return offsets, full

    def get_capture_messages(self, messages, system_prompt=None):
        type(self).n_capture_calls += 1
        _, _, full = self._lens(messages, system_prompt)
        L, D, NH, HD = self.N_LAYERS, self.D, self.NH, self.HD
        resid = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
        embed = torch.arange(D, dtype=torch.float32)
        head_z = torch.arange(L * NH * HD, dtype=torch.float32).reshape(L, NH, HD)
        attn_row = torch.zeros(L, NH, full, dtype=torch.float32)
        return {"resid": resid, "embed": embed, "head_z": head_z, "attn_row": attn_row, "seq_len": full}


class FakeTokenizer:
    def __init__(self, gen="<assistant>"):
        self.vocab = {}
        self.gen = gen

    def _id(self, tok):
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def _ids(self, text):
        return [self._id(t) for t in text.split()]

    def __call__(self, text, **kwargs):
        return {"input_ids": self._ids(text)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = ""
        for m in messages:
            s += f"<{m['role']}> {m['content']} </{m['role']}> "
        if add_generation_prompt:
            s += self.gen + " "
        s = s.strip()
        return self._ids(s) if tokenize else s


class FakeAlignModel(LanguageModel):
    """Bypasses model loading; exercises the real concrete LanguageModel methods."""

    def __init__(self, tokenizer, prompt_fn):
        self.tokenizer = tokenizer
        self._prompt_fn = prompt_fn

    def _get_prompt(self, prompt):
        return self._prompt_fn(prompt)

    def _get_transformer_layers(self):
        return []


# --------------------------------------------------------------------------------------
# Tests: collect helper functions
# --------------------------------------------------------------------------------------
def test_collect_helpers():
    turns = [
        {"turn": 1, "question": "q1", "response": "a1"},
        {"turn": 2, "question": "q2", "response": "a2"},
        {"turn": 3, "question": "q3", "response": "a3"},
    ]
    msgs = cc.build_messages(turns, upto_turn=2)
    assert msgs == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ], msgs
    tags = cc.message_tags(turns, upto_turn=2)
    assert tags == [("user", 1), ("assistant", 1), ("user", 2)], tags
    ok("build_messages / message_tags accumulate context up to upto_turn")

    # annotate_segments maps offset spans to user_t/assistant_t/generation_prompt labels.
    offsets = [
        {"role": "user", "index": 0, "start": 0, "end": 5},
        {"role": "assistant", "index": 1, "start": 5, "end": 9},
        {"role": "user", "index": 2, "start": 9, "end": 14},
        {"role": "generation_prompt", "index": 3, "start": 14, "end": 17},
    ]
    segs = cc.annotate_segments(offsets, tags, has_system=False)
    labels = [s["segment"] for s in segs]
    assert labels == ["user_t1", "assistant_t1", "user_t2", "generation_prompt"], labels
    assert segs[0]["turn"] == 1 and segs[2]["turn"] == 2
    ok("annotate_segments labels user/assistant turns + generation_prompt")

    # has_system shifts the tag index by one.
    offsets_sys = [{"role": "system", "index": 0, "start": 0, "end": 3}] + [
        {**o, "index": o["index"] + 1, "start": o["start"] + 3, "end": o["end"] + 3} for o in offsets
    ]
    segs_sys = cc.annotate_segments(offsets_sys, tags, has_system=True)
    assert [s["segment"] for s in segs_sys] == ["system", "user_t1", "assistant_t1", "user_t2", "generation_prompt"]
    ok("annotate_segments handles a leading system segment")


def test_load_turn_labels_and_index(tmp: Path):
    labeled = tmp / "labeled.jsonl"
    rows = [
        {"conversation_id": "c1", "turn": 1, "jailbroken": False},
        {"conversation_id": "c1", "turn": 2, "jailbroken": True},
        {"conversation_id": "c2", "turn": 1, "jailbroken": None},
    ]
    labeled.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labels = cc.load_turn_labels(str(labeled))
    assert labels[("c1", 1)] is False and labels[("c1", 2)] is True and labels[("c2", 1)] is None
    assert cc.load_turn_labels(None) == {}
    ok("load_turn_labels keys by (conversation_id, turn)")

    index_path = tmp / "index.jsonl"
    index = {"b": {"conversation_id": "b", "x": 2}, "a": {"conversation_id": "a", "x": 1}}
    cc.write_index(index_path, index)
    reloaded = cc.load_index(index_path)
    assert reloaded == index
    # write_index sorts by conversation_id.
    first = json.loads(index_path.read_text().splitlines()[0])
    assert first["conversation_id"] == "a"
    ok("load_index / write_index round-trip (sorted)")


# --------------------------------------------------------------------------------------
# Test: end-to-end collect_crescendo.main()
# --------------------------------------------------------------------------------------
def test_collect_end_to_end(tmp: Path):
    input_path = tmp / "convs.jsonl"
    convs = [
        {
            "id": "conv_jb",
            "objective": "obj A",
            "objective_index": 0,
            "sample_index": 0,
            "turns": [
                {"turn": 1, "question": "hello there", "response": "hi back"},
                {"turn": 2, "question": "tell me more please", "response": "ok sure thing"},
            ],
        },
        {
            "id": "conv_never",
            "objective": "obj B",
            "objective_index": 1,
            "sample_index": 0,
            "turns": [
                {"turn": 1, "question": "single turn only", "response": "a response"},
            ],
        },
        {"id": "bad", "error": "skip me"},  # should be skipped by load_conversations
    ]
    input_path.write_text("\n".join(json.dumps(c) for c in convs) + "\n")

    labeled_path = tmp / "labeled.jsonl"
    labeled_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"conversation_id": "conv_jb", "turn": 1, "jailbroken": False},
                {"conversation_id": "conv_jb", "turn": 2, "jailbroken": True},
                {"conversation_id": "conv_never", "turn": 1, "jailbroken": False},
            ]
        )
        + "\n"
    )

    out_dir = tmp / "acts_out"

    orig = cc.get_model_info
    cc.get_model_info = lambda name: {"class": FakeCaptureModel}
    orig_argv = sys.argv
    sys.argv = [
        "collect_crescendo.py",
        "--model_name", "fake",
        "--device", "cpu",
        "--input", str(input_path),
        "--labeled", str(labeled_path),
        "--out_dir", str(out_dir),
        "--dtype", "float32",
    ]
    try:
        cc.main()
    finally:
        cc.get_model_info = orig
        sys.argv = orig_argv

    # safetensors keys + shapes
    st = load_file(str(out_dir / "acts" / "conv_jb.safetensors"))
    L, D, NH, HD = FakeCaptureModel.N_LAYERS, FakeCaptureModel.D, FakeCaptureModel.NH, FakeCaptureModel.HD
    assert tuple(st["resid"].shape) == (2, L, D), st["resid"].shape
    assert tuple(st["resid_embed"].shape) == (2, D), st["resid_embed"].shape
    assert tuple(st["head_z"].shape) == (2, L, NH, HD), st["head_z"].shape
    assert "attn_pattern_t1" in st and "attn_pattern_t2" in st
    # ragged: later turn has a longer key dimension than the earlier one
    assert st["attn_pattern_t1"].shape[-1] < st["attn_pattern_t2"].shape[-1]
    assert st["attn_pattern_t1"].shape[:2] == (L, NH)
    ok("collect: safetensors has resid/resid_embed/head_z + ragged attn_pattern_t<turn>")

    # meta.json round-trip
    meta = json.loads((out_dir / "meta" / "conv_jb.json").read_text())
    assert meta["turns"] == [1, 2]
    assert meta["jailbroken"] == [False, True]
    assert meta["pattern_keys"] == ["attn_pattern_t1", "attn_pattern_t2"]
    assert len(meta["pattern_seq_lens"]) == 2 and meta["pattern_seq_lens"][0] < meta["pattern_seq_lens"][1]
    assert meta["pattern_seq_lens"][1] == st["attn_pattern_t2"].shape[-1]
    assert meta["shapes"]["resid_embed"] == [2, D]
    # segment offsets present at user/assistant granularity
    segs_t2 = meta["segment_offsets"][1]
    seg_labels = [s["segment"] for s in segs_t2]
    assert seg_labels == ["user_t1", "assistant_t1", "user_t2", "generation_prompt"], seg_labels
    ok("collect: meta.json round-trips labels, pattern_seq_lens, segment offsets")

    # index join
    index = cc.load_index(out_dir / "index.jsonl")
    assert set(index) == {"conv_jb", "conv_never"}  # 'bad' skipped
    assert index["conv_jb"]["jailbroken"] == [False, True]
    assert index["conv_never"]["num_turns"] == 1
    ok("collect: index.jsonl joins per-turn jailbroken labels, skips errored convs")

    return out_dir


def _run_collect(input_path, out_dir, labeled=None):
    orig = cc.get_model_info
    cc.get_model_info = lambda name: {"class": FakeCaptureModel}
    orig_argv = sys.argv
    argv = ["collect_crescendo.py", "--model_name", "fake", "--device", "cpu",
            "--input", str(input_path), "--out_dir", str(out_dir), "--dtype", "float32"]
    if labeled is not None:
        argv += ["--labeled", str(labeled)]
    sys.argv = argv
    try:
        cc.main()
    finally:
        cc.get_model_info = orig
        sys.argv = orig_argv


# --------------------------------------------------------------------------------------
# Test: outcome_key buckets unlabeled vs never vs t<turn>  (heatmap fix #1)
# --------------------------------------------------------------------------------------
def test_outcome_key():
    assert heatmap.outcome_key(None) == "unlabeled"
    assert heatmap.outcome_key({"turns": [1, 2], "jailbroken": [None, None]}) == "unlabeled"
    assert heatmap.outcome_key({"turns": [1, 2], "jailbroken": []}) == "unlabeled"
    assert heatmap.outcome_key({"turns": [1, 2], "jailbroken": [False, False]}) == "never"
    assert heatmap.outcome_key({"turns": [1, 2, 3], "jailbroken": [False, True, True]}) == "t2"
    # A real 'never' (labeled but no jailbreak) must NOT collapse into unlabeled.
    assert heatmap.outcome_key({"turns": [1], "jailbroken": [False]}) != "unlabeled"
    ok("heatmap.outcome_key: separates unlabeled (all-None) from labeled-never and t<turn>")


# --------------------------------------------------------------------------------------
# Test: resume backfills labels for already-collected convs without recompute  (fix #2)
# --------------------------------------------------------------------------------------
def test_resume_label_refresh(tmp: Path):
    input_path = tmp / "resume_convs.jsonl"
    input_path.write_text(json.dumps({
        "id": "rc", "objective": "obj", "objective_index": 0, "sample_index": 0,
        "turns": [
            {"turn": 1, "question": "q one", "response": "r one"},
            {"turn": 2, "question": "q two", "response": "r two"},
        ],
    }) + "\n")
    out_dir = tmp / "resume_out"

    # Run 1: no --labeled -> all-None labels.
    FakeCaptureModel.n_capture_calls = 0
    _run_collect(input_path, out_dir, labeled=None)
    after_run1 = FakeCaptureModel.n_capture_calls
    idx = cc.load_index(out_dir / "index.jsonl")
    assert idx["rc"]["jailbroken"] == [None, None]

    # Run 2: add --labeled, same out_dir, no --overwrite.
    labeled = tmp / "resume_labels.jsonl"
    labeled.write_text("\n".join(json.dumps(r) for r in [
        {"conversation_id": "rc", "turn": 1, "jailbroken": False},
        {"conversation_id": "rc", "turn": 2, "jailbroken": True},
    ]) + "\n")
    _run_collect(input_path, out_dir, labeled=labeled)

    # No recompute happened (conv was skipped) ...
    assert FakeCaptureModel.n_capture_calls == after_run1, "resume should not recompute activations"
    # ... but labels were backfilled into both index and meta.
    idx2 = cc.load_index(out_dir / "index.jsonl")
    assert idx2["rc"]["jailbroken"] == [False, True], idx2["rc"]["jailbroken"]
    meta = json.loads((out_dir / "meta" / "rc.json").read_text())
    assert meta["jailbroken"] == [False, True], meta["jailbroken"]
    ok("collect: resume backfills --labeled into index+meta without recomputing activations")


def test_refresh_preserves_existing_labels(tmp: Path):
    """refresh_turn_labels must not clobber a real label with None for an unlabeled turn."""
    meta_path = tmp / "preserve_meta.json"
    meta_path.write_text(json.dumps({"jailbroken": [True, None]}))
    index = {"pc": {"turns": [1, 2], "jailbroken": [True, None]}}

    # Only turn 2 has a fresh label; turn 1 has none -> existing True must survive.
    turn_labels = {("pc", 2): True}
    changed = cc.refresh_turn_labels("pc", turn_labels, index, meta_path)
    assert changed is True
    assert index["pc"]["jailbroken"] == [True, True], index["pc"]["jailbroken"]
    assert json.loads(meta_path.read_text())["jailbroken"] == [True, True]

    # No fresh labels at all -> nothing changes, existing labels preserved.
    changed2 = cc.refresh_turn_labels("pc", {}, index, meta_path)
    assert changed2 is False
    assert index["pc"]["jailbroken"] == [True, True]
    ok("collect: refresh_turn_labels preserves existing labels against null turns")


# --------------------------------------------------------------------------------------
# Test: refusal_matrix rejects probe/resid layer-depth mismatch  (fix #5)
# --------------------------------------------------------------------------------------
def test_refusal_matrix_layer_bounds(tmp: Path, acts_dir: Path):
    svm_dir = tmp / "svm_toodeep"
    svm_dir.mkdir()
    D = FakeCaptureModel.D
    # One probe beyond the collected layer count -> max layer index == N_LAYERS (out of range).
    for layer in range(FakeCaptureModel.N_LAYERS + 1):
        torch.save({"w": torch.randn(D), "b": torch.tensor(0.0)}, svm_dir / f"svm_layer{layer:02d}.pt")

    orig_argv = sys.argv
    sys.argv = ["refusal_matrix.py", "--model_name", "fake",
                "--activations_dir", str(acts_dir), "--svm_dir", str(svm_dir),
                "--out_dir", str(tmp / "refusal_bad")]
    raised = False
    try:
        rm.main()
    except ValueError:
        raised = True
    finally:
        sys.argv = orig_argv
    assert raised, "expected ValueError when probe layer index exceeds stored resid depth"
    ok("refusal_matrix: raises on probe/resid layer-depth mismatch")


# --------------------------------------------------------------------------------------
# Test: refusal_matrix.main() over the produced container
# --------------------------------------------------------------------------------------
def test_refusal_matrix(tmp: Path, acts_dir: Path):
    svm_dir = tmp / "svm"
    svm_dir.mkdir()
    D = FakeCaptureModel.D
    torch.manual_seed(0)
    for layer in range(FakeCaptureModel.N_LAYERS):
        torch.save({"w": torch.randn(D), "b": torch.tensor(0.05)}, svm_dir / f"svm_layer{layer:02d}.pt")

    refusal_dir = acts_dir / "refusal"
    orig_argv = sys.argv
    sys.argv = [
        "refusal_matrix.py",
        "--model_name", "fake",
        "--activations_dir", str(acts_dir),
        "--svm_dir", str(svm_dir),
    ]
    try:
        rm.main()
    finally:
        sys.argv = orig_argv

    mats = load_file(str(refusal_dir / "refusal_matrices.safetensors"))
    assert tuple(mats["conv_jb"].shape) == (FakeCaptureModel.N_LAYERS, 2)  # (L, T)
    assert tuple(mats["conv_never"].shape) == (FakeCaptureModel.N_LAYERS, 1)
    ok("refusal_matrix: writes (layer x turn) phi matrices keyed by conversation")

    with open(refusal_dir / "refusal_long.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert "jailbroken" in reader.fieldnames, reader.fieldnames
        rows = list(reader)
    jb_vals = {(r["conversation_id"], r["turn"]): r["jailbroken"] for r in rows}
    assert jb_vals[("conv_jb", "1")] == "0" and jb_vals[("conv_jb", "2")] == "1"
    ok("refusal_matrix: refusal_long.csv carries a jailbroken column")

    # phi value matches a hand computation for one (conv, turn, layer).
    probe0 = torch.load(svm_dir / "svm_layer00.pt")
    resid0 = load_file(str(acts_dir / "acts" / "conv_jb.safetensors"))["resid"][0, 0].float()
    expected = float(resid0 @ probe0["w"].float() + probe0["b"].float())
    assert abs(float(mats["conv_jb"][0, 0]) - expected) < 1e-3
    ok("refusal_matrix: phi = w.z + b matches direct computation")

    return refusal_dir


# --------------------------------------------------------------------------------------
# Test: analysis/01_probe_heatmap.py renders (subprocess; module name starts with a digit)
# --------------------------------------------------------------------------------------
def test_probe_heatmap(refusal_dir: Path):
    out_png = refusal_dir / "heatmap_by_outcome.png"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "analysis" / "01_probe_heatmap.py"),
         "--refusal_dir", str(refusal_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out_png.exists() and out_png.stat().st_size > 0
    ok("01_probe_heatmap: renders heatmap_by_outcome.png from phi matrices + index")


# --------------------------------------------------------------------------------------
# Test: attn/MLP residual decomposition reconstruction
# --------------------------------------------------------------------------------------
def test_decomposition_reconstruction():
    L, D, NH, HD = 4, 6, 2, 3
    torch.manual_seed(1)
    embed = torch.randn(D)
    W_O = torch.randn(L, NH, HD, D)
    head_z = torch.randn(L, NH, HD)
    mlp_out_true = torch.randn(L, D)

    # Forward chain (synthetic ground truth):
    #   attn_out[l]  = sum_h head_z[l,h] @ W_O[l,h]
    #   resid_mid[l] = resid_post[l-1] + attn_out[l]   (resid_post[-1] := embed)
    #   resid_post[l]= resid_mid[l] + mlp_out[l]
    resid = torch.zeros(L, D)
    prev = embed
    for l in range(L):
        attn_out = torch.einsum("hd,hde->e", head_z[l], W_O[l])
        resid[l] = prev + attn_out + mlp_out_true[l]
        prev = resid[l]

    # Reconstruct from ONLY the stored signals (resid, embed, head_z) + fixed weights W_O.
    rec_mlp = torch.zeros(L, D)
    prev = embed
    for l in range(L):
        attn_out = torch.einsum("hd,hde->e", head_z[l], W_O[l])
        resid_mid = prev + attn_out
        rec_mlp[l] = resid[l] - resid_mid
        prev = resid[l]

    assert torch.allclose(rec_mlp, mlp_out_true, atol=1e-4), (rec_mlp - mlp_out_true).abs().max()
    ok("decomposition: mlp_out recovered from resid + embed + head_z + W_O (embed = resid[-1])")


# --------------------------------------------------------------------------------------
# Test: assert_token_alignment + segment_offsets via fake tokenizer
# --------------------------------------------------------------------------------------
def test_token_alignment():
    tok = FakeTokenizer(gen="<assistant>")

    def good_prompt(p):
        return f"<user> {p} </user> {tok.gen}"

    model_ok = FakeAlignModel(tok, good_prompt)
    model_ok.assert_token_alignment()  # must NOT raise
    ok("assert_token_alignment: passes when single-turn tail matches generation prompt")

    tok_bad = FakeTokenizer(gen="<assistant>")

    def bad_prompt(p):
        return f"<user> {p} </user> <WRONGSCAFFOLD>"

    model_bad = FakeAlignModel(tok_bad, bad_prompt)
    raised = False
    try:
        model_bad.assert_token_alignment()
    except RuntimeError:
        raised = True
    assert raised, "expected RuntimeError on mismatched generation-prompt scaffold"
    ok("assert_token_alignment: raises when single-turn tail diverges from template")


def test_segment_offsets_fake():
    tok = FakeTokenizer()
    model = FakeAlignModel(tok, lambda p: p)
    messages = [
        {"role": "user", "content": "first question here"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "second"},
    ]
    offsets, full_len = model.segment_offsets(messages)
    # Contiguous, non-overlapping spans covering [0, full_len), ending on generation_prompt.
    assert offsets[0]["start"] == 0
    for a, b in zip(offsets, offsets[1:]):
        assert a["end"] == b["start"], (a, b)
    assert offsets[-1]["role"] == "generation_prompt"
    assert offsets[-1]["end"] == full_len
    # full_len equals the tokenized generation-prompted template length.
    assert full_len == len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    ok("segment_offsets: contiguous spans, full_len matches gen-prompted tokenization")


# --------------------------------------------------------------------------------------
def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_collect_helpers()
        test_load_turn_labels_and_index(tmp)
        test_outcome_key()
        acts_dir = test_collect_end_to_end(tmp)
        test_resume_label_refresh(tmp)
        test_refresh_preserves_existing_labels(tmp)
        test_refusal_matrix_layer_bounds(tmp, acts_dir)
        refusal_dir = test_refusal_matrix(tmp, acts_dir)
        test_probe_heatmap(refusal_dir)
        test_decomposition_reconstruction()
        test_token_alignment()
        test_segment_offsets_fake()
    print(f"\nAll {len(PASSES)} checks passed.")


if __name__ == "__main__":
    main()
