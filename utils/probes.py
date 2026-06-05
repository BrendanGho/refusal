import os
from typing import Dict, List, Optional

import torch


ProbeDict = Dict[int, Dict[str, torch.Tensor]]


def discover_available_layers(svm_dir: str, probe_type: str = "svm") -> List[int]:
    if not os.path.isdir(svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {svm_dir}")

    if probe_type == "svm":
        prefix = "svm_layer"
    elif probe_type == "single_direction":
        prefix = "sd_layer"
    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    layer_ids: List[int] = []
    for name in os.listdir(svm_dir):
        if not (name.startswith(prefix) and name.endswith(".pt")):
            continue
        idx_str = name[len(prefix) : -len(".pt")]
        if idx_str.isdigit():
            layer_ids.append(int(idx_str))

    if not layer_ids:
        raise ValueError(f"No {prefix}XX.pt files found in {svm_dir}")

    return sorted(set(layer_ids))


def load_probes(
    *,
    probe_type: str,
    svm_dir: str,
    layer_indices: List[int],
    device: torch.device,
    explicit_reps_dir: Optional[str] = None,
) -> ProbeDict:
    if probe_type == "svm":
        return load_svms(svm_dir, layer_indices, device)

    if probe_type == "single_direction":
        reps_dir = infer_probe_reps_dir(svm_dir, explicit_reps_dir)
        if not os.path.isdir(reps_dir):
            raise FileNotFoundError(f"Single-direction representation directory not found: {reps_dir}")
        return load_single_direction_probes(svm_dir, layer_indices, device, reps_dir)

    raise ValueError(f"Unsupported probe_type: {probe_type}")


def load_svms(svm_dir: str, layer_indices: List[int], device: torch.device) -> ProbeDict:
    probes = {}
    for layer_idx in layer_indices:
        path = os.path.join(svm_dir, f"svm_layer{layer_idx:02d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing SVM checkpoint for layer {layer_idx}: {path}")

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            w = obj.float().view(-1)
            b = torch.tensor(0.0, dtype=torch.float32)
        elif isinstance(obj, dict) and "w" in obj and "b" in obj:
            w = obj["w"].float().view(-1)
            b = torch.as_tensor(obj["b"]).float().view(()).cpu()
        else:
            raise ValueError(f"Unexpected SVM format in {path}")

        probes[layer_idx] = {
            "w": w.to(device=device),
            "b": b.to(device=device),
        }
    return probes


def class_rep_paths(reps_dir: str) -> tuple[str, str]:
    return os.path.join(reps_dir, "HFx_train.pt"), os.path.join(reps_dir, "HLx_train.pt")


def has_class_representations(reps_dir: str) -> bool:
    hf_path, hl_path = class_rep_paths(reps_dir)
    return os.path.exists(hf_path) and os.path.exists(hl_path)


def infer_probe_reps_dir(svm_dir: str, explicit_reps_dir: Optional[str]) -> str:
    if explicit_reps_dir is not None:
        return explicit_reps_dir

    if has_class_representations(svm_dir):
        return svm_dir

    normalized = os.path.normpath(svm_dir)
    basename = os.path.basename(normalized)
    parent = os.path.dirname(normalized)

    # Mix-trained probes store sd_layerXX.pt in train_svm_mix_N, while the source
    # representations live in the sibling mix/ directory.
    candidate_parent = parent
    if basename.startswith("train_svm_mix_"):
        mix_dir = os.path.join(candidate_parent, "mix")
        if has_class_representations(mix_dir):
            return mix_dir

    return svm_dir


def load_class_representations(reps_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    hf_path, hl_path = class_rep_paths(reps_dir)
    if not os.path.exists(hf_path):
        raise FileNotFoundError(f"Missing harmful train representations for single-direction bias: {hf_path}")
    if not os.path.exists(hl_path):
        raise FileNotFoundError(f"Missing harmless train representations for single-direction bias: {hl_path}")

    X_harm = torch.load(hf_path, map_location="cpu").float()
    X_harmless = torch.load(hl_path, map_location="cpu").float()
    if X_harm.ndim not in {2, 3}:
        raise ValueError(f"Expected {hf_path} to have shape (N, L, D) or (N, D), got {tuple(X_harm.shape)}")
    if X_harmless.ndim not in {2, 3}:
        raise ValueError(f"Expected {hl_path} to have shape (N, L, D) or (N, D), got {tuple(X_harmless.shape)}")
    return X_harm, X_harmless


def layer_centroids(
    X_harm_all: torch.Tensor,
    X_harmless_all: torch.Tensor,
    layer_idx: int,
    reps_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_harm = X_harm_all
    X_harmless = X_harmless_all
    if X_harm.ndim == 3:
        X_harm = X_harm[:, layer_idx, :]
    if X_harmless.ndim == 3:
        X_harmless = X_harmless[:, layer_idx, :]

    if X_harm.shape[-1] != X_harmless.shape[-1]:
        raise ValueError(
            f"Representation hidden dims differ for layer {layer_idx}: "
            f"harmful={X_harm.shape[-1]} harmless={X_harmless.shape[-1]}"
        )

    return X_harm.mean(dim=0), X_harmless.mean(dim=0)


def load_single_direction_probes(
    svm_dir: str,
    layer_indices: List[int],
    device: torch.device,
    reps_dir: str,
) -> ProbeDict:
    probes = {}
    X_harm_all, X_harmless_all = load_class_representations(reps_dir)
    for layer_idx in layer_indices:
        path = os.path.join(svm_dir, f"sd_layer{layer_idx:02d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing single-direction checkpoint for layer {layer_idx}: {path}")

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            w = obj.float().view(-1)
        elif isinstance(obj, dict):
            if "sd" in obj:
                w = obj["sd"].float().view(-1)
            elif "w" in obj:
                w = obj["w"].float().view(-1)
            elif "direction" in obj:
                w = obj["direction"].float().view(-1)
            else:
                raise ValueError(f"Unexpected single-direction format in {path}: missing 'sd', 'w', or 'direction'")
        else:
            raise ValueError(f"Unexpected single-direction format in {path}")

        mu_harm, mu_harmless = layer_centroids(X_harm_all, X_harmless_all, layer_idx, reps_dir)
        if w.numel() != mu_harm.numel():
            raise ValueError(
                f"Hidden dim mismatch in layer {layer_idx}: single direction has {w.numel()} dims, "
                f"but centroids from {reps_dir} have {mu_harm.numel()} dims"
            )

        raw_norm = torch.linalg.vector_norm(w).clamp_min(1e-12)
        w = w / raw_norm

        # For unit w aligned with mu_harm - mu_harmless, place the boundary
        # halfway between class centroids. Scores remain readable signed
        # activation-space distances along this direction.
        b = -0.5 * torch.dot(w, mu_harm + mu_harmless)
        probes[layer_idx] = {
            "w": w.to(device=device),
            "b": b.float().view(()).to(device=device),
            "raw_norm": raw_norm.float().view(()).to(device=device),
        }
    return probes
