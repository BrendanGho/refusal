from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import optuna

from utils.args import build_layers_arg, selected_layers_from_arg
from utils.margin_utils import normalize_margin
from utils.probes import discover_available_layers
from utils.runtime import cuda_visible_devices_for_device


@dataclass(frozen=True)
class SearchConfig:
    target: str
    target_path: str
    model_name: str
    device: str
    svm_dir: str
    probe_type: str
    probe_reps_dir: str | None
    beta: float
    dataset: str
    limit: int | None
    max_new_tokens: int
    batch_size: int
    out_dir: str
    seed: int
    margin_low: float
    margin_high: float
    margin_step: float | None
    all_layers: bool
    layer_start_low: int
    layer_start_high: int
    layer_end_low: int | None
    layer_end_high: int | None
    min_layer_span: int
    max_layer_span: int | None


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def eval_judge_methodology() -> str:
    return "harmbench_mistral"


def eval_metric_key() -> str:
    return "harmbench_mistral_success_rate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian optimization over CLE layer ranges and margins."
    )

    parser.add_argument(
        "--target",
        type=str,
        default="pipeline",
        choices=["pipeline", "projection"],
        help="Script to optimize: cle-a.py (pipeline/CLE-A) or cle-p.py (projection/CLE-P).",
    )
    parser.add_argument("--model_name", type=str, default="llama2-7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--svm_dir", type=str, default=None)
    parser.add_argument(
        "--probe_type",
        type=str,
        default="svm",
        choices=["svm", "single_direction"],
        help="Probe artifact type passed to the optimized script.",
    )
    parser.add_argument(
        "--probe_reps_dir",
        type=str,
        default=None,
        help="Directory containing HFx_train.pt and HLx_train.pt for single-direction probe biases.",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random", "qmc"])
    parser.add_argument("--direction", type=str, default="maximize", choices=["minimize", "maximize"])
    parser.add_argument("--n_startup_trials", type=int, default=None)
    parser.add_argument("--margin_low", type=float, default=None)
    parser.add_argument("--margin_high", type=float, default=None)
    parser.add_argument(
        "--margin_step",
        type=float,
        default=None,
        help="Optional discretization step for margin. Example: 0.1",
    )
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="Search only margin while fixing layers to 'all'.",
    )

    parser.add_argument("--layer_start_low", type=int, default=None)
    parser.add_argument("--layer_start_high", type=int, default=None)
    parser.add_argument(
        "--layer_end_low",
        type=int,
        default=None,
        help="Optional lower bound for the end-exclusive layer index.",
    )
    parser.add_argument(
        "--layer_end_high",
        type=int,
        default=None,
        help="Optional upper bound for the end-exclusive layer index.",
    )
    parser.add_argument("--min_layer_span", type=int, default=1)
    parser.add_argument("--max_layer_span", type=int, default=None)

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///runs/llama2-7b/pipeline_optuna/study.db",
    )
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_run_tag(
    dataset: str,
    limit: int | None,
    layers_arg: str,
    probe_type: str,
    beta: float,
    margin: float,
    seed: int,
    selected_layers: list[int] | None = None,
) -> str:
    if selected_layers is None:
        selected_layers = selected_layers_from_arg(layers_arg) if layers_arg.strip().lower() != "all" else []
    if layers_arg.strip().lower() == "all":
        layers_str = "all"
    else:
        if len(selected_layers) == 1:
            layers_str = str(selected_layers[0])
        else:
            layers_str = layers_arg.replace(",", "_").replace("-", "to")

    if limit:
        limit_str = f"limit{limit}"
    else:
        limit_str = "FULL"

    probe_tag = "" if probe_type == "svm" else f"_probe{probe_type}"
    margin_tag = f"margin{margin}"
    return f"{dataset}_{limit_str}_layers{layers_str}{probe_tag}_beta{beta}_{margin_tag}_seed{seed}"


def target_path_for_name(target: str) -> str:
    root = repo_root()
    if target == "pipeline":
        return os.path.join(root, "cle-a.py")
    if target == "projection":
        return os.path.join(root, "cle-p.py")
    raise ValueError(f"Unsupported target: {target}")


def make_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "tpe":
        startup = args.n_startup_trials
        if startup is None:
            startup = max(8, args.trials // 4)
        return optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=True,
            group=True,
            n_startup_trials=startup,
        )
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    if args.sampler == "qmc":
        return optuna.samplers.QMCSampler(seed=args.seed)
    raise ValueError(f"Unsupported sampler: {args.sampler}")


def validate_search_bounds(args: argparse.Namespace, available_layers: list[int]) -> SearchConfig:
    if args.all_layers and any(
        value is not None
        for value in (
            args.layer_start_low,
            args.layer_start_high,
            args.layer_end_low,
            args.layer_end_high,
            args.max_layer_span,
        )
    ):
        raise ValueError("--all_layers cannot be combined with layer window arguments")
    if args.margin_low is None or args.margin_high is None:
        raise ValueError("--margin_low and --margin_high are required")

    if args.margin_low > args.margin_high:
        raise ValueError("--margin_low must be <= --margin_high")
    if args.margin_step is not None and args.margin_step <= 0:
        raise ValueError("--margin_step must be > 0")
    if args.min_layer_span < 1:
        raise ValueError("--min_layer_span must be >= 1")

    if args.svm_dir is None:
        base_svm_dir = os.path.join(repo_root(), "dataset", "representations", args.model_name, "train_svm")
        svm_dir = base_svm_dir
    else:
        svm_dir = args.svm_dir

    min_available = min(available_layers)
    max_available = max(available_layers)
    max_end_exclusive = max_available + 1

    start_low = args.layer_start_low if args.layer_start_low is not None else min_available
    start_high = args.layer_start_high if args.layer_start_high is not None else max_available
    end_low = args.layer_end_low
    end_high = args.layer_end_high if args.layer_end_high is not None else max_end_exclusive

    if args.all_layers:
        start_low = start_high = min_available
        end_low = end_high = max_end_exclusive

    if start_low < min_available or start_high > max_available:
        raise ValueError(
            f"Requested layer_start bounds [{start_low}, {start_high}] exceed available layers "
            f"[{min_available}, {max_available}]"
        )
    if end_high > max_end_exclusive:
        raise ValueError(
            f"Requested layer_end_high={end_high} exceeds max end-exclusive bound {max_end_exclusive}"
        )

    if start_low > start_high:
        raise ValueError("Invalid layer start bounds")
    if end_low is not None and end_low > end_high:
        raise ValueError("Invalid layer end bounds")

    max_span = args.max_layer_span
    if max_span is not None and max_span < args.min_layer_span:
        raise ValueError("--max_layer_span must be >= --min_layer_span")

    target_path = target_path_for_name(args.target)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target script not found: {target_path}")

    base_out_dir = args.out_dir or os.path.join(repo_root(), "completions", args.model_name, args.target)
    out_dir = base_out_dir

    return SearchConfig(
        target=args.target,
        target_path=target_path,
        model_name=args.model_name,
        device=args.device,
        svm_dir=svm_dir,
        probe_type=args.probe_type,
        probe_reps_dir=args.probe_reps_dir,
        beta=args.beta,
        dataset=args.dataset,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        out_dir=out_dir,
        seed=args.seed,
        margin_low=args.margin_low,
        margin_high=args.margin_high,
        margin_step=args.margin_step,
        all_layers=args.all_layers,
        layer_start_low=start_low,
        layer_start_high=start_high,
        layer_end_low=end_low,
        layer_end_high=end_high,
        min_layer_span=args.min_layer_span,
        max_layer_span=max_span,
    )


def sample_layer_range(trial: optuna.Trial, cfg: SearchConfig) -> tuple[int, int]:
    layer_start = trial.suggest_int("layer_start", cfg.layer_start_low, cfg.layer_start_high)

    min_end = layer_start + cfg.min_layer_span
    if cfg.layer_end_low is not None:
        min_end = max(min_end, cfg.layer_end_low)

    max_end = cfg.layer_end_high if cfg.layer_end_high is not None else cfg.layer_start_high + 1
    if cfg.max_layer_span is not None:
        max_end = min(max_end, layer_start + cfg.max_layer_span)

    if min_end > max_end:
        raise ValueError(
            f"No valid layer_end for sampled start={layer_start}. "
            f"Check fixed bounds / min/max span constraints."
        )

    layer_end = trial.suggest_int("layer_end", min_end, max_end)

    return layer_start, layer_end


class TrialExecutor:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg

    def run(
        self,
        *,
        layers_arg: str,
        margin: float,
        run_tag: str,
    ) -> tuple[float, str]:
        eval_path = os.path.join(self.cfg.out_dir, "evaluation", f"evaluation_{run_tag}.json")
        completions_path = os.path.join(self.cfg.out_dir, f"completions_{run_tag}.json")
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)

        attack_cmd = self.build_attack_command(layers_arg, margin)
        subprocess.run(attack_cmd, cwd=repo_root(), check=True)
        self.require_file(completions_path, "completions")

        eval_cmd = self.build_eval_command(completions_path, eval_path)
        subprocess.run(eval_cmd, cwd=repo_root(), check=True, env=self.eval_env())
        self.require_file(eval_path, "evaluation")
        with open(eval_path) as f:
            evaluation = json.load(f)
        return float(evaluation[eval_metric_key()]), eval_path

    def build_attack_command(
        self,
        layers_arg: str,
        margin: float,
    ) -> list[str]:
        cfg = self.cfg
        cmd: list[str] = [
            sys.executable,
            cfg.target_path,
            "--model_name",
            cfg.model_name,
            "--device",
            cfg.device,
            "--svm_dir",
            cfg.svm_dir,
            "--probe_type",
            cfg.probe_type,
            "--layers",
            layers_arg,
            "--beta",
            str(cfg.beta),
            "--margin",
            str(margin),
            "--dataset",
            cfg.dataset,
            "--max_new_tokens",
            str(cfg.max_new_tokens),
            "--batch_size",
            str(cfg.batch_size),
            "--out_dir",
            cfg.out_dir,
            "--seed",
            str(cfg.seed),
        ]
        if cfg.probe_reps_dir is not None:
            cmd.extend(["--probe_reps_dir", cfg.probe_reps_dir])
        if cfg.limit is not None:
            cmd.extend(["--limit", str(cfg.limit)])
        return cmd

    @staticmethod
    def build_eval_command(completions_path: str, eval_path: str) -> list[str]:
        return [
            sys.executable,
            os.path.join(repo_root(), "utils", "eval_jailbreaks.py"),
            "--completions_path",
            completions_path,
            "--methodologies",
            eval_judge_methodology(),
            "--evaluation_path",
            eval_path,
        ]

    def eval_env(self) -> dict[str, str]:
        env = os.environ.copy()
        eval_cuda_visible_devices = cuda_visible_devices_for_device(self.cfg.device)
        if eval_cuda_visible_devices is not None:
            # Keep the HarmBench judge on the same GPU requested for the trial.
            env["CUDA_VISIBLE_DEVICES"] = eval_cuda_visible_devices
        return env

    @staticmethod
    def require_file(path: str, label: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected {label} file was not created: {path}")


def objective(trial: optuna.Trial, cfg: SearchConfig, trial_executor: TrialExecutor) -> float:
    if cfg.all_layers:
        layer_start = cfg.layer_start_low
        layer_end = cfg.layer_end_high
        if layer_end is None:
            raise ValueError("Internal error: --all_layers requires a fixed end-exclusive layer bound.")
        layers_arg = "all"
    else:
        layer_start, layer_end = sample_layer_range(trial, cfg)
        layers_arg = build_layers_arg(layer_start, layer_end)

    margin = trial.suggest_float("margin", cfg.margin_low, cfg.margin_high, step=cfg.margin_step)
    margin = normalize_margin(margin, cfg.margin_step)

    run_tag = build_run_tag(
        dataset=cfg.dataset,
        limit=cfg.limit,
        layers_arg=layers_arg,
        probe_type=cfg.probe_type,
        beta=cfg.beta,
        margin=margin,
        seed=cfg.seed,
    )
    selected_layers = selected_layers_from_arg(layers_arg) if layers_arg != "all" else list(range(layer_start, layer_end))
    trial.set_user_attr("layers_arg", layers_arg)
    trial.set_user_attr("layer_start", layer_start)
    trial.set_user_attr("layer_end", layer_end)
    trial.set_user_attr("selected_layers", selected_layers)
    trial.set_user_attr("margin", margin)
    trial.set_user_attr("probe_type", cfg.probe_type)
    trial.set_user_attr("target", cfg.target)
    trial.set_user_attr("run_tag", run_tag)

    metric, eval_path = trial_executor.run(layers_arg=layers_arg, margin=margin, run_tag=run_tag)
    trial.set_user_attr("evaluation_path", eval_path)
    return metric


def save_artifacts(
    study: optuna.Study,
    args: argparse.Namespace,
    cfg: SearchConfig,
    study_name: str,
    results_dir: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    if args.storage is None:
        pkl_path = os.path.join(results_dir, f"{study_name}_study.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(study, f)

    best_trial = study.best_trial
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "sampler": args.sampler,
        "target": cfg.target,
        "target_path": cfg.target_path,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
        "search_space": {
            "margin_low": cfg.margin_low,
            "margin_high": cfg.margin_high,
            "margin_step": cfg.margin_step,
            "all_layers": args.all_layers,
            "layer_start_low": cfg.layer_start_low,
            "layer_start_high": cfg.layer_start_high,
            "layer_end_low": cfg.layer_end_low,
            "layer_end_high": cfg.layer_end_high,
            "min_layer_span": cfg.min_layer_span,
            "max_layer_span": cfg.max_layer_span,
        },
    }

    trial_rows: list[dict[str, Any]] = []
    for t in study.trials:
        trial_rows.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "start": t.datetime_start.isoformat() if t.datetime_start else None,
                "end": t.datetime_complete.isoformat() if t.datetime_complete else None,
                "duration_sec": t.duration.total_seconds() if t.duration else None,
            }
        )

    json_path = os.path.join(results_dir, f"{study_name}_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "trials": trial_rows}, f, indent=2)


def main() -> None:
    args = parse_args()
    root = repo_root()

    # 1. Resolve script/data/output paths.
    if args.svm_dir is None:
        base_svm_dir = os.path.join(root, "dataset", "representations", args.model_name, "train_svm")
        args.svm_dir = base_svm_dir
    if args.out_dir is None:
        base_out_dir = os.path.join(root, "completions", args.model_name, args.target)
        args.out_dir = base_out_dir

    # 2. Build the validated search space from available probe layers.
    available_layers = discover_available_layers(args.svm_dir, args.probe_type)
    cfg = validate_search_bounds(args, available_layers)

    results_dir = os.path.join(root, "runs", args.model_name, f"{args.target}_optuna")
    os.makedirs(results_dir, exist_ok=True)

    study_name = args.study_name or (
        f"{args.model_name}_{args.dataset}_{args.target}_{args.sampler}_"
        f"{'alllayers_' if args.all_layers else ''}"
        f"margin{cfg.margin_low}-{cfg.margin_high}"
    )
    sampler = make_sampler(args)

    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            direction=args.direction,
            sampler=sampler,
            storage=args.storage,
            load_if_exists=args.resume,
        )
    else:
        study = optuna.create_study(study_name=study_name, direction=args.direction, sampler=sampler)

    print(f"--- Optuna {cfg.target.title()} Search ---")
    print(f"Model: {cfg.model_name}")
    print(f"Target: {cfg.target_path}")
    print(f"Probe type: {cfg.probe_type}")
    if cfg.probe_reps_dir is not None:
        print(f"Probe reps dir: {cfg.probe_reps_dir}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Sampler: {args.sampler}")
    print(f"Objective: {args.direction} HarmBench ASR")
    if cfg.all_layers:
        print("Search mode: all layers fixed, margin optimized")
    else:
        print("Search mode: layer window and shared margin optimized")
    print(f"Available layers: {available_layers[0]}..{available_layers[-1]}")
    print(f"Results dir: {results_dir}")

    # 3. Each Optuna trial runs the selected CLE script, then evaluates it.
    trial_executor = TrialExecutor(cfg)

    # 4. Search layer window and one shared margin.
    study.optimize(
        lambda trial: objective(trial, cfg, trial_executor),
        n_trials=args.trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    # 5. Persist the study and a compact JSON summary.
    save_artifacts(study, args, cfg, study_name, results_dir)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Best attrs: {study.best_trial.user_attrs}")


if __name__ == "__main__":
    main()
