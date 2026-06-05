[![Generic badge](https://img.shields.io/badge/Project-Page-orange.svg)](https://latentevasion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21706-b31b1b.svg)](https://arxiv.org/pdf/2605.21706)


# Latent-space Attacks for Refusal Evasion in Language Models

Refusal is the functional behavior that enables safety-aligned language models to reject harmful or unethical requests. Recent work in mechanistic interpretability has shown that refusal can be represented through directions or decision boundaries in the model's latent space. This repository explores a complementary perspective: refusal suppression can be formulated as a latent-space evasion problem against linear refusal probes.

We introduce Controlled Latent-space Evasion (CLE), a family of attacks that intervene on internal activations in order to move harmful-prompt representations across per-layer refusal boundaries. Instead of removing a fixed refusal direction, CLE uses trained linear probes to compute controlled perturbations at selected transformer layers. The attack is governed by a small set of interpretable hyperparameters: the intervened layers, the projection margin, the probe type, and the perturbation strength.

This codebase contains the main scripts for training refusal probes, searching CLE hyperparameters, generating completions, and evaluating attack success with LLM-based judges.

## Method Overview

We start from contrastive harmful and harmless prompt splits and extract post-instruction latent representations from the target language model. For each transformer layer, we train a linear probe that separates harmful from harmless representations. These probes define layer-wise refusal boundaries in activation space.

Given a harmful prompt, CLE modifies the model's internal states so that selected activations evade these learned boundaries. The intervention uses the probe normal vector and intercept to compute a controlled movement toward the non-refusal side of the boundary. A margin parameter controls how far the activation is pushed beyond the boundary, while a beta parameter scales the strength of the intervention.

The repository implements two variants:

- `cle-a.py`: CLE-A, the additive variant. It computes a fixed per-layer delta from the prompt forward pass and applies that delta during generation.
- `cle-p.py`: CLE-P, the projective variant. It applies the controlled projection directly during both prefill and autoregressive generation.

Both variants share the same probe format and expose a similar command-line interface for models, layers, margins, datasets, batching, and evaluation.

## Repository Layout

- `classifier/train_latent.py`: trains per-layer harmful/harmless refusal probes.
- `cle-a.py`: runs the additive CLE-A attack.
- `cle-p.py`: runs the projective CLE-P attack.
- `optuna_search.py`: performs Bayesian optimization over layer windows and margins.
- `utils/eval_jailbreaks.py`: evaluates saved completions with HarmBench judges.
- `utils/probes.py`: loads SVM and single-direction probe artifacts.
- `utils/hooks.py`: contains the activation hooks used by CLE-A and CLE-P.
- `utils/runtime.py`: handles model loading, dataset loading, evaluation dispatch, and common runtime setup.
- `models/`: contains model wrappers and the Hugging Face model registry.
- `dataset/`: contains prompt splits and processed validation/test prompt files.

Generated probes, completions, evaluations, plots, and Optuna studies are written under `dataset/representations/`, `completions/`, and `runs/`.

## Using Our Code

We show here the general pipeline used to reproduce the experiments.

### 1. Setup

Create a fresh Python environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use gated Hugging Face models, set your token before running the scripts:

```bash
export HF_TOKEN=<your_huggingface_token>
```

### 2. Train Refusal Probes

Train per-layer linear probes on the model-specific harmful and harmless training prompts:

```bash
python -m classifier.train_latent \
  --model_name <model_name> \
  --device cuda \
  --artifact_dir ./dataset/representations \
  --n_samples 128
```

The script writes probe checkpoints and train representations under:

```text
dataset/representations/<model_name>/train_svm/
```

The training splits used by the probe-training script are stored in `dataset/splits/`.

### 3. Search Attack Hyperparameters

Run Bayesian optimization on the validation set to search for the best contiguous layer window and shared margin:

```bash
python optuna_search.py \
  --target pipeline \
  --model_name llama2-7b \
  --device cuda:0 \
  --dataset harmbench_val \
  --margin_low 0.1 \
  --margin_high 5.0 \
  --margin_step 0.1 \
  --layer_start_low 3 \
  --layer_start_high 24 \
  --layer_end_low 6 \
  --layer_end_high 32 \
  --trials 500 \
  --sampler tpe
```

Use `--target projection` to optimize CLE-P instead of CLE-A.

During validation, `optuna_search.py` evaluates completions with the HarmBench validation judge. Each trial writes generated completions, evaluations, and Optuna artifacts under:

```text
completions/<model_name>/<target>/
runs/<model_name>/<target>_optuna/
```

The best configuration is saved in the Optuna result summary:

```json
{
  "best_params": {
    "layer_start": 5,
    "layer_end": 25,
    "margin": 3.1
  },
  "best_user_attrs": {
    "layers_arg": "5-25",
    "margin": 3.1,
    "evaluation_path": "..."
  }
}
```

Use `best_user_attrs.layers_arg` and `best_user_attrs.margin` for final test-set generation.

### 4. Run CLE-A on the Test Set

Run the additive variant with the selected layers and margin:

```bash
python cle-a.py \
  --model_name llama2-7b \
  --device cuda:0 \
  --svm_dir ./dataset/representations/llama2-7b/train_svm \
  --layers 5-25 \
  --margin 3.1 \
  --batch_size 32 \
  --dataset harmbench_test \
  --out_dir ./completions/llama2-7b/cle-a
```

CLE-A first computes a per-layer additive delta at the prompt position and then keeps that delta active while the model generates.

### 5. Run CLE-P on the Test Set

Run the projective variant with the selected layers and margin:

```bash
python cle-p.py \
  --model_name llama2-7b \
  --device cuda:0 \
  --svm_dir ./dataset/representations/llama2-7b/train_svm \
  --layers 5-25 \
  --margin 1.6 \
  --batch_size 32 \
  --dataset harmbench_test \
  --out_dir ./completions/llama2-7b/cle-p
```

CLE-P keeps projection hooks active throughout prefill and generation.

### 6. Evaluate Saved Completions

Add `--evaluate` to either CLE command if you want to run evaluation immediately after generation.

You can also evaluate an existing completions file:

```bash
python utils/eval_jailbreaks.py \
  --completions_path ./completions/llama2-7b/cle-a/completions_<run_tag>.json \
  --methodologies harmbench \
  --evaluation_path ./completions/llama2-7b/cle-a/evaluation/evaluation_<run_tag>.json
```

Final test-set evaluation uses the Llama-based HarmBench judge through `--methodologies harmbench`. This is intentionally different from validation-time optimization, which uses the Mistral-based validation judge.

## Additional Options

To intervene on all available probe layers, pass:

```bash
--layers all
```

To intervene on a contiguous layer window, pass an end-exclusive interval:

```bash
--layers 5-25
```

To intervene on specific layers, pass a comma-separated list:

```bash
--layers 10,14,20
```

CLE also supports per-layer margins aligned with the selected layers:

```bash
python cle-a.py \
  --model_name llama2-7b \
  --device cuda:0 \
  --svm_dir ./dataset/representations/llama2-7b/train_svm \
  --layers 5-8 \
  --margin 3.1 \
  --batch_size 32 \
  --layer_margins 2.9,3.1,3.3 \
  --dataset harmbench_test
```

The scripts can load two probe types:

- `--probe_type svm`: loads learned SVM probes from `svm_layerXX.pt`.
- `--probe_type single_direction`: loads `sd_layerXX.pt` directions and derives the midpoint bias from `HFx_train.pt` and `HLx_train.pt`.

## Acknowledgment

This work was partly supported by the EU-funded Horizon Europe projects Sec4AI4Sec (GA No. 101120393) and CoEvolution (GA No. 101168560), by project FISA-2023-00128, funded under the MUR program “Fondo Italiano per le Scienze Applicate,” and by Fondazione di Sardegna under the project “LatentShield: Protecting Large Language Models from Prompt Injection in Latent Space” (CUP: F83C26000350007).

<p align="center">
  <img src="media/logo_s4ai4sec.png" alt="Sec4AI4Sec logo" height="70">
  <img src="media/coevol_logo.png" alt="CoEvolution logo" height="70">
  <img src="media/fisa.jpg" alt="FISA logo" height="70">
  <img src="media/fds.jpg" alt="Fondo per la Repubblica Digitale logo" height="70">
</p>
