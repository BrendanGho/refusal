import torch
import numpy as np
import os
import json
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

from models import Llama2_7b, Llama3_8b, Mistral7B_RR, Gemma3_12b, Phi35Mini, Qwen2_32b, Llama32_3b, Mistral7b_Instruct, Ministral3_14b, GPT20b, Mixtral8x7b_Instruct, Olmo3_7b, Phi4_15b, Qwen35_9b, DeepSeekR1_8b

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama2-7b")
    parser.add_argument('--device', default="cuda:1", type=str, help="cuda device")
    parser.add_argument('--artifact_dir', type=str, default="./dataset/representations/", help="Directory di salvataggio") 
    parser.add_argument('--n_samples', type=int, default=-1, help="Campioni per classe (-1 = tutti)")
    return parser.parse_args()

def get_model(model_name, device):
    models = {
        'llama2-7b': Llama2_7b, 'llama3-8b': Llama3_8b,
        'qwen2-32b': Qwen2_32b,
        'mistral-7b-rr': Mistral7B_RR,
        'gemma3-12b': Gemma3_12b, 'phi35mini': Phi35Mini, 'mixtral-8x7b': Mixtral8x7b_Instruct,
        'llama32-3b': Llama32_3b, 'mistral-7bv3': Mistral7b_Instruct, 'ministral3-14b': Ministral3_14b, 'gpt-oss-20b': GPT20b,
        'olmo3-7b': Olmo3_7b, 'phi4-15b': Phi4_15b, 'qwen35-9b': Qwen35_9b, 'deepseek-r1-8b': DeepSeekR1_8b
    }
    print(f"Loading model: {model_name} on device: {device}")
    return models[model_name](device=device)

def load_arditi(args):
    harmful_path = f"./dataset/splits/{args.model_name}/harmful_train_filtered.json"
    harmless_path = f"./dataset/splits/{args.model_name}/harmless_train_filtered.json"
    
    print("Loading Arditi dataset...")
    with open(harmful_path, "r", encoding="utf-8") as f:
        harmful_prompts = json.load(f)
    with open(harmless_path, "r", encoding="utf-8") as f:
        harmless_prompts = json.load(f)
    
    return harmful_prompts, harmless_prompts



def main():
    args = get_args()
    model = get_model(args.model_name, device=args.device)
    harmful_prompts, harmless_prompts = load_arditi(args)

    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[:args.n_samples]
        harmless_prompts = harmless_prompts[:args.n_samples]

    all_prompts = harmful_prompts + harmless_prompts
    all_labels = [1] * len(harmful_prompts) + [0] * len(harmless_prompts)

    # 1. Estrazione di TUTTI i layer contemporaneamente
    print(f"Extracting hidden states for ALL layers...")
    # X_all_layers sarà una lista di tensori (num_layers, dim)
    X_all_layers_list = []

    for p in tqdm(all_prompts, total=len(all_prompts)):
        # reps shape: (1, num_layers, dim)
        reps = model.get_representations(p, token_pos=-1)

        # NumPy does not support bfloat16; keep a stable dtype for downstream sklearn.
        X_all_layers_list.append(reps.to(dtype=torch.float32).cpu()) # Spostiamo su CPU per risparmiare VRAM

    # Convertiamo in un unico tensore (N_samples, num_layers, dim)
    X_full = torch.cat(X_all_layers_list, dim=0).numpy()
    Y = np.array(all_labels)
    num_harm = len(harmful_prompts)

    num_layers = X_full.shape[1]

    # Creazione cartella base
    model_dir = os.path.join(args.artifact_dir, args.model_name, 'train_svm')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    hf_path = os.path.join(model_dir, "HFx_train.pt")
    hl_path = os.path.join(model_dir, "HLx_train.pt")
    torch.save(torch.from_numpy(X_full[:num_harm]).float(), hf_path)
    torch.save(torch.from_numpy(X_full[num_harm:]).float(), hl_path)
    print(f"Saved Arditi train representations to {hf_path} and {hl_path}")

    # 2. Loop di training per ogni layer
    print(f"\nStarting SVM training for {num_layers} layers...")
    
    for l in range(num_layers):
        X_layer = X_full[:, l, :] # (N_samples, dim)
        X_harm = X_layer[:num_harm, :]
        X_harmless = X_layer[num_harm:, :]

        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, Y, test_size=0.1, random_state=42, stratify=Y
        )

        clf = LinearSVC(C=0.1, dual="auto", max_iter=1000000, random_state=42)
        clf.fit(X_train, y_train)

        # SD direction: centroid(harmful) - centroid(harmless)
        mu_harm = X_harm.mean(axis=0)
        mu_harmless = X_harmless.mean(axis=0)
        sd = mu_harm - mu_harmless
        sd_norm = np.linalg.norm(sd) + 1e-12
        sd_unit = sd / sd_norm
        sd_bias = -0.5 * float(np.dot(sd_unit, mu_harm + mu_harmless))
        sd_test_scores = X_test @ sd_unit + sd_bias
        sd_test_pred = (sd_test_scores > 0).astype(y_test.dtype)
        sd_acc = float(np.mean(sd_test_pred == y_test))


        # Cosine similarity between SVM w and SD
        w = clf.coef_[0]
        w_norm = np.linalg.norm(w) + 1e-12
        cos_sim = float(np.dot(w, sd) / (w_norm * sd_norm))


        acc = clf.score(X_test, y_test)
        report = classification_report(y_test, clf.predict(X_test), target_names=['Harmless', 'Harmful'], output_dict=False)
        
        # Salvataggio layer specifico
        save_obj = {
            "w": torch.from_numpy(clf.coef_[0]).float(),
            "b": torch.tensor(clf.intercept_[0]).float(),
            "layer_idx": l,
            "model_name": args.model_name,
            "hidden_dim": X_layer.shape[-1],
            "accuracy": acc,
            "single_direction_accuracy": sd_acc,
            "single_direction_bias": sd_bias,
            "single_direction_norm": float(sd_norm),
            "report": report,
            "cosine_w_sd": cos_sim,
        }

        save_path = os.path.join(model_dir, f"svm_layer{l:02d}.pt")
        torch.save(save_obj, save_path)

        sd_save_obj = {
            "sd": torch.from_numpy(sd).float(),
            "layer_idx": l,
            "model_name": args.model_name,
            "hidden_dim": X_layer.shape[-1],
            "cosine_w_sd": cos_sim,
            "accuracy": sd_acc,
            "bias": sd_bias,
            "norm": float(sd_norm),
            "normalized_by_default_in_pipeline": True,
        }
        sd_save_path = os.path.join(model_dir, f"sd_layer{l:02d}.pt")
        torch.save(sd_save_obj, sd_save_path)
        
        print(
            f"Layer {l:02d} | SVM Acc: {acc:.4f} | SD Acc: {sd_acc:.4f} | cos(w,SD)={cos_sim:+.4f} | "
        )

if __name__ == "__main__":
    main()
