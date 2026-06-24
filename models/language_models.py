import torch
import os
import functools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class LanguageModel(ABC):
    def __init__(self, model_name: str, system_prompt=None, device='cuda', quantization_config=None):
        print(f"[Model] Initializing {model_name} on {device}...")
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt
        self.quantization_config = quantization_config
        
        # Load Model & Tokenizer using the robust V2 logic
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Loads the model with quantization support (From Version 2)."""
        if self.model is None or self.tokenizer is None:
            token = os.environ.get('HF_TOKEN')
            print(f"Downloading/Loading model components...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, trust_remote_code=True)
            self.tokenizer.padding_side = "left" 
            
            # Handle Quantization
            bnb_config = None
            if self.quantization_config:
                if self.quantization_config == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quantization_config == "8bit":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load Model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                quantization_config=bnb_config,
                device_map=self.device,
                dtype=torch.float16 if bnb_config is None else None,
                trust_remote_code=True
            )
            self.model.eval()
            self.model.requires_grad_(False)

            # Fix Tokenizer Padding
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = 'left'
                    self.tokenizer.pad_token = '<|extra_0|>'
                    
            # Sync pad token id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            print("Model loaded successfully.")

    
    @abstractmethod
    def _get_prompt(self, prompt: str):
        """Formats a user prompt according to the model's chat template."""
        pass

    @abstractmethod
    def _get_transformer_layers(self):
        """Returns the model's transformer block modules."""
        pass

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        return [self._get_prompt(prompt) for prompt in prompts]

    def prepare_inputs(self, prompt: str, **kwargs):
        formatted_prompt = self._get_prompt(prompt=prompt)
        return self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

    def prepare_batch_inputs(self, prompts: list[str], **kwargs):
        formatted_prompts = self._format_prompts(prompts)
        return self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

    def get_representations(self, prompt: str, token_pos: int, **kwargs) -> torch.Tensor:
        """Extracts hidden states (Version 2)."""
        inputs = self.prepare_inputs(prompt, **kwargs)
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)

        # Handle different output structures
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states
        else:
            # Fallback for some models
            return None

        token_reps = torch.cat([h[:, token_pos, :] for h in hidden_states[1:]]) # Skip embedding layer
        
        return token_reps.unsqueeze(0) # (1, num_layers, dim)

    def get_representations_messages(
        self,
        messages: list[dict],
        token_pos: int = -1,
        system_prompt: str | None = None,
    ) -> torch.Tensor:
        """Last-token hidden states across all layers for a multi-turn message list.

        `messages` is a chat list like [{"role": "user", ...}, {"role": "assistant", ...}, ...].
        The list is chat-templated with add_generation_prompt=True and re-tokenized through
        the exact same path as get_representations, so the extracted token_pos=-1
        (post-instruction) activation matches train_latent.py probe-training space.
        """
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)

        if not hasattr(outputs, "hidden_states"):
            return None

        token_reps = torch.cat([h[:, token_pos, :] for h in outputs.hidden_states[1:]])
        return token_reps.unsqueeze(0)  # (1, num_layers, dim)

    # ------------------------------------------------------------------
    # Richer, model-agnostic activation capture (collect_crescendo.py)
    # ------------------------------------------------------------------
    def _attn_output_projections(self) -> list[torch.nn.Module]:
        """Return each transformer layer's attention output projection (o_proj).

        The *input* to this module is the concatenated per-head attention values
        (TransformerLens `hook_z`), shape (batch, seq, n_heads * head_dim). Model
        agnostic: tries the common attribute names across HF architectures.
        """
        layers = self._get_transformer_layers()
        projs: list[torch.nn.Module] = []
        for layer in layers:
            attn = (
                getattr(layer, "self_attn", None)
                or getattr(layer, "attn", None)
                or getattr(layer, "attention", None)
            )
            if attn is None:
                raise AttributeError(
                    f"Could not locate attention module on layer {type(layer).__name__}"
                )
            proj = None
            for name in ("o_proj", "out_proj", "dense", "wo", "c_proj"):
                proj = getattr(attn, name, None)
                if proj is not None:
                    break
            if proj is None:
                raise AttributeError(
                    f"Could not locate attention output projection on {type(attn).__name__}"
                )
            projs.append(proj)
        return projs

    def _head_dims(self) -> tuple[int, int]:
        """(n_query_heads, head_dim) read from the model config (never hardcoded)."""
        cfg = self.model.config
        n_heads = cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
        return int(n_heads), int(head_dim)

    def _formatted_ids(self, messages: list[dict], add_generation_prompt: bool) -> list[int]:
        """Token ids via the same format-then-tokenize path as get_capture_messages.

        Using this (not apply_chat_template(tokenize=True)) keeps segment offsets and the
        token-alignment check aligned with the captured sequence even when the template +
        tokenizer double the BOS.
        """
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return list(self.tokenizer(formatted)["input_ids"])

    def _formatted_len(self, messages: list[dict], add_generation_prompt: bool) -> int:
        """Token length via the same format-then-tokenize path as get_capture_messages."""
        return len(self._formatted_ids(messages, add_generation_prompt))

    def assert_token_alignment(self, eps_probe: str = "Probe alignment sentence.") -> None:
        """Guard the one invariant that, if broken, silently invalidates every capture.

        The probes (train_latent.py) score the post-instruction token of a *single-turn*
        prompt. Collection scores the post-instruction token of a *multi-turn* chat
        template. Both must end on the identical generation-prompt scaffold token(s).
        Raises RuntimeError on mismatch. Pure tokenizer ops — no GPU/forward work.

        Both sides are tokenized via the *same* format-then-tokenize path the capture uses
        (apply_chat_template(tokenize=False) -> tokenizer), so the check validates the exact
        sequence get_capture_messages produces, not a divergent tokenize=True path.
        """
        single_ids = list(self.tokenizer(self._get_prompt(eps_probe))["input_ids"])
        msgs = [
            {"role": "user", "content": "Earlier benign question."},
            {"role": "assistant", "content": "Earlier benign answer."},
            {"role": "user", "content": eps_probe},
        ]
        full = self._formatted_ids(msgs, add_generation_prompt=True)
        no_gen = self._formatted_ids(msgs, add_generation_prompt=False)
        gen_tail = full[len(no_gen):]  # content-independent generation-prompt scaffold

        if not gen_tail:
            raise RuntimeError(
                "Token alignment check: add_generation_prompt produced no trailing tokens; "
                "the chat template does not append a generation prompt as expected."
            )
        if single_ids[-len(gen_tail):] != gen_tail:
            raise RuntimeError(
                "Token alignment FAILED: the single-turn (probe-training) prompt does not "
                "end on the same generation-prompt scaffold as the multi-turn template.\n"
                f"  single-turn tail: {single_ids[-len(gen_tail):]}\n"
                f"  multi-turn  tail: {gen_tail}\n"
                "Collected activations would live in a different token position than the "
                "probes were trained on. Aborting before any GPU work."
            )
        if single_ids[-1] != full[-1]:
            raise RuntimeError(
                f"Token alignment FAILED at token_pos=-1: single={single_ids[-1]} "
                f"multi={full[-1]}."
            )

    def segment_offsets(self, messages: list[dict], system_prompt: str | None = None):
        """Per-segment [start, end) token spans for the captured (gen-prompted) sequence.

        Returns (offsets, full_len). `offsets` is a list of dicts with keys
        role/index/start/end covering each chat message plus a final
        role='generation_prompt' span. Computed by incremental tokenization, which is
        valid for prefix-stable chat templates (Llama-3 et al.). full_len must equal the
        captured sequence length.
        """
        msgs = (
            [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        ) + list(messages)
        offsets = []
        prev = 0
        for i, m in enumerate(msgs):
            end = self._formatted_len(msgs[: i + 1], add_generation_prompt=False)
            offsets.append({"role": m["role"], "index": i, "start": prev, "end": end})
            prev = end
        full_len = self._formatted_len(msgs, add_generation_prompt=True)
        offsets.append(
            {"role": "generation_prompt", "index": len(msgs), "start": prev, "end": full_len}
        )
        return offsets, full_len

    def get_capture_messages(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> dict:
        """Single forward pass capturing the three exploratory signals at token_pos=-1.

        Returns CPU float32 tensors:
          resid    (num_layers, d_model)            -- residual stream per layer
          embed    (d_model,)                       -- embedding-layer output (hidden_states[0])
          head_z   (num_layers, n_heads, head_dim)  -- pre-W_O per-head attention output
          attn_row (num_layers, n_heads, seq_k)     -- last-token attention pattern row
          seq_len  int

        MLP signals are intentionally NOT captured: attn_out / resid_mid / mlp_out are
        derivable from resid + head_z + W_O at analysis time (see plan), with no extra
        hook and no LayerNorm-space ambiguity. `embed` is the residual entering layer 0, so
        the decomposition resid_mid[l] = resid_post[l-1] + attn_out[l] is complete for every
        layer (use embed as resid_post[-1]).

        NOTE: the forward runs through SDPA, not eager / output_attentions. Eager (and the
        transformers 5.x output-capturing wrapper) materializes the full [n_heads, seq, seq]
        attention matrix per layer, which OOMs at long multi-turn contexts (~2 GB fp32 softmax
        per layer at seq~4700). SDPA computes the attention OUTPUT in O(seq) memory without
        building that matrix; we wrap the registered sdpa fn to additionally compute only the
        last query's attention row (O(seq)). Net peak attention memory is O(seq), not O(seq^2),
        so enable_eager_attention is no longer required before capture.
        """
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        n_heads, head_dim = self._head_dims()

        z_store: dict[int, torch.Tensor] = {}
        attn_store: dict[int, torch.Tensor] = {}
        handles = []

        # head_z: pre-W_O per-head values, captured from each o_proj input.
        projections = self._attn_output_projections()
        for li, proj in enumerate(projections):
            def make_pre_hook(idx):
                def pre_hook(module, args):
                    x = args[0]  # (batch, seq, n_heads * head_dim)
                    z_store[idx] = x[:, -1, :].detach().to("cpu", torch.float32)
                return pre_hook
            handles.append(proj.register_forward_pre_hook(make_pre_hook(li)))

        # attn_row: last-token attention pattern. Materializing the full (n_heads, seq, seq)
        # matrix per layer -- which eager attention / output_attentions does -- OOMs at long
        # multi-turn contexts (one layer's fp32 softmax is ~2 GB at seq~4700). Instead run the
        # forward through SDPA, which computes the attention OUTPUT in O(seq) memory without ever
        # building the full matrix, and wrap the registered sdpa fn to ALSO compute just the last
        # query's attention row (O(seq)). Net peak attention memory is O(seq), not O(seq^2).
        def _repeat_kv(h, n_rep):
            b, n_kv, s, d = h.shape
            if n_rep == 1:
                return h
            return h[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)

        real_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kwargs):
            attn_output, _ = real_sdpa(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
            k_rep = _repeat_kv(key, getattr(module, "num_key_value_groups", 1))  # (1, n_heads, seq_k, hd)
            scores = torch.matmul(query[:, :, -1:, :], k_rep.transpose(2, 3))     # (1, n_heads, 1, seq_k)
            if scaling is not None:
                scores = scores * scaling
            if attention_mask is not None:  # None => pure causal; last query sees all keys
                scores = scores + attention_mask[:, :, -1:, : scores.shape[-1]]
            attn_store[int(module.layer_idx)] = (
                torch.softmax(scores, dim=-1, dtype=torch.float32)[0, :, 0, :].to("cpu", torch.float32)
            )
            return attn_output, None

        prev_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = "sdpa"
        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_capture
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    output_hidden_states=True,
                    use_cache=False,  # single forward; KV cache is dead weight here
                )
        finally:
            del ALL_ATTENTION_FUNCTIONS["sdpa"]  # drop local override, restore global sdpa
            self.model.config._attn_implementation = prev_impl
            for h in handles:
                h.remove()

        if not getattr(outputs, "hidden_states", None):
            raise RuntimeError("Model did not return hidden_states.")

        embed = outputs.hidden_states[0][0, -1, :].float().cpu()  # (D,) residual entering layer 0
        hidden_states = outputs.hidden_states[1:]  # skip embedding layer
        resid = torch.stack([h[0, -1, :].float().cpu() for h in hidden_states])  # (L, D)

        n_layers = len(hidden_states)
        if len(attn_store) != n_layers:
            raise RuntimeError(
                f"Captured {len(attn_store)} attention rows but expected {n_layers}. The sdpa "
                "attention wrapper did not run for every layer -- check that the model dispatches "
                "attention through ALL_ATTENTION_FUNCTIONS['sdpa']."
            )

        expected = n_heads * head_dim
        head_z_rows = []
        for i in range(n_layers):
            flat = z_store[i].reshape(-1)
            if flat.numel() != expected:
                raise RuntimeError(
                    f"head_z width mismatch at layer {i}: o_proj input has {flat.numel()} "
                    f"elements but n_heads*head_dim = {n_heads}*{head_dim} = {expected}. "
                    "Per-head reshape would be wrong for this model; check _head_dims()."
                )
            head_z_rows.append(flat.view(n_heads, head_dim))
        head_z = torch.stack(head_z_rows)  # (L, n_heads, head_dim)

        attn_row = torch.stack([attn_store[i] for i in range(n_layers)])  # (L, n_heads, seq_k)

        return {
            "resid": resid,
            "embed": embed,
            "head_z": head_z,
            "attn_row": attn_row,
            "seq_len": int(inputs.input_ids.shape[1]),
        }

    def enable_eager_attention(self) -> None:
        """Force eager attention so output_attentions returns real patterns.

        The model was already loaded (likely with sdpa), so mutating only config is not
        reliable across transformers versions — some bind the implementation per-module at
        load. Prefer the supported runtime switch, then mutate every submodule config as a
        fallback. get_capture_messages still hard-raises if attentions come back None, so a
        version where none of this takes effect fails loud rather than silently.
        """
        target = "eager"
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter):
            try:
                setter(target)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"set_attn_implementation('{target}') failed: {e!r}; falling back to config mutation.")
        self.model.config._attn_implementation = target
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = target
        for module in self.model.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = target
            if hasattr(module, "_attn_implementation"):
                module._attn_implementation = target

    def ablate_weights(self, direction: torch.Tensor):

        pass
    
    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs) -> str:
        """
        Generates text from the model. 
        Used for evaluating whether the jailbreak worked.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model/Tokenizer not loaded.")

        inputs = self.prepare_inputs(prompt, **kwargs)
        if inputs is None:
            return ""
        
        # 2. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False  # Deterministic for reproducibility
                
            )
            
        # 3. Decode (Skipping the input prompt in the output)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    def batch_generate(self, prompts: list[str], max_new_tokens: int = 64, **kwargs) -> list[str]:
        """
        Batched deterministic generation. The returned list preserves input order.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model/Tokenizer not loaded.")
        if not prompts:
            return []

        inputs = self.prepare_batch_inputs(prompts, **kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_width = inputs.input_ids.shape[1]
        decoded: list[str] = []
        for row in outputs:
            generated_text = self.tokenizer.decode(
                row[prompt_width:],
                skip_special_tokens=True,
            )
            decoded.append(generated_text.strip())
        return decoded
