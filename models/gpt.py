import torch
import os
from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer


OSS_REFUSAL_TOKS = [40]


class GPT20b(LanguageModel):
    """A class to manage the 'openai/gpt-oss-20b' model."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device='cuda',
        system_prompt=None,
        quantization_config=None,
        reasoning_effort: str = "low",
    ):
        self.reasoning_effort = reasoning_effort
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = "gpt-oss"
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())
        self.refusal_token_id = OSS_REFUSAL_TOKS

    def load_model(self):
        """
        GPT-OSS override: load without quantization arguments.
        """
        if self.model is None or self.tokenizer is None:
            token = os.environ.get("HF_TOKEN")
            print("Downloading/Loading model components...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=token,
                trust_remote_code=True,
            )
            self.tokenizer.padding_side = "left"

            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map=self.device,
                dtype=model_dtype,
                trust_remote_code=True,
            )
            # Keep a consistent dtype across modules (important for GPT-OSS MoE kernels).
            self.model.to(dtype=model_dtype)
            self.model.eval()
            self.model.requires_grad_(False)

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = "left"
                    self.tokenizer.pad_token = "<|extra_0|>"

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Model loaded successfully.")

    def _get_prompt(self, prompt):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
        )

    def _get_transformer_layers(self):
        return self.model.model.layers

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(
            self.model.model.embed_tokens.weight.data, direction
        )

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(
                block.self_attn.o_proj.weight.data.T, direction
            ).T

            if hasattr(block.mlp, "experts"):
                for expert in block.mlp.experts:
                    expert.down_proj.weight.data = get_ablated_matrix(
                        expert.down_proj.weight.data.T, direction
                    ).T
            else:
                block.mlp.down_proj.weight.data = get_ablated_matrix(
                    block.mlp.down_proj.weight.data.T, direction
                ).T

        print("✓ Weights ablated.")
