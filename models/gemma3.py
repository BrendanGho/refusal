import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Gemma3_12b(LanguageModel):
    """A class to manage the 'google/gemma-3-12b-it' model."""
    TEXT_BACKBONE_PATH = "model.language_model"

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
    ):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self._normalize_root_config_for_text_backbone()
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())

    def _get_text_backbone(self):
        current = self.model
        try:
            for attr in self.TEXT_BACKBONE_PATH.split("."):
                current = getattr(current, attr)
            return current
        except AttributeError as exc:
            raise AttributeError(
                f"Expected Gemma-3 text backbone at `model.{self.TEXT_BACKBONE_PATH}`."
            ) from exc

    def _normalize_root_config_for_text_backbone(self):
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is None:
            return

        for attr in ("num_hidden_layers", "hidden_size", "vocab_size"):
            if not hasattr(self.model.config, attr) and hasattr(text_cfg, attr):
                setattr(self.model.config, attr, getattr(text_cfg, attr))

    def load_model(self):
        """
        Gemma-3 override: prefer BF16 to avoid FP16 instabilities on some layers.
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

            model_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map=self.device,
                dtype=model_dtype,
                trust_remote_code=True,
            )
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
            self._normalize_root_config_for_text_backbone()
            print("Model loaded successfully.")

    def _get_prompt(self, prompt=""):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _get_transformer_layers(self):
        return self._get_text_backbone().layers

    def ablate_weights(self, direction: torch.Tensor):
        text_backbone = self._get_text_backbone()
        text_backbone.embed_tokens.weight.data = get_ablated_matrix(
            text_backbone.embed_tokens.weight.data,
            direction,
        )

        for block in text_backbone.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(
                block.self_attn.o_proj.weight.data.T,
                direction,
            ).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(
                block.mlp.down_proj.weight.data.T,
                direction,
            ).T

        print("✓ Weights ablated.")
