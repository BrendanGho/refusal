import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Phi4_15b(LanguageModel):
    """A class to manage the 'microsoft/Phi-4-reasoning-vision-15B' model."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-reasoning-vision-15B",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
    ):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())
        self.refusal_token_id = [100348]

    def load_model(self):
        """
        Phi-4 reasoning vision override: load with eager attention.
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
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map=self.device,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self.model.eval()
            self.model.requires_grad_(False)

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                else:
                    raise ValueError("The tokenizer does not have a pad token or an eos token to use as a pad token.")

            self.tokenizer.padding_side = "left"
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Model loaded successfully.")

    def _get_prompt(self, prompt=""):
        if getattr(self.tokenizer, "chat_template", None):
            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        raise NotImplementedError("Prompt construction for non-chat-template tokenizers is not implemented.")

    def _get_transformer_layers(self):
        return self.model.model.layers

    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(
            self.model.model.embed_tokens.weight.data,
            direction,
        )

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(
                block.self_attn.o_proj.weight.data.T,
                direction,
            ).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(
                block.mlp.down_proj.weight.data.T,
                direction,
            ).T

        print("✓ Weights ablated.")
