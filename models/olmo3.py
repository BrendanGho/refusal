import torch

from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Olmo3_7b(LanguageModel):
    """A class to manage the 'allenai/Olmo-3-7B-Instruct' model."""

    def __init__(
        self,
        model_name: str = "allenai/Olmo-3-7B-Instruct",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
    ):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())

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
