import torch

from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Mixtral8x7b_Instruct(LanguageModel):
    """A class to manage the 'mistralai/Mixtral-8x7B-Instruct-v0.1' model."""

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device="cuda",
        system_prompt=None,
        quantization_config=None,
    ):
        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        super().__init__(
            model_name,
            system_prompt,
            device,
            quantization_config,
            model_dtype=model_dtype,
        )
        self.template_name = "mistral"
        self.hidden_dimension = self.model.config.hidden_size
        self.num_layer = len(self._get_transformer_layers())
        self.refusal_token_id = self.tokenizer.encode("I", add_special_tokens=False)
        self.tokenizer.padding_side = "left"

    def _get_prompt(self, prompt):
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

            moe = getattr(block, "block_sparse_moe", None)
            if moe is not None and hasattr(moe, "experts"):
                for expert in moe.experts:
                    expert.w2.weight.data = get_ablated_matrix(
                        expert.w2.weight.data.T,
                        direction,
                    ).T
                continue

            if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
                block.mlp.down_proj.weight.data = get_ablated_matrix(
                    block.mlp.down_proj.weight.data.T,
                    direction,
                ).T

        print("✓ Weights ablated.")
