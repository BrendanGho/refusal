import torch 
from .language_models import LanguageModel
from utils.models_utils import get_ablated_matrix


class Qwen2_32b(LanguageModel):
    
    """A class to manage the 'Qwen/Qwen2.5-32B-Instruct' model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct", device='cuda', system_prompt=None, quantization_config=None):
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        super().__init__(model_name, system_prompt, device, quantization_config, model_dtype=model_dtype)
        self.hidden_dimension = 5120
        self.num_layer = 64
        self.load_model()
        self.tokenizer.padding_side = 'left'

    def _get_prompt(self, prompt=''):
        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def _get_transformer_layers(self):
        return self.model.model.layers

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   
