import torch 
from .language_models import LanguageModel
from .system_prompts import llama_sys
from utils.models_utils import ablate_weights, get_ablated_matrix

class Phi35Mini(LanguageModel):

    "A class to manage the microsoft/Phi-3.5-mini-instruct model."

    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct", device='cuda', system_prompt="You are a Helpful Assistant.", quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'mistral'
        self.hidden_dimension = 3072
        self.num_layer = 32
        self.refusal_token_id = [306] #'I'
        self.load_model()

    def _get_prompt(self, prompt):

        if self.system_prompt is not None:
            conv = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def _get_transformer_layers(self):
        return self.model.model.layers

    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]
    
    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
