import torch 
from .language_models import LanguageModel
from .system_prompts import llama_sys
from utils.models_utils import ablate_weights, get_ablated_matrix


class Llama2_7b(LanguageModel):
    """
    A class to manage the 'meta-llama/Llama-2-7b-chat-hf' model.

    """
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", system_prompt=llama_sys, device='cuda', quantization_config=None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'meta-llama/Llama-2-7b-chat-hf'.
            system_prompt: System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config: Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
        """

        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 4096
        self.num_layer = 32

    def _get_prompt(self, prompt):
        """
        Formats the prompt using FastChat conversation template.
        If target is provided, formats it as the assistant's answer.
        
        Args:
            prompt (str): The input prompt
            target (str, optional): The target response to format as assistant's answer
        """
        if self.system_prompt is not None:
            return f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n{prompt} [/INST]"
        else:
            return f"[INST] {prompt} [/INST]"

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

