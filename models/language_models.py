import torch
import os
import functools
from abc import ABC, abstractmethod
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
                torch_dtype=torch.float16 if bnb_config is None else None,
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
