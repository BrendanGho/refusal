# all models from here
from models.llama2 import (
    Llama2_7b,
)
from models.llama3 import (
    Llama3_8b,
)
from models.qwen2 import (
    Qwen2_32b,
)   
from models.qwen35 import (
    Qwen35_9b,
) 
from models.mistralRR import (
    Mistral7B_RR,
)   
from models.gemma3 import (
    Gemma3_12b,
)
from models.llama32 import (
    Llama32_3b
)
from models.mistral_instruct import (
    Mistral7b_Instruct
)
try:
    from models.ministral3 import (
        Ministral3_14b
    )
except ImportError as exc:
    _MINISTRAL3_IMPORT_ERROR = exc

    class Ministral3_14b:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Ministral3_14b is unavailable in this environment because "
                "`transformers.Mistral3ForConditionalGeneration` could not be imported."
            ) from _MINISTRAL3_IMPORT_ERROR
from models.phi3mini import (
    Phi35Mini
)
from models.gpt import (
    GPT20b
)
from models.mixtral import (
    Mixtral8x7b_Instruct
)
from models.olmo3 import (
    Olmo3_7b
)
from models.phi4 import (
    Phi4_15b
)
from models.deepseek_r1 import (
    DeepSeekR1_8b
)

# this is the list of all the models that will be imported when from clf_refusal.models import * is called
__all__ = [
    "Llama2_7b",
    "Llama3_8b",
    "Qwen2_32b",
    "Qwen35_9b",
    "Mistral7B_RR",
    "Gemma3_12b",
    "Llama32_3b",
    "Mistral7b_Instruct",
    "Ministral3_14b",
    "Phi35Mini",
    "GPT20b",
    "Mixtral8x7b_Instruct",
    "Olmo3_7b",
    "Phi4_15b",
    "DeepSeekR1_8b"

]
