__version__ = "0.0.1"

from huggingface.transformers import LogitsProcessor
import torch

class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    """

    def __init__(self, min_length: int, eos_token_id: int):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass
