__version__ = "0.0.1"

from huggingface.transformers import LogitsProcessor
import torch

class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    """

    def __init__(self, allowed_words_ids: List[List[int]], eos_token_id: int):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def _tokens_match() -> bool:
        pass

    def _calc_allowed_words_ids(self, prev_input_ids: List[List[int]]) -> Iterable[int]:
        allowed_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            allowed_tokens_slice = []
            for allowed_token_seq in self.allowed_words_id_length_greater_than_1:
                if self._tokens_match(prev_input_ids_slice, allowed_token_seq[:-1]):
                    allowed_tokens_slice.append(allowed_token_seq[-1])

            allowed_tokens.append(allowed_tokens_slice)

        return allowed_tokens

    def _set_scores_to_neg_inf_unless_allowed(self, scores: torch.Tensor, allowed_tokens: List[List[int]]) -> torch.Tensor:
        pass