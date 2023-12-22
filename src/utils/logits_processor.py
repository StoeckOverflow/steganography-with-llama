import torch
import numpy as np
from typing import Callable, Union, List
from transformers import LogitsProcessor

__all__ = ["ExtendCompletionLength"]

class ExtendCompletionLength(LogitsProcessor):
    def __init__(
        self,
        min_completion_length: int = 0,
        detokenizer: Callable[[np.ndarray], str] = None,
        eos_token_id: Union[List[int], int] = 2,  # 2 is the default eos token id
    ):
        self.min_completion_length = min_completion_length  # +2 because of the start and end token
        self._tokens_or_chars = "chars" if detokenizer is not None else "tokens"
        self.detokenize = detokenizer
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(scores.shape) == 1:
            scores = scores.reshape(1,-1)

        if self._tokens_or_chars == "chars":
            cur_len = len(self.detokenize(input_ids))
        elif self._tokens_or_chars == "tokens":
            cur_len = input_ids.shape[-1]

        if cur_len < self.min_completion_length + self.prompt_length:
            for i in self.eos_token_id:
                scores[:, i] = 0
        else:
            print("cur_len >= min_completion_length")
        return scores
    
    def update_prompt_length(self, prompt_tokens: List[int]):
        self.prompt_length = len(prompt_tokens)

    def update_min_completion_length(self, new_min_completion_length: int):
        self.min_completion_length = new_min_completion_length