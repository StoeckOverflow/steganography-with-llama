import time
from typing import Callable, Iterable, List, Optional, Union
from llama_cpp import Llama
from transformers import LogitsProcessor, LogitsProcessorList, TypicalLogitsWarper, NoRepeatNGramLogitsProcessor
import torch
import numpy as np

__all__ = [
    "CustomLogitsProcessor",
    "ExtendCompletionLength",
    "WhitelistAndLength",
]

class CustomLogitsProcessor(LogitsProcessorList):
    def __init__(
            self,
            min_completion_length: int = 0,
            tokenizer: Callable[[np.ndarray], int] = None,
            detokenizer: Callable[[np.ndarray], str] = None,
            weird_chars: Optional[Union[List[bytes], List[int]]] = None,
            eos_token_id: Union[List[int], int] = 2,  # 2 is the default eos token id
            penalize_ngrams: Optional[List[int]] = None,
            ):
        super().__init__()
        self._log_interesting_tokens_x = []
        self._avoid_weird = AvoidGeneratingWeirdTokens(tokenizer, weird_chars)
        self.append(self._avoid_weird)
        self._log_interesting_tokens = np.empty((4096, len(eos_token_id) + len(self._avoid_weird.weird_chars)))
        self._log_interesting_tokens_x.extend(self._avoid_weird.weird_chars)
        
        if penalize_ngrams is not None:
            for ngram in penalize_ngrams:
                self.append(NoRepeatNGramLogitsProcessor(ngram))
        self._extend_completion = ExtendCompletionLength(min_completion_length, detokenizer, eos_token_id)
        self.append(self._extend_completion)
        self._log_interesting_tokens_x.extend(eos_token_id if isinstance(eos_token_id, Iterable) else [eos_token_id])
        self._log_interesting_tokens_x = np.array(self._log_interesting_tokens_x)
        self.i = 0
        # self.append(TypicalLogitsWarper())
    
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> torch.FloatTensor:
        start = time.time()
        if len(input_ids.shape) < 2:
            input_ids = torch.LongTensor(input_ids).view(1, -1)
            scores = torch.FloatTensor(scores).view(1, -1)
        out = super().__call__(input_ids, scores, **kwargs)
        self._log_interesting_tokens[self.i] = out.squeeze()[self._log_interesting_tokens_x]
        self.i += 1
        return out
    
    def update_prompt_length(self, *args, **kwargs):
        self._extend_completion.update_prompt_length(*args, **kwargs)

    def update_min_completion_length(self, *args, **kwargs):
        self._extend_completion.update_min_completion_length(*args, **kwargs)


class AvoidGeneratingWeirdTokens(LogitsProcessor):
    def __init__(self,
                 tokenizer: Callable[[np.ndarray], int] = None,
                 weird_chars: Optional[Union[List[str], List[int]]] = None,
                 ):
        if weird_chars is None:
            weird_chars = []
        elif isinstance(weird_chars[0], bytes):
            weird_chars = [tokenizer(weird_char, add_bos=False)[0] for weird_char in weird_chars]
        self.weird_chars = weird_chars

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in self.weird_chars:
            scores[:, i] = 0
        return torch.FloatTensor(scores).view(1, -1)

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

    @staticmethod
    def split_in_separators(txt, seps = None):
        if seps is None:
            seps = [" ", ",", ":", ";", ".", "-", "_", "/", "\\"]
        default_sep = seps[0]

        # we skip seps[0] because that's the default separator
        for sep in seps[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip() for i in txt.split(default_sep) if len(i) > 0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(scores.shape) == 1:
            scores = scores.reshape(1,-1)

        if self._tokens_or_chars == "chars":
            cur_len = len(self.detokenize(input_ids))
        elif self._tokens_or_chars == "tokens":
            cur_len = input_ids.shape[-1]
        elif self._tokens_or_chars == "words":
            cur_len = self.split_in_separators(self.detokenize(input_ids), [" "])

        if cur_len < self.min_completion_length + self.prompt_length:
            for i in self.eos_token_id:
                scores[:, i] = 0
        else:
            pass
            print("cur_len >= min_completion_length")
        return torch.tensor(scores).view(1,-1)
    
    def update_prompt_length(self, prompt_tokens: List[int]):
        self.prompt_length = len(prompt_tokens)

    def update_min_completion_length(self, new_min_completion_length: int):
        self.min_completion_length = new_min_completion_length


class WhitelistedTokens(LogitsProcessor):
    def __init__(self, total_tokens: int, whitelist_inds: np.ndarray, device: torch.device = torch.device("cpu")):
        """
        Set all non-whitelisted tokens to -inf

        param: whitelist: inds of whitelisted tokens
        """
        self.whitelist = np.zeros(total_tokens, dtype=bool)
        self.whitelist[np.array(whitelist_inds, dtype=int)] = True
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[~self.whitelist] = -np.inf
        return scores
    

class WhitelistAndLength(LogitsProcessorList):
    def __init__(
        self,
        whitelist_inds: np.ndarray,
        min_completion_length: int = 0,
        detokenizer: Callable[[np.ndarray], str] = None,
        eos_token_id: Union[List[int], int] = 2,  # 2 is the default eos token id,
        total_tokens: int = 32_000,
    ):
        self._extend_completion = ExtendCompletionLength(min_completion_length, detokenizer, eos_token_id)
        self.append(WhitelistedTokens(total_tokens, whitelist_inds))
        self.append(self._extend_completion)
        self.append(TypicalLogitsWarper())

    def update_prompt_length(self, *args, **kwargs):
        self._extend_completion.update_prompt_length(*args, **kwargs)

    def update_min_completion_length(self, *args, **kwargs):
        self._extend_completion.update_min_completion_length(*args, **kwargs)