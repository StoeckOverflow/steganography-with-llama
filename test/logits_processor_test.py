#%%
import time
from typing import Callable, Iterable, List, Optional, Union
from llama_cpp import Llama
from transformers import LogitsProcessor, LogitsProcessorList, TypicalLogitsWarper, NoRepeatNGramLogitsProcessor
import torch
import numpy as np

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
        self._log_interesting_tokens = np.empty((4096, len(eos_token_id) + len(weird_chars)))
        self._log_interesting_tokens_x = []
        self.append(AvoidGeneratingWeirdTokens(tokenizer, weird_chars))
        self._log_interesting_tokens_x.extend(self[-1].weird_chars)
        # self.append(TypicalLogitsWarper())
        if penalize_ngrams is not None:
            for ngram in penalize_ngrams:
                self.append(NoRepeatNGramLogitsProcessor(ngram))
        self.append(ExtendCompletionLength(min_completion_length, detokenizer, eos_token_id))
        self._log_interesting_tokens_x.extend(eos_token_id if isinstance(eos_token_id, Iterable) else [eos_token_id])
        self._log_interesting_tokens_x = np.array(self._log_interesting_tokens_x)
        self.i = 0
    
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> torch.FloatTensor:
        start = time.time()
        if len(input_ids.shape) < 2:
            input_ids = torch.LongTensor(input_ids).view(1, -1)
            scores = torch.FloatTensor(scores).view(1, -1)
        out = super().__call__(input_ids, scores, **kwargs)
        self._log_interesting_tokens[self.i] = out.squeeze()[self._log_interesting_tokens_x]
        self.i += 1
        return out
    


class AvoidGeneratingWeirdTokens(LogitsProcessor):
    def __init__(self,
                 tokenizer: Callable[[np.ndarray], int] = None,
                 weird_chars: Optional[Union[List[str], List[int]]] = None,
                 ):
        if weird_chars is None:
            weird_chars = []
        if isinstance(weird_chars[0], bytes):
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
        self.min_completion_length = min_completion_length
        self._tokens_or_chars = "chars" if detokenizer is not None else "tokens"
        self.detokenize = detokenizer
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(scores.shape) == 1:
            scores = scores.reshape(1,-1)

        if self._tokens_or_chars == "chars":
            cur_len = len(self.detokenize(input_ids.squeeze()))
        elif self._tokens_or_chars == "tokens":
            cur_len = input_ids.shape[-1]

        if cur_len < self.min_completion_length:
            for i in self.eos_token_id:
                scores[:, i] = 0
        return scores
    
    def update_min_completion_length(self, new_min_completion_length: int):
        self.min_completion_length = new_min_completion_length

llm = Llama(model_path="resources/llama-2-7b.Q5_K_M.gguf", n_ctx=4096, seed=1337, verbose=False, logits_all=True, n_threads=None, use_mlock=False, n_gpu_layers=-1)
# %%
replace_from_unicode = {
    "\u2013": "-",  # Weird length middle line
    "\u2014": "-",  # Weird length middle line
    "\u2018": "'",  # Opening single quote
    "\u2019": "'",  # Closing single quote (and apostrophe)
    "\u201c": '"',  # Opening double quotes
    "\u201d": '"',  # Closing double quotes
    "\u2022": "*",  # Bullet point
    "\u2026": ",",  # Horizontal ellipsis
}
replace_to_unicode_first = {
    "-": ["\u2013", "\u2014", "-"],  # Just randomly choose I guess - [392, 78, 1961]
    "'": ["\u2018", "\u2019"],  # Always replace, care for order - [31, 1948]
    '"': ["\u201c", "\u201d"],  # Always replace, care for order - [1198, 1147]
    "*": ["\u2022", "*"],  # When talking, might be used to censor - [8, 6]
    ",": [",", "\u2026"],  # When talking, might be used as pause - [6851, 19]
}
#%%
prompt = "Testing this prompt is"
logits_processor = CustomLogitsProcessor(
    min_completion_length=10,
    tokenizer=llm.tokenize,
    detokenizer=llm.detokenize,
    weird_chars=[c.encode() for c in replace_from_unicode.keys()],
    eos_token_id=[2, llm.tokenize(b"!", add_bos=False)[0]],
    # penalize_ngrams=list(range(5, 10))
)
# logits_processor = LogitsProcessorList([
#     ExtendCompletionLength(10, llm.detokenize, eos_token_id=llm.token_eos()),
# ])
#%%
gen = llm(prompt, seed=42, top_p=0, max_tokens=0, logprobs=2**2, temperature=0, top_k=1, logits_processor=logits_processor, repeat_penalty=1.5)["choices"][0]
#%%
print(prompt + gen["text"])
print(len(gen["logprobs"]["tokens"]))
# %%
gen["finish_reason"]
# %%

# %%
