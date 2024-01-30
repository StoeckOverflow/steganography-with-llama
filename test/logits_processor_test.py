#%%
from typing import Callable, List, Union
from llama_cpp import Llama
from transformers import LogitsProcessor, LogitsProcessorList
import torch
import numpy as np

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
            cur_len = len(self.detokenize(input_ids))
        elif self._tokens_or_chars == "tokens":
            cur_len = input_ids.shape[-1]

        if cur_len < self.min_completion_length:
            for i in self.eos_token_id:
                scores[:, i] = 0
        return scores
    
    def update_min_completion_length(self, new_min_completion_length: int):
        self.min_completion_length = new_min_completion_length

llm = Llama(model_path="resources/llama-2-7b.Q5_K_M.gguf", seed=1337, verbose=False, logits_all=True, n_threads=None, use_mlock=False, n_gpu_layers=0)
#%%
prompt = "Testing this prompt is"
logits_processor = LogitsProcessorList([
    ExtendCompletionLength(10, llm.detokenize, eos_token_id=llm.token_eos()),
])
#%%
gen = llm(prompt, top_p=0, max_tokens=0, logprobs=2**2, temperature= 0, top_k=1, logits_processor=logits_processor, repeat_penalty=1.5)["choices"][0]
#%%
print(prompt + gen["text"])
print(len(gen["logprobs"]["tokens"]))
# %%
gen["finish_reason"]
# %%
