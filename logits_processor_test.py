#%%
from llama_cpp import Llama
from transformers import ExponentialDecayLengthPenalty, LogitsProcessorList
import torch
import numpy as np

class HomebrewLogitsProcessor(ExponentialDecayLengthPenalty):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(scores.shape) == 1:
            scores = scores.reshape(1,-1)
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            for i in self.eos_token_id:
                penalty_idx = cur_len - self.regulation_start
                # To support negative logits we compute the penalty of the absolute value and add to the original logit
                scores[:, i] = scores[:, i] + np.abs(scores[:, i]) * (pow(self.regulation_factor, penalty_idx) - 1)
        return scores

llm = Llama(model_path="resources/llama-2-7b.Q5_K_M.gguf", seed=1337, verbose=False, logits_all=True, n_threads=None, use_mlock=False)
#%%
prompt = "Testing this prompt is"
logits_processor = LogitsProcessorList([
    HomebrewLogitsProcessor((0, 100.), [*llm.tokenize(bytes(".", "utf-8"))[1:]], len(prompt)),
])
llm.token_eos()
gen = llm(prompt, top_p=0, max_tokens=0, logprobs=2**2, temperature= 0, top_k=1, logits_processor=logits_processor, stop=["."])["choices"][0]
gen
# %%
gen["finish_reason"]
# %%
