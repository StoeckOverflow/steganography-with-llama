from .hider import Hider
from ..codecs import ProbabilityOrderCodec
from llama_cpp import Llama

class ProbabilityOrderHider(Hider):
    
    def __init__(self, seed=None):
        llm = Llama(model_path="llama-2-7b.Q5_K_M.gguf", seed=seed, verbose=False, logits_all=True, n_threads=None, use_mlock=True)
        super().__init__(ProbabilityOrderCodec, llm)