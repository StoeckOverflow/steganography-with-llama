from .hider import Hider
from ..codecs.synonym_hash_codec import SynonymHashCodec
from llama_cpp import Llama

class SynonymHider(Hider):
    
    def __init__(self, seed=1337, disable_tqdm=True):
        llm = Llama(model_path="llama-2-7b.Q5_K_M.gguf", seed=seed, verbose=False, logits_all=True, n_threads=None, use_mlock=True)
        super().__init__(SynonymHashCodec, llm, disable_tqdm)