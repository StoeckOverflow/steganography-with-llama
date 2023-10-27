from llama_cpp import Llama
import itertools
from typing import Callable

encode_secret = lambda secret: bin(int.from_bytes(secret.encode(), "little"))

def decode_secret(binary_secret, max_len = 100, endian = "little"):
    for i in range(max_len):
        try:
            return int(binary_secret, 2).to_bytes(i, endian).decode()
        except:
            pass

def find_first_few_whitespace_indexes(string: str, num_of_whitsp: int = 5):
    inds = []
    counter = 0
    str_iter = (char for char in string)
    next_token = next(str_iter, None)
    while len(inds) < num_of_whitsp + 1 and next_token is not None:
        if next_token == " ":
            inds.append(counter)
        counter += 1
        next_token = next(str_iter, None)
    return inds

def split_in_separators(txt, seps = None):
    if seps is None:
        seps = [" ", ",", ":", ";", ".", "-", "_", "/", "\\"]
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep) if len(i) > 0]

def initialize_token_getter(llm: Llama) -> Callable:
    def _get_valid_token(prompt: str, bits_per_token: int = 3) -> str:
        next_token_probs = list(llm(prompt, top_p=0, max_tokens=1, logprobs=2**bits_per_token, temperature= 0, top_k=1)["choices"][0]["logprobs"]["top_logprobs"][0])
        if bits_per_token > 3: # generate stop token
            return next_token_probs
        for a,b in itertools.combinations(next_token_probs, 2):
            if a in b or b in a:
                return [next_token_probs[0]]
        else:
            return next_token_probs
    
    return _get_valid_token

def update_news_feed(current_news_feed: str, next_token: str, max_len: int) -> str:
    if len(current_news_feed) < 450:
        return current_news_feed + next_token