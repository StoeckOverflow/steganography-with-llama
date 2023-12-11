from llama_cpp import Llama
import itertools
from typing import Callable
import numpy as np
from tqdm import tqdm
from typing import List

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
        if bits_per_token > 3: # If bits per token is larger than 3, we don't need to check for duplicates, since we want to generate the stopping token
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
    

'Baseline Metrics, which may be useful for detection'

'Probability of tokenized text'
def get_probabilities(llm: Llama, text: str):
    'get probabilites estimation of tokenized text'
    tokenized_text = llm.tokenizer().encode(text)
    llm.reset()
    llm.eval(tokenized_text)
    logits = np.array(llm._scores)
    probabilities = softmax(logits)

    return probabilities

'Entropy'  
def get_entropy(llm: Llama, text: str):
    tokenized = llm.tokenizer().encode(text)

    llm.reset()
    llm.eval(tokenized)

    logits = np.array(llm._scores)
    softmax_logits = softmax(logits)
    log_softmax_logits = np.log(softmax_logits)
    neg_entropy = softmax_logits * log_softmax_logits
    entropy = -neg_entropy.sum(-1)

    return np.mean(entropy)

'Rank'
def get_rank(llm: Llama, text: str):
    tokenized = llm.tokenizer().encode(text)

    llm.reset()
    llm.eval(tokenized)

    logits = np.array(llm._scores)

    ranks = []
    for i, logit in enumerate(logits):
        sorted_indices = np.argsort(logit)[::-1]  # Descending sort
        token_id = tokenized[i]
        token_rank = np.where(sorted_indices == token_id)[0][0]
        ranks.append(token_rank + 1)  # Convert to 1-indexed rank

    return np.mean(ranks)

'Log Likelihood'
def compute_embeddings_llama(llm: Llama, text) -> List[float]:
    return llm.embed(text.encode('utf-8'))
    
def get_ll(llm: Llama, text: str):
    tokenized_text = llm.tokenizer().encode(text)
    llm.reset()
    llm.eval(tokenized_text)
    logits = np.array(llm._scores)
    softmax_logits = softmax(logits)
    log_likelihood = 0.0
    for i, token_id in enumerate(tokenized_text):
        prob = softmax_logits[i, token_id]
        log_likelihood += np.log(prob)

    return log_likelihood

def get_lls(llm: Llama, texts: [str], disable_tqdm):
    lls = []
    for text in tqdm(texts, desc='Log Likelihood for Text Estimation', disable=disable_tqdm):
        lls.append(get_ll(llm,text))
    return lls

'Perplexity'
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_perplexity(llm: Llama, text: str):
    tokenized_text = llm.tokenizer().encode(text)
    llm.reset()
    llm.eval(tokenized_text)
    logits = np.array(llm._scores)
    softmax_logits = softmax(logits)
    log_likelihood = 0.0
    for i, token_id in enumerate(tokenized_text):
        prob = softmax_logits[i, token_id] if softmax_logits[i, token_id] != 0 else softmax_logits[i, token_id]+1e-10
        log_likelihood += np.log(prob)

    avg_neg_log_likelihood = -log_likelihood / len(tokenized_text)

    return np.exp(avg_neg_log_likelihood)

