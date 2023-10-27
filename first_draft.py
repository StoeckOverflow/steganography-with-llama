#%%
from llama_cpp import Llama
import json
import tqdm
from functools import reduce
import itertools
from typing import Callable

# Load model
llm = Llama(model_path="./llama-2-7b.Q5_K_M.gguf", seed=1337, verbose=False, logits_all=True, n_threads=4)
#%%
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
    def _get_valid_token(prompt: str) -> str:
        next_token_probs = list(llm(prompt, top_p=0, max_tokens=1, logprobs=2**bits_per_token, temperature= 0, top_k=1)["choices"][0]["logprobs"]["top_logprobs"][0])
        for a,b in itertools.combinations(next_token_probs, 2):
            if a in b or b in a:
                return [next_token_probs[0]]
        else:
            return next_token_probs
    
    return _get_valid_token

get_valid_token: Callable[[Llama], list[str]] = initialize_token_getter(llm)

# Load secret and news
with open("data/example_feed.json", "r") as f:
    infos = json.load(f)
secret, feed = infos["secret"], infos["feed"]
secret = secret[:5]
binary_secret = encode_secret(secret)[2:]
assert len(binary_secret) % 3 == 0
bits_per_token = 3
nr_prompt_words = 5
#%%
# Create stego feed from real feed for prompt
new_feed = " ".join(split_in_separators(feed[0])[:nr_prompt_words])

encode_token_sequence = []
output_file = open("llm_encoding_log.txt", "w")
# Iterate until message is contained in new_feed
for next_bits in tqdm.tqdm([binary_secret[i:i+bits_per_token] for i in range(0, len(binary_secret), bits_per_token)], "Message hidden: "):
    if len(next_bits) < 2:
        break
    while True:
        next_token_probs = get_valid_token(new_feed)
        if len(next_token_probs) > 1:
            chosen_ind = int(next_bits, 2)
            next_token = next_token_probs[chosen_ind]
            new_feed += next_token
            break
        new_feed += next_token_probs[0]
   
    encode_token_sequence.append(llm.tokenize(next_token.encode()))
    output_file.write(next_bits + "\n")
    output_file.write(repr(next_token_probs) + "\t" + repr(chosen_ind) + "\n")
    output_file.write(next_token + "\n" + new_feed + "\n")
    output_file.flush()

print(new_feed)
output_file.close()
with open("secret_raw_feed.txt", "w") as f:
    f.write(new_feed)
# %%
with open("secret_raw_feed.txt", "r") as f:
    new_feed = f.read()
# Decode
tmp_feed = " ".join(split_in_separators(new_feed)[:nr_prompt_words])

decoded_message = ""
output_file = open("llm_decoding_log.txt", "w")
i = len(tmp_feed)

decode_token_sequence = []
# Iterate until news feed is exhausted
pbar = tqdm.tqdm(total=len(new_feed))
while i < len(new_feed):
    while True:
        next_token_probs = get_valid_token(tmp_feed)
        if len(next_token_probs) > 1:
            max_len_next_token = max([len(token) for token in next_token_probs]) # Check the max length to check in message where the next token has to be found
            next_possible_tokens = [token for token in next_token_probs if token in new_feed[i:i+max_len_next_token]] # If there's still more than one that would fit
            next_possible_tokens = [sorted(next_possible_tokens, key= lambda x: new_feed[i:i+max_len_next_token].find(x))[0]] # Keep the one that occurrs first -> next token
            assert len(next_possible_tokens) == 1, f"{next_possible_tokens=} - {next_token_probs=} - {new_feed[i:i+max_len_next_token]}" # Assert that we only have one candidate left
            next_token = next_possible_tokens[0]
            chosen_ind = next_token_probs.index(next_token)
            tmp_feed += next_token
            i = len(tmp_feed)
            break
        tmp_feed += next_token_probs[0]
        i = len(tmp_feed)
    decode_token_sequence.append(llm.tokenize(next_token.encode()))
    next_bits = bin(chosen_ind)[2:]
    if len(next_bits) < 3:
        next_bits = (3-len(next_bits))*"0" + next_bits
    decoded_message += next_bits
    output_file.write(next_bits + "\n")
    output_file.write(repr(next_token_probs) + "\t" + repr(chosen_ind) + "\n")
    output_file.write(next_token + "\n" + tmp_feed + "\n")
    output_file.flush()
    pbar.n = i
    pbar.refresh()

print(binary_secret)
print(decoded_message)
print(secret)
print(decode_secret(decoded_message))
assert binary_secret.startswith(decoded_message)
# %%
