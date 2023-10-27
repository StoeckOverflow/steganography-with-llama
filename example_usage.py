import json
from llama_cpp import Llama
from encoders.probability_order_encoder import ProbabilityOrderEncoder
from utils import encode_secret, decode_secret

llm = Llama(model_path="./llama-2-7b.Q5_K_M.gguf", seed=1337, verbose=False, logits_all=True, n_threads=4)

# Load secret and news
with open("data/example_feed.json", "r") as f:
    infos = json.load(f)
secret, feed = infos["secret"], infos["feed"]
secret = secret[:5]
binary_secret = encode_secret(secret)[2:]
bits_per_token = 3
assert len(binary_secret) % bits_per_token == 0 # For now if these don't match we can't hide the whole secret
nr_prompt_words = 5

poe = ProbabilityOrderEncoder(llm)
doctored_news_feed = poe.encode_news_feed(feed, binary_secret, bits_per_token, nr_prompt_words=nr_prompt_words)
extracted_secret = poe.decode_news_feed(feed, ...)