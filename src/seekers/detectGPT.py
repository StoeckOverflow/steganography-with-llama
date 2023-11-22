import numpy as np
import re
import tqdm
import functools
import torch
import math
from llama_cpp import Llama
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import sys

'Static Variables'
DEVICE = 'cpu'
local_model_path_to_t5_3B = 'resources/T5_3B'
model_path = '/resources/llama-2-7b.Q5_K_M.gguf'
base_model= Llama(model_path)
base_tokenizer= base_model.tokenizer()
mask_model= T5ForConditionalGeneration.from_pretrained(local_model_path_to_t5_3B)
mask_tokenizer= T5Tokenizer.from_pretrained(local_model_path_to_t5_3B)

pattern = re.compile(r"<extra_id_\d+>")
def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, mask_top_p=1.0):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)# outputs.shape: torch.Size([20, 57])
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def perturb_texts_(texts, span_length=5, pct=0.3, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
        if attempts > 10:
            break
    

    return perturbed_texts

def perturb_texts(texts, chunk_size=20, ceil_pct=False):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], ceil_pct=ceil_pct))
    return outputs

def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])

def get_ll(text):
    tokenized_text = base_tokenizer.encode(text, add_bos=True, special=False)
    
    log_likelihood = 0.0

    for i in range(len(tokenized_text)):
        base_model.eval(tokenized_text[:i+1])
        logits = base_model._scores[i, :].tolist()
        log_probs = base_model.logits_to_logprobs(logits)
        token_log_prob = log_probs[tokenized_text[i]]
        log_likelihood += token_log_prob

    return log_likelihood

def get_lls(texts):
    return [get_ll(text) for text in texts]

def run_DetectGPT_single_text(text, n_perturbations=5):
    torch.manual_seed(42)
    np.random.seed(42)

    original_ll = get_ll(text)
    perturbed_text = perturb_texts(text, n_perturbations)
    perturbed_ll = get_lls(perturbed_text)

    mean_perturbed_ll = np.mean([i for i in perturbed_ll if not math.isnan(i)])
    std_perturbed_ll = np.std([i for i in perturbed_ll if not math.isnan(i)]) if len([i for i in perturbed_ll if not math.isnan(i)]) > 1 else 1

    prediction_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll

    return prediction_score

def run_DetectGPT_feed(feed, n_perturbations=5, threshold = 0.2):
    torch.manual_seed(42)
    np.random.seed(42)
    prediction_scores = []

    perturb_fn = functools.partial(perturb_texts, n_perturbations=n_perturbations)

    for text in feed:
        original_ll = get_ll(text)

        perturbed_text = perturb_fn(text)
        perturbed_ll = get_lls(perturbed_text)

        mean_perturbed_ll = np.mean([i for i in perturbed_ll if not math.isnan(i)])
        std_perturbed_ll = np.std([i for i in perturbed_ll if not math.isnan(i)]) if len([i for i in perturbed_ll if not math.isnan(i)]) > 1 else 1

        prediction_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll
        prediction_scores.append(prediction_score)

    decision = any(score > threshold for score in prediction_scores)

    return {"result": decision}

def calculate_prediction_scores(training_feeds, n_perturbations=5):
    scores = []
    perturb_fn = functools.partial(perturb_texts, n_perturbations=n_perturbations)

    for feed in training_feeds:
        for text in feed:
            original_ll = get_ll(text)
            perturbed_text = perturb_fn(text)
            perturbed_ll = get_lls(perturbed_text)

            mean_perturbed_ll = np.mean([i for i in perturbed_ll if not math.isnan(i)])
            std_perturbed_ll = np.std([i for i in perturbed_ll if not math.isnan(i)]) if len([i for i in perturbed_ll if not math.isnan(i)]) > 1 else 1

            prediction_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll
            scores.append(prediction_score)

    return scores

def determine_threshold(scores, percentile=95):
    return np.percentile(scores, percentile)

def process_feeds_and_determine_threshold(feeds, n_perturbations=5, percentile=95):
    scores = calculate_prediction_scores(feeds, n_perturbations)
    threshold = determine_threshold(scores, percentile)
    return threshold

if __name__ == "__main__":
    input_data = json.load(sys.stdin)
    feed = input_data["feed"]
    #threshold = process_feeds_and_determine_threshold(feeds,, n_perturbations=5, percentile=95)
    #result = run_DetectGPT_feed(feed, threshold=threshold)
    result = run_DetectGPT_feed(feed)
    json.dump(result, sys.stdout)