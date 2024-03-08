from ..utils.llama_utils import get_ll, get_lls, compute_embeddings_llama
from .seeker import Seeker
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import re
from tqdm import tqdm
import torch
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

class detectGPTseeker(Seeker):

    def __init__(self, disable_tqdm) -> None:
        super().__init__(disable_tqdm)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.mask_model = T5ForConditionalGeneration.from_pretrained('resources/t5-large', local_files_only=True).to(self.DEVICE)
        #self.mask_tokenizer= T5Tokenizer.from_pretrained('resources/t5-large', local_files_only=True, legacy=False)
        self.pattern = re.compile(r"<extra_id_\d+>")
        
    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False, buffer_size=1) -> str:
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

    def count_masks(self, texts) -> [int]:
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, texts, mask_top_p=1.0) -> [str]:
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.DEVICE)
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)# outputs.shape: torch.Size([20, 57])
        test =  self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return test
    
    def replace_masks_llama_cpp(self, texts) -> [str]:
        for i in range(len(texts)):
            text = texts[i]
            output_text = ""
            j=0

            for match in self.pattern.finditer(text):
                before_id = text[:match.start()]
                after_id = text[match.end():]
                generated_text = self.base_model(prompt=before_id, suffix=after_id, stop='.', max_tokens=5)['choices'][0]['text']
                output_text += generated_text if j > 0 else before_id + generated_text
                j+=1

            texts[i] = output_text

        return texts

    def apply_extracted_fills(self, masked_texts, extracted_fills) -> [str]:
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

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

    def extract_fills(self, texts) -> [[str]]:
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def perturb_texts_(self, texts, span_length=5, pct=0.3, ceil_pct=False) -> [str]:
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks_llama_cpp(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
            if attempts > 10:
                break

        return perturbed_texts
    
    def perturb_texts_llama_cpp(self, texts, span_length=5, pct=0.3, ceil_pct=False) -> [str]:
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        perturbed_texts = self.replace_masks_llama_cpp(masked_texts)
        return perturbed_texts

    def perturb_texts(self, texts, chunk_size=20, ceil_pct=False) -> [str]:
        outputs = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations", disable=self.disable_tqdm):
            outputs.extend(self.perturb_texts_llama_cpp(texts[i:i + chunk_size], ceil_pct=ceil_pct))
        return outputs

    def drop_last_word(self, text) -> str:
        return ' '.join(text.split(' ')[:-1])

    def run_DetectGPT_single_text(self, text, n_perturbations=1) -> float:
        torch.manual_seed(42)
        np.random.seed(42)

        original_ll = get_ll(self.base_model, text)
        perturbed_text = self.perturb_texts([text], n_perturbations)
        perturbed_ll = get_lls(self.base_model, perturbed_text, self.disable_tqdm)

        mean_perturbed_ll = np.nanmean(perturbed_ll)
        std_perturbed_ll = np.nanstd(perturbed_ll)

        std_perturbed_ll = std_perturbed_ll if std_perturbed_ll > 0 else 1

        prediction_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll

        return prediction_score

    def run_DetectGPT_feed(self, feed, n_perturbations=5) -> [float]:
        torch.manual_seed(42)
        np.random.seed(42)
        feed = [text for text in feed if len(text.split()) > 49 ]
        if len(feed) < 15:
            return [694201337]
        
        perturbed_texts = self.perturb_texts(feed, n_perturbations)

        original_lls = get_lls(self.base_model, feed, self.disable_tqdm)
        
        perturbed_lls = get_lls(self.base_model, perturbed_texts, self.disable_tqdm)
        mean_perturbed_lls = np.mean([i for i in perturbed_lls if not math.isnan(i)])
        
        std_perturbed_lls = np.std([i for i in perturbed_lls if not math.isnan(i)]) if (len([i for i in perturbed_lls if not (math.isnan(i) or 0)]) > 1) else 1

        prediction_scores = (original_lls - mean_perturbed_lls) / std_perturbed_lls
        print(f"Prediction scores: \n+ {prediction_scores}")
        
        return prediction_scores

    def run_AuthentiGPT_feed(self, feed, n_pertubations=5) -> [float]:
        feed = [text for text in feed if len(text.split()) > 49 ]
        if len(feed) < 15:
            return [694201337]

        perturbed_feeds = self.perturb_texts(feed, n_pertubations)
        
        similiarity_list = []
        for single_feed, perturbed_feed in tqdm(zip(feed, perturbed_feeds),desc='Evaluate Cosine Similarities'):
            try:
                embeddings_original = compute_embeddings_llama(self.base_model, single_feed)
                print(embeddings_original)
                embeddings_perturbed = compute_embeddings_llama(self.base_model, perturbed_feed)
                print(embeddings_perturbed)
                
                embeddings_original_array = np.array(embeddings_original).reshape(1,-1)
                embeddings_perturbed_array = np.array(embeddings_perturbed).reshape(1,-1)
                
                similiarity_list.append(cosine_similarity(embeddings_original_array, embeddings_perturbed_array)[0][0])
            except Exception as e:
                print(f"An error occurred: {e}")
                similiarity_list.append(None)
        
        print(f"Similarities: {similiarity_list}")
        return similiarity_list
        
    def calculate_prediction_scores(self, training_feed, n_perturbations=5) -> [float]:
        start_time = time.time()
        torch.manual_seed(42)
        np.random.seed(42)
        print("Start pertubing texts")
        perturbed_texts = self.perturb_texts(training_feed, n_perturbations)
        print("Start estimating likelihood for original feeds")
        original_lls = get_lls(self.base_model, training_feed, self.disable_tqdm)
        print("Start estimating likelihood for perturbed feeds")
        perturbed_lls = get_lls(self.base_model, perturbed_texts, self.disable_tqdm)

        mean_perturbed_lls = np.mean([i for i in perturbed_lls if not math.isnan(i)])
        print(f"Mean of perturbed lls: {mean_perturbed_lls}")
        std_perturbed_lls = np.std([i for i in perturbed_lls if not math.isnan(i)]) if (len([i for i in perturbed_lls if not (math.isnan(i) or 0)]) > 1) else 1
        print(f"Standard deviation of perturbed lls: {std_perturbed_lls}")

        prediction_scores = (original_lls - mean_perturbed_lls) / std_perturbed_lls
        print(f"Prediction scores: \n+ {prediction_scores}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time/60}")
        return prediction_scores

    def detect_secret(self, newsfeed: list[str]) -> bool:
        # Positive scores = human-written, likely a newspaper article
        # Negative scores = machine-written, likely modified with stego
        
        prediction_scores = self.run_AuthentiGPT_feed(newsfeed)
        if prediction_scores[0] == 694201337:
            decision = True
        else:
            decision = any(score < 0.2 for score in prediction_scores)
        
        return decision