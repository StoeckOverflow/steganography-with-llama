import math
from collections import Counter
import json
from tqdm import tqdm
from .seeker import Seeker
from ..utils.llama_utils import get_probabilities
#from functools import lru_cache
# TODO: RunTime Optimization and Decision Function with thresholding, Explanatation Markdown, Integration in seeker class
class dnaGPTseeker(Seeker):
    
    def __init__(self, disable_tqdm) -> None:
        super().__init__(disable_tqdm)
    
    #@lru_cache(maxsize=512)
    def split_input_and_contine(self, text, K=5):
        words = text.split()
        cut_off = len(words) // 2

        x = ' '.join(words[:cut_off])
        y0 = ' '.join(words[cut_off:])
        y0_sequences = []
        for i in range(K):
            generated = self.base_model.create_completion(prompt=x, max_tokens=cut_off)['choices'][0]['text']
            y0_sequences.append(generated)

        #For Debugging purposes:
        #print(f"original text: {text} \ny0: {y0} \ny0_sequences: ")
        #for elem in y0_sequences:
        #    print(elem + '\n')
    
        return y0, y0_sequences

    def ngrams(self,sequence, n):
        """Generate n-grams from a sequence."""
        if not isinstance(sequence, (list, str)) or not isinstance(n, int) or n <= 0:
            #print(f"n: {n} \n sequence: {sequence}")
            raise ValueError("Invalid input for n-grams generation.")

        words = sequence.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def n_gram_distance(self, y0, y0_sequences, K=5, n0=1, N=9):
        'N-Gram Distance Calculation (B-Score)'
        score_sum = 0.0
        for k in range(1,K):
            human_text = y0
            model_output = y0_sequences[k-1]
            for n in range(n0, N + 1):
                human_ngrams = Counter(self.ngrams(human_text, n))
                model_ngrams = Counter(self.ngrams(model_output, n))
                intersection = human_ngrams & model_ngrams
                f_n = n * math.log(n)
                if len(human_ngrams) * len(model_output) != 0:
                    term = sum(intersection.values()) / (len(human_ngrams) * len(model_output))
                    score_sum += term * f_n

        return score_sum / K

    def calculate_evidence(self, y0, y0_sequences, n=3):
        """
        Calculate the evidence En as the overlap of n-grams between model output and human text.
        """
        En = set()
        ngrams_y0 = set(self.ngrams(y0, n))

        for Yk in y0_sequences:
            ngrams_Yk = set(self.ngrams(Yk, n))
            En |= ngrams_Yk & ngrams_y0

        return len(En)

    def relative_entropy_distance(self, x, y0_sequences, K):
        '''
        Relative Entropy Distance Calculation (W-Score)
        ATTENTION: Is not tested now
        '''

        x_probs = get_probabilities(x)
        y0_sequences_probs = [get_probabilities(elem) for elem in y0_sequences]
        score_sum = 0

        for k in range(K):
            min_length = min(len(x_probs), len(y0_sequences_probs[k]))
            for i in range(min_length):
                p_x = x_probs[i]
                p_y = y0_sequences_probs[k][i]

                # Avoid log(0) by adding a small constant if necessary
                p_x = max(p_x, 1e-10)
                p_y = max(p_y, 1e-10)

                score_sum += p_x * math.log(p_x / p_y)

        return score_sum / (K * min_length)


    def calculate_blackbox_scoring_for_newsfeed(self, texts, k=5, distance_method='n-gram'):
        'Do black box scoring for all texts and return bool weather the text has stego or not'
        prediction_results = []
        for text in tqdm(texts, desc=f"Evaluate {distance_method} score for texts", disable=self.disable_tqdm):
            y0, y0_sequences = self.split_input_and_contine(text,k)
            if distance_method == 'n-gram':
                b_score, evidence = self.n_gram_distance(y0, y0_sequences, k), self.calculate_evidence(y0, y0_sequences)
                prediction_results.append((b_score, evidence))
            elif distance_method == 'entropy':
                entropy_distance = self.relative_entropy_distance(y0, y0_sequences)
                prediction_results.append(entropy_distance)
            

        if distance_method == 'n-gram':
            print(f"Blackbox results in the format (b_score, evidence): : \n {prediction_results}")
        elif distance_method == 'entropy':
            print(f"Entropy results: \n {prediction_results}")
        return prediction_results
    
    def detect_secret(self,newsfeed):
        prediction_scores = self.calculate_blackbox_scoring_for_newsfeed(newsfeed, distance_method='n-gram')
        '''
        For Black Box Method with ngram distance you can also measure the evidence, which serves as an additional score
        If the evidence is very high, it is very likely the text is machine-written (contains stego)
        '''
        decision = decision = any((t[0] > 2 or t[1] > 2) for t in prediction_scores)
        return decision
    
if __name__ == "__main__":
    #input_data = json.load(sys.stdin)
    #feed = input_data["feed"]
    with open('resources/feeds/example_feed.1') as file:
        input_data_1 = json.load(file)
    benign_feed = input_data_1['feed']
    with open('resources/feeds/example_feed.-1') as file:
        input_data_2 = json.load(file)
    stego_feed = input_data_2['feed']
    print('Calculate BlackBox Scoring for benign feed')
    result = dnaGPTseeker.calculate_blackbox_scoring_for_newsfeed(benign_feed)
    print('Calculate Blackbox Scoring for Stego feed')
    result = dnaGPTseeker.alculate_blackbox_scoring_for_newsfeed(stego_feed)
    #json.dump(result, sys.stdout)