import math
from tqdm import tqdm
from .seeker import Seeker
from ..utils.llama_utils import get_probabilities
import re
import numpy as np

class dnaGPTseeker(Seeker):
    
    def __init__(self, disable_tqdm) -> None:
        super().__init__(disable_tqdm)

    def split_input_and_contine(self, text, K=5):
        words = text.split()
        cut_off = len(words) // 2

        x = ' '.join(words[:cut_off])
        y0 = ' '.join(words[cut_off:])
        y0_sequences = []
        for i in range(K):
            generated = re.sub(r'[.,;!]','', self.base_model.create_completion(prompt=x, max_tokens=cut_off, temperature=0.7)['choices'][0]['text']).lower()
            y0_sequences.append(generated)

        #For Debugging purposes:
        #print(f"original text: {text} \ny0: {y0} \ny0_sequences: ")
        #for elem in y0_sequences:
        #    print(elem + '\n')
    
        return y0.lower(), y0_sequences

    def ngrams(self,sequence, n) -> [str]:
        """Generate n-grams from a sequence."""
        if not isinstance(sequence, (list, str)) or not isinstance(n, int) or n <= 0:
            #print(f"n: {n} \n sequence: {sequence}")
            raise ValueError("Invalid input for n-grams generation.")

        words = sequence.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def n_gram_distance(self, y0, y0_sequences, K=5, n0=1, N=9) -> float:
        'N-Gram Distance Calculation (B-Score)'
        score_sum = 0.0
        for k in range(K):
            human_text = y0
            model_output = y0_sequences[k-1]
            for n in range(n0, N + 1):
                if len(model_output) != 0:
                    y_0_grams = set(self.ngrams(human_text, n))
                    if len(y_0_grams) != 0:
                        y_k_grams = set(self.ngrams(model_output, n))
                        intersection = set.intersection(y_0_grams, y_k_grams)
                        f_n = n * math.log(n)
                        term = len(intersection) / (len(model_output) * len(y_0_grams))
                        score_sum += term * f_n
        return score_sum / K

    def calculate_evidence(self, y0, y0_sequences, n=1) -> int:
        """
        Calculate the evidence En as the overlap of n-grams between model output and human text.
        """
        En = set()
        ngrams_y0 = set(self.ngrams(y0, n))

        for Yk in y0_sequences:
            ngrams_Yk = set(self.ngrams(Yk, n))
            En |= ngrams_Yk & ngrams_y0

        return len(En)

    def relative_entropy_distance(self, y0, y0_sequences, K=5) -> float:
        '''
        Relative Entropy Distance Calculation (W-Score)
        ATTENTION: Is not tested now
        '''
        x_probs = get_probabilities(self.base_model, y0)
        y0_sequences_probs = [get_probabilities(self.base_model, elem) for elem in y0_sequences]
        score_sum = 0

        for k in range(K):
            min_length = min(len(x_probs), len(y0_sequences_probs[k]))
            for i in range(min_length):
                p_x = x_probs[i]
                p_y = y0_sequences_probs[k][i]

                # Avoid log(0) by adding a small constant if necessary
                p_x = np.maximum(p_x, 1e-10)
                p_y = np.maximum(p_y, 1e-10)

        return score_sum / (K * min_length)

    def calculate_scoring_for_newsfeed(self, texts, k=5, distance_method='n-gram') -> [bool]:
        'Do scoring for all texts and return bool weather the text has stego or not'
        prediction_results = []
        for text in tqdm(texts, desc=f"Evaluate {distance_method} score for texts", disable=self.disable_tqdm):
            y0, y0_sequences = self.split_input_and_contine(text,k)
            if distance_method == 'n-gram':
                b_score = self.n_gram_distance(y0, y0_sequences, k)
                evidence = self.calculate_evidence(y0, y0_sequences)
                n_gram_evaluation = b_score > evidence
                print(f"{b_score} > {evidence} = {n_gram_evaluation}")
                prediction_results.append(n_gram_evaluation)
            elif distance_method == 'entropy':
                entropy_distance_evaluation = self.relative_entropy_distance(y0, y0_sequences) > self.calculate_evidence(y0, y0_sequences)
                prediction_results.append(entropy_distance_evaluation)

        print(f"Results: \n {prediction_results}")
        return prediction_results
    
    def detect_secret(self,newsfeed) -> bool:
        return True if any(True for prediction in self.calculate_scoring_for_newsfeed(newsfeed, distance_method='entropy')) else False