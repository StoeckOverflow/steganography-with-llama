import json
import tqdm
from typing import Iterable, List, Tuple
from ..utils import split_in_separators, initialize_token_getter, decode_secret
from .codec import Codec
from llama_cpp import Llama

class ProbabilityOrderCodec(Codec):

    def __init__(self, llm: Llama):
        super().__init__(llm)
        self.get_valid_token = initialize_token_getter(llm)

    def encode_single_string(self, binary_secret: str, prompt: str, bits_per_token: int = 3, soft_max_chars_limit: int = 450) -> Tuple[str, str]:
        """
        Encode as many bits of the binary secret into news_string.

        Args:
            prompt (str): prompt words to use for the next encoding
            binary_secret (str): string secret in binary representation
            bits_per_token (int): how many bits of the binary secret are encoded per prompt
            soft_max_chars_limit (int, optional): _description_. Defaults to 450.

        Returns:
            tuple[str, str]: [news_string containing, secret and all bits that did not fit]
        """
        # Create stego feed from real feed for prompt
        doctored_newsfeed = " ".join(split_in_separators(prompt))
        return_bits = ""

        log = {
            "prompts": [],
            "predicted_tokens": [],
        }
        # Iterate until message is contained in newsfeed
        for i, next_bits in tqdm.tqdm([(_i, binary_secret[_i:_i+bits_per_token]) for _i in range(0, len(binary_secret), bits_per_token)], "Message hidden: "):
            if len(next_bits) < bits_per_token:
                return_bits = next_bits
                break
            while True:
                next_token_probs = self.get_valid_token(doctored_newsfeed, bits_per_token=bits_per_token)
                log["prompts"].append(doctored_newsfeed)
                log["predicted_tokens"].append(next_token_probs)
                if len(next_token_probs) > 1:
                    chosen_ind = int(next_bits, 2)
                    next_token = next_token_probs[chosen_ind]
                    doctored_newsfeed = doctored_newsfeed + next_token
                    if len(doctored_newsfeed) > soft_max_chars_limit and doctored_newsfeed.endswith("."):
                        return doctored_newsfeed, binary_secret[i+bits_per_token:]
                    break
                doctored_newsfeed = doctored_newsfeed + next_token_probs[0]

        if len(return_bits) == 0:
            # Add a token that's outside of the scope of the coding to signal that the whole message has been encoded
            doctored_newsfeed += self.get_valid_token(doctored_newsfeed, bits_per_token=bits_per_token+1)[(2**bits_per_token)+1] 

        # with open(f"{uniquify('resources/backlog/encoding_history')}.json", "w") as f:
        #     f.write(json.dumps(log, indent=2))

        return doctored_newsfeed, return_bits
    
    def encode_newsfeed(self, newsfeed: List[str], binary_secret: str, bits_per_token: int = 3, soft_max_chars_lim: int = 450, nr_prompt_words: int = 5) -> str:
        """
        Encodes the binary_secret in the newsfeed.

        Args:
            newsfeed (list[str]): A list of different newsfeeds as strings
            binary_secret (str): string secret in binary representation
            bits_per_token (int): how many bits of the binary secret are encoded per prompt
            soft_max_chars_lim (int, optional): _description_. Defaults to 450.
            nr_prompt_words (int, optional): how many words of the next prompt are used to encode. Defaults to 5.

        Returns:
            str: newsfeed with encoded secret concatenated with the rest of newsfeed, which was not used for encoding.
        """
        
        remaining_secret = binary_secret
        doctored_newsfeed = []
        for news_string in newsfeed:
            prompt = " ".join(split_in_separators(news_string)[:nr_prompt_words])
            modified_news_string, remaining_secret = self.encode_single_string(remaining_secret, prompt, bits_per_token, soft_max_chars_lim)
            doctored_newsfeed.append(modified_news_string)
            if len(remaining_secret) < bits_per_token:
                break
                
        return doctored_newsfeed + newsfeed[len(doctored_newsfeed):]

    def decode_single_string(self, news_string: str, nr_prompt_words: int = 5, bits_per_token: int = 3) -> Tuple[str, bool]:
        """
        Decodes the binary_secret out of a single newsfeed string.
        
        Args:
            newsfeed (str): single newsfeed as string
            nr_prompt_words (int, optional): how many words of the next prompt are used to encode. Defaults to 5.
            bits_per_token (int): how many bits of the binary secret are encoded per prompt

        Returns:
            str: decoded secret.
        """
        
        tmp_feed = " ".join(split_in_separators(news_string)[:nr_prompt_words])

        decoded_message = ""
        i = len(tmp_feed)
        
        decode_token_sequence = []
        # Iterate until news feed is exhausted
        pbar = tqdm.tqdm(total=len(news_string))
        while i < len(news_string):
            while True:
                next_token_probs = self.get_valid_token(tmp_feed, bits_per_token=bits_per_token)
                if len(next_token_probs) > 1:
                    max_len_next_token = max([len(token) for token in next_token_probs]) # Check the max length to check in message where the next token has to be found
                    next_possible_tokens = [token for token in next_token_probs if token in news_string[i:i+max_len_next_token]] # If there's still more than one that would fit
                    next_possible_tokens = sorted(next_possible_tokens, key= lambda x: news_string[i:i+max_len_next_token].find(x)) # Keep the one that occurs first -> next token
                    if not next_possible_tokens:
                        """
                        If the next token does not belong to the top 2**bits_per_token
                        then we assume the whole message has been decoded.
                        """
                        return decoded_message, True
                    # assert len(next_possible_tokens) == 1, f"{next_possible_tokens=} - {next_token_probs=} - {news_string[i:i+max_len_next_token]}" # Assert that we only have one candidate left
                    next_token = next_possible_tokens[0]
                    chosen_ind = next_token_probs.index(next_token)
                    tmp_feed += next_token
                    i = len(tmp_feed)
                    break
                tmp_feed += next_token_probs[0]
                i = len(tmp_feed)
            decode_token_sequence.append(self.llm.tokenize(next_token.encode()))
            next_bits = bin(chosen_ind)[2:]
            if len(next_bits) < 3:
                next_bits = (3-len(next_bits))*"0" + next_bits
            decoded_message += next_bits
            pbar.n = i
            pbar.refresh()

        return decoded_message, False
    
    def decode_newsfeed(self, newsfeed: List[str], nr_prompt_words: int = 5, bits_per_token: int = 3) -> str:
        """
        Decodes the binary secret from a list of newsfeed strings.
        ToDo: Decoding into Text Message from bitstream
        Args:
            newsfeed (list[str]): A list of newsfeed strings.
            nr_prompt_words (int): The number of words used as the prompt for encoding.
            bits_per_token (int): The number of bits encoded per token.

        Returns:
            str: The full encoded secret.
        """
        remaining_decoded_secret = []
        for news_string in newsfeed:
            binary_secret, decoded = self.decode_single_string(news_string, nr_prompt_words, bits_per_token)
            
            if decoded:
                remaining_decoded_secret.append(binary_secret)
                concatenated_binary_secret = ''.join(remaining_decoded_secret)
                return decode_secret(concatenated_binary_secret)
            
            remaining_decoded_secret.append(binary_secret)
            
        concatenated_binary_secret = ''.join(remaining_decoded_secret)
        return decode_secret(concatenated_binary_secret)