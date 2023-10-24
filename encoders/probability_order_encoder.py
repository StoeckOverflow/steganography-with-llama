import tqdm

from llama_cpp import Llama

from utils import split_in_separators, initialize_token_getter


class ProbabilityOrderEncoder:

    def __init__(self, llm: Llama):
        self.get_valid_token = initialize_token_getter(llm)
        self.llm = llm

    def first_draft_encode(self, prompt: str, binary_secret: str, bits_per_token: int) -> str:
        # Create stego feed from real feed for prompt
        new_feed = " ".join(split_in_separators(prompt))

        encode_token_sequence = []
        output_file = open(repr(self.__class__) + "_encoding_log.txt", "w")
        # Iterate until message is contained in new_feed
        for next_bits in tqdm.tqdm([binary_secret[i:i+bits_per_token] for i in range(0, len(binary_secret), bits_per_token)], "Message hidden: "):
            if len(next_bits) < 2:
                break
            while True:
                next_token_probs = self.get_valid_token(new_feed, bits_per_token= bits_per_token)
                if len(next_token_probs) > 1:
                    chosen_ind = int(next_bits, 2)
                    next_token = next_token_probs[chosen_ind]
                    new_feed = new_feed + next_token
                    break
                new_feed = new_feed + next_token_probs[0]
        
            encode_token_sequence.append(self.llm.tokenize(next_token.encode()))
            output_file.write(next_bits + "\n")
            output_file.write(repr(next_token_probs) + "\t" + repr(chosen_ind) + "\n")
            output_file.write(next_token + "\n" + new_feed + "\n")
            output_file.flush()
        output_file.write(repr(encode_token_sequence))
        output_file.close()
        return new_feed

    def first_draft_decode(self, news_feed: str, nr_prompt_words: int, bits_per_token: int = 3) -> str:
        """
        Takes a news_feed as input and returns the extracted binary message
        """
        # Decode
        tmp_feed = " ".join(split_in_separators(news_feed)[:nr_prompt_words])

        decoded_message = ""
        output_file = open(repr(self.__class__) + "_decoding_log.txt", "w")
        i = len(tmp_feed)

        decode_token_sequence = []
        # Iterate until news feed is exhausted
        pbar = tqdm.tqdm(total=len(news_feed))
        while i < len(news_feed):
            while True:
                next_token_probs = self.get_valid_token(tmp_feed, bits_per_token= bits_per_token)
                if len(next_token_probs) > 1:
                    max_len_next_token = max([len(token) for token in next_token_probs]) # Check the max length to check in message where the next token has to be found
                    next_possible_tokens = [token for token in next_token_probs if token in news_feed[i:i+max_len_next_token]] # If there's still more than one that would fit
                    next_possible_tokens = [sorted(next_possible_tokens, key= lambda x: news_feed[i:i+max_len_next_token].find(x))[0]] # Keep the one that occurrs first -> next token
                    assert len(next_possible_tokens) == 1, f"{next_possible_tokens=} - {next_token_probs=} - {news_feed[i:i+max_len_next_token]}" # Assert that we only have one candidate left
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
            output_file.write(next_bits + "\n")
            output_file.write(repr(next_token_probs) + "\t" + repr(chosen_ind) + "\n")
            output_file.write(next_token + "\n" + tmp_feed + "\n")
            output_file.flush()
            pbar.n = i
            pbar.refresh()

        return decoded_message


    def encode_single_string(self, prompt: str, binary_secret: str, bits_per_token: int, soft_max_chars_limit: int = 450) -> tuple[str, str]:
        """
        Encode as many bits of the binary secret into news_string, return news_string containing
        secret and all bits that did not fit.
        """

        # Create stego feed from real feed for prompt
        doctored_news_feed = " ".join(split_in_separators(prompt))

        # Iterate until message is contained in new_feed
        for i, next_bits in tqdm.tqdm([(_i, binary_secret[_i:_i+bits_per_token]) for _i in range(0, len(binary_secret), bits_per_token)], "Message hidden: "):
            if len(next_bits) < bits_per_token:
                return doctored_news_feed, next_bits
            while True:
                next_token_probs = self.get_valid_token(doctored_news_feed, bits_per_token=bits_per_token)
                if len(next_token_probs) > 1:
                    chosen_ind = int(next_bits, 2)
                    next_token = next_token_probs[chosen_ind]
                    doctored_news_feed = doctored_news_feed + next_token
                    if len(doctored_news_feed) > soft_max_chars_limit and doctored_news_feed.endswith("."):
                        return doctored_news_feed, binary_secret[i+bits_per_token:]
                    break
                doctored_news_feed = doctored_news_feed + next_token_probs[0]

        # Add a token that's outside of the scope of the coding to signal that the whole message has been decoded.
        doctored_news_feed += self.get_valid_token(doctored_news_feed, bits_per_token=bits_per_token+1)[(2**bits_per_token)+1] 
        return doctored_news_feed, ""
    
    def encode_news_feed(self, news_feed: list[str], binary_secret: str, bits_per_token: int, soft_max_chars_lim: int = 450, nr_prompt_words: int = 5) -> str:
        remaining_secret = binary_secret
        doctored_news_feed = []
        for news_string in news_feed:
            prompt = " ".join(split_in_separators(news_string)[:nr_prompt_words])
            modified_news_string, remaining_secret = self.encode_single_string(prompt, remaining_secret, bits_per_token, soft_max_chars_lim)
            doctored_news_feed.append(modified_news_string)
            if len(remaining_secret) < bits_per_token:
                break

        return doctored_news_feed + news_feed[len(doctored_news_feed):]



    def decode_single_string(self, news_string: str, nr_prompt_words: int, bits_per_token: int = 3) -> str:
        """
        Decodes as many bits as possible until news_string is exhausted or exit condition
        is met.
        """
        output_file = open(repr(self.__class__) + "_decoding_news_string_log.txt", "w")

        # Decode
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
                    next_possible_tokens = [sorted(next_possible_tokens, key= lambda x: news_string[i:i+max_len_next_token].find(x))[0]] # Keep the one that occurrs first -> next token
                    if len(next_possible_tokens) == 0:
                        """
                        If the next token does not belong to the top 2**bits_per_token
                        then we assume the whole message has been decoded.
                        """
                        return decoded_message
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
            output_file.write(next_bits + "\n")
            output_file.write(repr(next_token_probs) + "\t" + repr(chosen_ind) + "\n")
            output_file.write(next_token + "\n" + tmp_feed + "\n")
            output_file.flush()
            pbar.n = i
            pbar.refresh()

        return decoded_message
    
    def decode_news_feed(self):
        raise NotImplementedError