from typing import Literal, Union, List, Dict
from collections.abc import Iterable
import sys
import numpy as np
import json
from llama_cpp import Llama
from ..codecs import DynamicArithmeticEncoding
from ..utils import Dec2BinConverter
from ..hiders import ArithmeticProbOrdHider


class DynamicPOE:
    """
    Combines a Dynamic Arithmetic Encoding Codec with it's hider.
    """
    def __init__(self, bits_per_token: int = 3, skip_tokens: int = 0, skip_feeds: int = 0, vocabulary: Iterable = None, path_to_llm: str = "resources/llama-2-7b.Q5_K_M.gguf", disable_tqdm: bool = True, n_gpu_layers: int = 0):
        if vocabulary is None:
            vocabulary = self.get_default_vocabulary()
            self.vocabulary = vocabulary
        llm = Llama(model_path=path_to_llm, n_ctx=512, seed=1337, verbose=False, logits_all=True, n_threads=None, use_mlock=False, n_gpu_layers=n_gpu_layers)
        self.codec = DynamicArithmeticEncoding(frequency_table={char: 1 for char in vocabulary})
        self.hider = ArithmeticProbOrdHider(llm, bits_per_token=bits_per_token, skip_tokens=skip_tokens, skip_feeds=skip_feeds, disable_tqdm=disable_tqdm)
    
    @staticmethod
    def get_default_vocabulary() -> Iterable:
        return [chr(i) for i in range(32, 127)]

    def get_highest_compression(self, message: str, starting_x: float = np.pi, i_step: float = 0.01):
        x = starting_x
        i = 0
        while True:
            enc = DynamicArithmeticEncoding(frequency_table={char: 1 for char in self.vocabulary})
            encoded_msg = enc.encode(message, bits_per_decimal= x+i)
            encoded_binary_messages = Dec2BinConverter.get_bin_from_decimal(encoded_msg)
            decoded_encoded_msg = Dec2BinConverter.get_decimal_from_bin(*encoded_binary_messages)
            decoded_msg = enc.decode(decoded_encoded_msg, self.vocabulary)
            if message == decoded_msg:
                i += i_step
            else:
                return x+i-i_step

    def hide(
            self,
            message: str,
            news_feed: List[str],
            try_extra_compression: bool = True,
            nr_prompt_words: int = 5,
            labeled_for_training_flag=False,
            soft_max_chars_lim: int = 450,
            chars_or_words_lim: Literal["Chars", "Words"] = "Chars",
            ) -> Dict[str, List[str]]:
        avg_char_len_per_article = sum([len(article) for article in news_feed])/len(news_feed)
        if avg_char_len_per_article > 700:
            soft_max_chars_lim = 480
            chars_or_words_lim = "Words"
        else:
            chars_or_words_lim = "Chars"
        bits_per_decimal = self.get_highest_compression(message) if try_extra_compression else np.pi
        encoded_msg = self.codec.encode(message, bits_per_decimal)
        encoded_binary_messages = Dec2BinConverter.get_bin_from_decimal(encoded_msg, bits_per_token=self.hider.bits_per_token)
        if labeled_for_training_flag:
            doctored_news_feed, length_rest = self.hider.hide_in_whole_newsfeed(
                news_feed,
                encoded_binary_messages,
                nr_prompt_words=nr_prompt_words,
                labeled_for_training_flag=labeled_for_training_flag,
                soft_max_chars_lim=soft_max_chars_lim,
                chars_or_words_lim=chars_or_words_lim,
                )
            return doctored_news_feed, length_rest
        else:
            doctored_news_feed = self.hider.hide_in_whole_newsfeed(
                news_feed,
                encoded_binary_messages,
                nr_prompt_words=nr_prompt_words,
                labeled_for_training_flag=False,
                soft_max_chars_lim=soft_max_chars_lim,
                chars_or_words_lim=chars_or_words_lim,
                )
            return doctored_news_feed
    
    def recover(self, doctored_news_feed: List[str], vocabulary: Iterable = None, nr_prompt_words: int = 5) -> Dict[str, str]:
        if vocabulary is None:
            vocabulary = self.get_default_vocabulary()
        decoded_binary_messages = self.hider.retrieve_multiple_secrets_from_news_feed(doctored_news_feed, nr_prompt_words=nr_prompt_words)
        decoded_encoded_msg = Dec2BinConverter.get_decimal_from_bin(*decoded_binary_messages)
        decoded_msg = self.codec.decode(decoded_encoded_msg, vocabulary)
        return {"secret": decoded_msg}
    
    def hide_interface(self) -> None:
        json_file: Dict[str, List[str]] = json.load(sys.stdin)
        message, news_feed = json_file["secret"], json_file["feed"]
        doctored_news_feed = self.hide(message, news_feed)
        json.dump(doctored_news_feed, sys.stdout)

    def recover_interface(self) -> None:
        json_file: Dict[str, List[str]] = json.load(sys.stdin)
        doctored_news_feed = json_file["feed"]
        recovered_message = self.recover(doctored_news_feed)
        json.dump(recovered_message, sys.stdout)