from typing import Optional
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal, getcontext
import sys

import numpy as np

from ..utils import Dec2BinConverter


__all__ = [
    "DynamicArithmeticEncoding",
]


class DynamicArithmeticEncoding:
    """
    ArithmeticEncoding is a class that builds the arithmetic encoding.
    """
    SEQ_FINISH = "SEQUENCE_FINISHED_CHARACTER"

    def __init__(self, frequency_table: dict):
        """
        Needs frequency table with vocabulary
        """
        self.frequency_table = frequency_table
        self.frequency_table[self.SEQ_FINISH] = 1
        
        self.update_prob_table("")

    @staticmethod
    def shannon_entropy(pk, base=2) -> float:
        return -np.sum(pk * np.log(pk)) / np.log(base)

    def update_prob_table(self, char: str) -> None:
        """
        Updates the probability table with the counts of the str
        """
        count = Counter(char)
        for char, freq in count.items():
            self.frequency_table[char] += freq

        self.probability_table = self.get_probability_table(self.frequency_table)

    def get_probability_table(self, frequency_table: Optional[dict]) -> Optional[dict]:
        """
        Normalizes the frequency table
        """
        total = sum(frequency_table.values())
        return {key: value/total for key, value in frequency_table.items()}
    
    def process_stage(self, stage_min: Decimal, stage_max: Decimal) -> dict[str, Decimal]:
        """
        Gets the stage interval and processes the probability chart for the next character.
        """
        stage_span = stage_max - stage_min
        stage_probs = {}

        bottom = stage_min
        for char, prob in self.probability_table.items():
            top = Decimal(prob)*stage_span + bottom
            stage_probs[char] = [bottom, top]
            bottom = top
        
        return stage_probs

    def set_decimal_precision(self, msg: str, bits_per_decimal: float = np.pi) -> None:
        if bits_per_decimal == 0:
            getcontext().prec = 200
            return
        probs_array = np.array(list(self.probability_table.values()))
        entropy = self.shannon_entropy(probs_array)
        min_amount_of_bits = int(entropy * len(msg) + (entropy * len(msg) % 1 > 0))
        needed_digits = int(min_amount_of_bits/bits_per_decimal)
        sys.set_int_max_str_digits(max(needed_digits + 1, sys.get_int_max_str_digits()))
        getcontext().prec = needed_digits

    def encode(self, msg: str, bits_per_decimal: float = np.pi) -> tuple[Decimal, Decimal, Decimal]:
        """
        Iteratively go through the message and find decimal representation of msg.
        If self.dynamic_probs is True, the freq_table gets updated online.
        """

        self.set_decimal_precision(msg, bits_per_decimal)

        bot = Decimal(0.0)
        top = Decimal(1.0)

        for char in msg:
            stage_probs = self.process_stage(bot, top)
            self.update_prob_table(char)
            bot, top = stage_probs[char]

        stage_probs = self.process_stage(bot, top)
        bot, top = stage_probs[self.SEQ_FINISH]

        return (bot+top)/2
    
    def decode(self, encoded_msg: Decimal, vocabulary: Iterable) -> str:

        if isinstance(vocabulary, Iterable):
            vocabulary = {char: 1 for char in vocabulary}
        else:
            raise ValueError("`vocabulary` has to be of type `str` or `dict`")
        self.frequency_table = vocabulary
        self.frequency_table[self.SEQ_FINISH] = 1
        self.update_prob_table("")

        bot = Decimal(0.0)
        top = Decimal(1.0)

        decoded_msg = ""
        while True:
            stage_probs = self.process_stage(bot, top)

            for char, stage_span in stage_probs.items():
                if stage_span[0] <= encoded_msg < stage_span[1]:
                    break
            if char == self.SEQ_FINISH:
                return decoded_msg
            bot, top = stage_probs[char]
            self.update_prob_table(char)
            decoded_msg += char

    def encode_into_binary(self, msg: str, bits_per_decimal: float = np.pi) -> str:
        return Dec2BinConverter.get_bin_from_decimal(self.encode(msg, bits_per_decimal))
    
    def decode_from_binary(self, mantissa: str, exponent: str, prec: str, vocabulary: Iterable) -> str:
        return self.decode(Dec2BinConverter.get_decimal_from_bin(mantissa, exponent, prec), vocabulary)