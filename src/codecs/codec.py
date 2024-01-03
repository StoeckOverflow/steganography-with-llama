from abc import ABC, abstractmethod
from llama_cpp import Llama

class Codec(ABC):
    
    def __init__(self, llm: Llama):
        self.llm = llm
    
    @abstractmethod
    def encode_single_string(self, binary_secret: str, **kwargs):
        """
        Encode a binary_secret string and return the encoded version.

        Args:
            binary_secret (str): The binary_secret string to encode.

        Returns:
            str: The encoded binary_secret.
        """
        pass
    
    @abstractmethod
    def encode_newsfeed(self, news_feed: list[str], binary_secret: str, **kwargs) -> str:
        """
        Encode a binary_secret string and return the encoded version and all bits that did not fit.

        Args:
            binary_secret (str): The binary_secret string to encode.

        Returns:
            str: The encoded binary_secret.
        """
        pass

    @abstractmethod
    def decode_single_string(self, newsfeed:str, **kwargs) -> str:
        """
        Decode an encoded binary_secret and return the original binary_secret string.

        Args:
            encoded_binary_secret (str): The encoded binary_secret to decode.

        Returns:
            str: The decoded original binary_secret.
        """
        pass
    
    @abstractmethod
    def decode_newsfeed(self, newsfeed: list[str], **kwargs) -> str:
        """
        Decode an encoded binary_secret and return the original binary_secret string.

        Args:
            encoded_binary_secret (str): The encoded binary_secret to decode.

        Returns:
            str: The decoded original binary_secret.
        """
        pass