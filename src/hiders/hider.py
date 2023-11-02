from abc import ABC, abstractmethod
import json
import sys
from ..utils import encode_secret
from llama_cpp import Llama
from ..codecs.codec import Codec

class Hider(ABC):
    
    def __init__(self, codec: Codec, llm:Llama):
        self.codec = codec(llm)
    
    def hide_secret(self, newsfeed: list[str], secret:str, output:str = "doctored_feed"):
        """
        Hide a secret within a newsfeed using the encoder of codec.

        Args:
            newsfeed (list[str]): The newsfeed text to hide the secret in.
            secret (str): The secret to hide as string.
            output (str, optional): Defines the output format. If string or json is given, it will be saved in thre respective file format.
            otherwise it will be saved as json with de given filename or with the name "doctored_feed" as default file name.

        Returns:
            encoded_newsfeed: The newsfeed text with the secret hidden.
        """
        binary_secret = encode_secret(secret)[2:]
        doctored_newsfeed = self.codec.encode_newsfeed(newsfeed, binary_secret)
        doctored_feed = {"feed": doctored_newsfeed}
        if(output == "string"):
            return doctored_newsfeed
        elif(output == "json"):
            return doctored_feed
        elif(output == "stdout"):
            json.dump(doctored_feed, sys.stdout)
            sys.stdout.flush()
        else:
            with open(output +".json", "w") as f:
                json.dump(doctored_feed, f)
                
    def reveal_secret(self, newsfeed: list[str], output:str = "decoded_feed"):
        """Decodes the secret out of a given newsfeed

        Args:
            newsfeed (list[str]): The newsfeed to decode.
            output (str, optional): Defines the output format. If string or json is given, it will be saved in thre respective file format.
            otherwise it will be saved as json with de given filename or with the name "doctored_feed" as default file name.

        Returns:
            decoded_secret: decoded secret out of given newsfeed.
        """
        doctored_secret = self.codec.decode_newsfeed(newsfeed)
        if(output == "string"):
            return doctored_secret
        elif(output == "json"):
            decoded_feed_json = {"secret": doctored_secret}
            return decoded_feed_json
        elif(output == "stdout"):
            decoded_feed_json = {"secret": doctored_secret}
            json.dump(decoded_feed_json, sys.stdout)
            sys.stdout.flush()
        else:
            decoded_feed_json = {"secret": doctored_secret}
            with open(output +".json", "w") as f:
                json.dump(decoded_feed_json, f)