from abc import ABC, abstractmethod
from llama_cpp import Llama
import sys
import json

class Seeker(ABC):
    
    def __init__(self, disable_tqdm) -> None:
        self.model_path = 'resources/llama-2-7b.Q5_K_M.gguf'
        self.base_model = Llama(
            model_path=self.model_path,
            verbose=False,        
            logits_all=True,      
            n_ctx=512,            # Maximum context size (number of tokens) the model can handle
            n_batch=512,          # Number of tokens to process in one batch
            n_threads=3,          # Number of threads llama operations can be processed
            n_threads_batch=3,    # similar to n_threads, but for batch processing (parallel execution of different llama operations)
            use_mlock=True,        # Use mlock to prevent paging the model to disk (depends on your system's memory)
            embedding=True
        )
        self.disable_tqdm = disable_tqdm
    
    @abstractmethod
    def detect_secret(self, newsfeed: list[str]) -> bool:
        """
        Detect and extract a secret from a given newsfeed.

        Args:
            newsfeed list[str]: The newsfeed text to search for a secret.

        Returns:
            boolean: True or False depending of containing hidden message or not.
        """
        pass
    
    def detection_interface(self) -> None:
        json_file: [str] = json.load(sys.stdin)
        newsfeed = json_file['feed']
        has_secret = {"result": self.detect_secret(newsfeed)}
        json.dump(has_secret, sys.stdout)   