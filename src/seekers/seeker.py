from abc import ABC, abstractmethod
from llama_cpp import Llama

class Seeker(ABC):
    
    def __init__(self) -> None:
        self.model_path = 'resources/llama-2-7b.Q5_K_M.gguf'
        self.base_model = Llama(
            model_path=self.model_path,
            verbose=False,        
            logits_all=True,      
            n_ctx=512,            # Maximum context size (number of tokens) the model can handle
            n_batch=512,          # Number of tokens to process in one batch
            use_mlock=True        # Use mlock to prevent paging the model to disk (depends on your system's memory)
        )
    
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