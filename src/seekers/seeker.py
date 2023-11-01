from abc import ABC, abstractmethod

class Seeker(ABC):
    
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