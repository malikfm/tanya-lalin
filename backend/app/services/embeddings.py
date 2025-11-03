import ollama
from typing import List

from config import settings


class EmbeddingService:
    def __init__(self):
        self.embedding_model = settings.embedding_dim  # Actually using the dimension value as reference
        self.ollama_model = "embeddinggemma"  # Default embedding model
        
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for the given text.
        
        Args:
            text (str): Input text to generate embedding for.
            
        Returns:
            List[float]: Embedding vector representing the semantic meaning of the text.
        """
        response = ollama.embed(model=self.ollama_model, input=text)
        return response["embeddings"][0]
