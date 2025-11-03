import ollama
from typing import List

from config import settings


class EmbeddingService:
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for the given text.
        
        Args:
            text (str): Input text to generate embedding for.
            
        Returns:
            List[float]: Embedding vector representing the semantic meaning of the text.
        """
        response = ollama.embed(model=settings.embedding_model, input=text)
        return response["embeddings"][0]
