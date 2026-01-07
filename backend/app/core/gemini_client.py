"""Google Gemini client for LLM and embedding operations."""
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from config import settings


class GeminiClient:
    """Client for Google Gemini API operations."""
    
    def __init__(self):
        """Initialize Gemini client with API key."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.llm_model = settings.llm_model
        self.embedding_model = settings.embedding_model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_text(
        self, 
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.7
    ) -> str:
        """Generate text using Gemini LLM.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0.0 - 1.0)
            
        Returns:
            Generated text response
        """
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            
            return response.embeddings[0].values
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=texts
            )
            
            return [emb.values for emb in response.embeddings]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


# Singleton instance
_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
