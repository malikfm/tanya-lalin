import ollama
from typing import List

from config import settings


class LLMService:
    def __init__(self, ollama_client: ollama.AsyncClient):
        self.ollama_client = ollama_client

    async def generate_response(self, query: str, relevant_chunks: List[dict]) -> str:
        """Generate a response based on the user query and relevant document chunks.
        
        Args:
            query (str): Original user query.
            relevant_chunks (List[dict]): List of relevant document chunks.
            
        Returns:
            str: Generated response based on the query and context.
        """
        # Format the context from the retrieved chunks
        context_parts = []
        for chunk in relevant_chunks:
            chunk_info = f"Pasal {chunk['article_number']}"
            if chunk["paragraph_number"] is not None:
                chunk_info += f", Ayat {chunk['paragraph_number']}"
            chunk_info += f" ({chunk['chunk_type']}): {chunk['text']}"
            context_parts.append(chunk_info)
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt with context
        prompt = (
            "Anda adalah seorang ahli dalam hukum lalu lintas Indonesia. Silakan jawab pertanyaan pengguna berdasarkan kutipan dokumen hukum berikut:"
            f"\n{context}"
            "\n\n---\n\n"
            "Pertanyaan:"
            f"\n{query}"
            "\n\n---\n\n"
            "Harap berikan jawaban yang jelas dan akurat berdasarkan kutipan hukum yang disediakan."
            " Jika jawabannya tidak terdapat dalam kutipan yang diberikan, mohon sampaikan hal tersebut dengan jelas."
            " Jawaban harus singkat dan, bila memungkinkan, merujuk pada pasal yang relevan."
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        # Generate response using Ollama
        response = await self.ollama_client.chat(
            model=settings.llm_model,
            messages=messages
        )
        
        return response["message"]["content"].strip()
