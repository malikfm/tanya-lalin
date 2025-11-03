import ollama
from typing import List

from app.db.models import LegalTextChunk


class LLMService:
    def __init__(self):
        self.llm_model = "llama3"  # Default model, can be configured
        
    def generate_response(self, query: str, relevant_chunks: List[LegalTextChunk]) -> str:
        """Generate a response based on the user query and relevant document chunks.
        
        Args:
            query (str): Original user query.
            relevant_chunks (List[LegalTextChunk]): List of relevant document chunks.
            
        Returns:
            str: Generated response based on the query and context.
        """
        # Format the context from the retrieved chunks
        context_parts = []
        for chunk in relevant_chunks:
            chunk_info = f"Pasal {chunk.article_number}"
            if chunk.paragraph_number is not None:
                chunk_info += f", Ayat {chunk.paragraph_number}"
            chunk_info += f" ({chunk.chunk_type}): {chunk.text}"
            context_parts.append(chunk_info)
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt with context
        prompt = f"""
        Anda adalah seorang ahli dalam hukum lalu lintas Indonesia. Silakan jawab pertanyaan pengguna berdasarkan kutipan dokumen hukum berikut:

        CONTEXT:
        {context}

        QUESTION:
        {query}

        Harap berikan jawaban yang jelas dan akurat berdasarkan teks hukum yang disediakan. Jika jawabannya tidak terdapat dalam konteks yang diberikan, mohon sampaikan hal tersebut dengan jelas. Jawaban harus singkat dan, bila memungkinkan, merujuk pada pasal yang relevan.
        """
        
        # Generate response using Ollama
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response["message"]["content"].strip()
