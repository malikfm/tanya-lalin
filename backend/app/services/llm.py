"""LLM service for generating responses based on retrieved context."""
from loguru import logger

from app.core.gemini_client import GeminiClient
from app.constants import ResponseMessages


SYSTEM_INSTRUCTION = f"""Anda adalah asisten ahli hukum lalu lintas Indonesia. Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan kutipan dokumen hukum yang diberikan.

ATURAN PENTING:
1. Jawab langsung dan ringkas tanpa frasa pembuka seperti "Berdasarkan dokumen...", "Menurut konteks...", atau sejenisnya.
2. Gunakan bahasa Indonesia formal dan mudah dipahami masyarakat umum.
3. Kutip nomor pasal dan ayat yang relevan, contoh: Pasal 106 ayat (4).
4. Jika informasi tidak ditemukan dalam konteks, nyatakan dengan jelas: "{ResponseMessages.NOT_FOUND}"
5. JANGAN mengarang atau menambahkan informasi yang tidak ada dalam konteks.
6. Jika ada sanksi pidana, sebutkan dengan jelas (kurungan, denda).
7. Gunakan format yang mudah dibaca (paragraf pendek, poin-poin jika perlu)."""


class LLMService:
    """Service for generating LLM responses."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
    
    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as context for the LLM.
        
        Args:
            chunks: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "Tidak ada dokumen yang relevan ditemukan."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "UU LLAJ")
            article = chunk.get("article_number", "?")
            paragraph = chunk.get("paragraph_number")
            chunk_type = chunk.get("chunk_type", "body")
            text = chunk.get("text", "")
            
            # Format reference
            if paragraph:
                ref = f"Pasal {article} ayat ({paragraph})"
            else:
                ref = f"Pasal {article}"
            
            type_label = "Penjelasan" if chunk_type == "elucidation" else "Isi"
            
            context_parts.append(f"[{i}] {ref} ({type_label}):\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _format_conversation_history(self, messages: list[dict]) -> str:
        """Format conversation history for context.
        
        Args:
            messages: List of previous messages
            
        Returns:
            Formatted conversation history
        """
        if not messages:
            return ""
        
        history_parts = []
        for msg in messages[-6:]:  # Last 6 messages (3 turns)
            role = "Pengguna" if msg["role"] == "user" else "Asisten"
            content = msg["content"]
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    async def generate_response(
        self,
        query: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict] | None = None
    ) -> str:
        """Generate a response based on user query and retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of relevant document chunks
            conversation_history: Optional previous conversation messages
            
        Returns:
            Generated response text
        """
        if not retrieved_chunks:
            return ResponseMessages.NO_RELEVANT_CHUNKS
        
        # Format context
        context = self._format_context(retrieved_chunks)
        
        # Format conversation history
        history = ""
        if conversation_history:
            history = self._format_conversation_history(conversation_history)
            history = f"\n\nRIWAYAT PERCAKAPAN:\n{history}\n"
        
        # Build prompt
        prompt = (
            "KUTIPAN DOKUMEN HUKUM LALU LINTAS:"
            f"\n{context}"
            f"\n{history}"
            "\nPERTANYAAN PENGGUNA:"
            f"\n{query}"
            "\n\nJawab pertanyaan di atas berdasarkan kutipan dokumen hukum yang diberikan."
            " Ikuti aturan yang telah ditetapkan."
        )

        logger.debug(f"Generating response for query: {query[:50]}...")
        
        try:
            response = await self.gemini_client.generate_text(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3
            )
            
            # Clean up response
            response = response.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ResponseMessages.ERROR
