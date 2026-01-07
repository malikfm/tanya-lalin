import json

import ollama
from loguru import logger
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
        if not relevant_chunks:
            return "Kami tidak menemukan ketentuan hukum secara langsung berkaitan dengan pertanyaan Anda."
        
        # Format the context from the retrieved chunks
        context_parts = []
        for chunk in relevant_chunks:
            context_part = {"sumber": chunk["source"], "pasal": chunk["article_number"]}
            if chunk["paragraph_number"] is not None:
                context_part["ayat"] = chunk["paragraph_number"]
            context_part["potongan_teks"] = chunk["text"]
            context_part["jenis"] = chunk["chunk_type"]
            context_parts.append(json.dumps(context_part))
        
        context = "\n".join(context_parts)
        
        # Create the prompt with context
        prompt = (
            "Anda adalah seorang ahli dalam hukum lalu lintas Indonesia."
            " Jawablah pertanyaan pengguna secara langsung dan ringkas berdasarkan kutipan dokumen hukum yang diberikan."
            " Jangan menulis frasa pembuka seperti 'Berdasarkan dokumen yang disediakan', 'Berdasarkan konteks', atau kalimat serupa."
            " Mulailah jawaban langsung dengan isi jawaban hukum yang relevan."
            " Jika informasi yang diminta tidak terdapat dalam kutipan, nyatakan hal tersebut dengan jelas, contoh"
            " 'Informasi yang diminta tidak terdapat dalam dokumen hukum yang menjadi rujukan saya.'"
            " Gunakan double-quotes jika Anda mengutip 'sumber', 'ayat', 'paragraf', atau 'potongan_teks' dari konteks."
            " Gunakan double-quotes juga jika Anda ingin menyoroti istilah penting dalam jawaban Anda."
            " Jawaban harus menggunakan bahasa Indonesia formal."
            " Setiap istilah dalam konteks yang bukan bahasa Indonesia (kecuali bagian 'potongan_teks') harus diterjemahkan ke padanan istilah hukum yang sesuai."
            " Bila memungkinkan, sertakan rujukan ayat atau pasal yang relevan."
            "\n\n---\n\n"
            "Kutipan dokumen hukum:"
            f"\n{context}"
            "\n\n---\n\n"
            "Pertanyaan pengguna:"
            f"\n{query}"
            "\n\n---\n\n"
            "Silakan berikan jawaban akhir sesuai instruksi di atas."
        )

        logger.debug(f"Query: {query}")
        logger.debug(f"Context: {context}")

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
        
        return response["message"]["content"].strip().replace("*", "")
