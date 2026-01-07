"""Application constants."""


class ResponseMessages:
    """Static response messages used by the application."""
    
    # Message returned by LLM when info not found (must match SYSTEM_INSTRUCTION)
    NOT_FOUND = (
        "Maaf, informasi mengenai hal tersebut tidak ditemukan dalam "
        "dokumen hukum yang menjadi rujukan saya."
    )
    
    # Message returned when no relevant chunks are retrieved
    NO_RELEVANT_CHUNKS = (
        "Maaf, saya tidak menemukan informasi yang relevan dalam "
        "dokumen hukum lalu lintas untuk menjawab pertanyaan Anda. "
        "Silakan coba dengan pertanyaan yang berbeda atau lebih spesifik."
    )
