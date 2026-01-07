"""Query rewriter service for transforming user queries to legal terminology."""
import json
import re
from loguru import logger

from app.core.gemini_client import GeminiClient


# Common term mappings (Indonesian everyday language -> legal terms)
TERM_MAPPINGS = {
    "lampu merah": ["Alat Pemberi Isyarat Lalu Lintas", "APILL", "isyarat lalu lintas"],
    "lampu lalu lintas": ["Alat Pemberi Isyarat Lalu Lintas", "APILL"],
    "traffic light": ["Alat Pemberi Isyarat Lalu Lintas", "APILL"],
    "menerobos": ["melanggar aturan perintah atau larangan", "tidak mematuhi"],
    "ngebut": ["batas kecepatan", "kecepatan maksimal", "kecepatan tinggi"],
    "ugal-ugalan": ["membahayakan", "keselamatan lalu lintas"],
    "sim": ["Surat Izin Mengemudi"],
    "stnk": ["Surat Tanda Nomor Kendaraan Bermotor"],
    "bpkb": ["Buku Pemilik Kendaraan Bermotor"],
    "helm": ["helm standar nasional Indonesia"],
    "sabuk pengaman": ["sabuk keselamatan"],
    "parkir sembarangan": ["parkir", "larangan parkir", "tempat parkir"],
    "bahu jalan": ["bahu Jalan", "badan jalan"],
    "trotoar": ["trotoar", "fasilitas Pejalan Kaki", "fasilitas pejalan kaki"],
    "menyalip": ["melewati Kendaraan", "mendahului"],
    "zigzag": ["gerakan lalu lintas", "manuver berbahaya"],
    "motor": ["Sepeda Motor", "Kendaraan Bermotor"],
    "mobil": ["Kendaraan Bermotor", "kendaraan beroda empat"],
    "tilang": ["pidana", "denda", "pelanggaran"],
    "denda": ["pidana denda", "sanksi"],
    "kaca spion": ["kaca spion", "spion"],
    "plat nomor": ["Tanda Nomor Kendaraan Bermotor", "TNKB"],
    "lawan arah": ["melawan arus", "arah lalu lintas"],
    "jalur busway": ["lajur khusus", "jalur khusus"],
    "zebra cross": ["tempat penyeberangan", "penyeberangan Pejalan Kaki"],
}

# Specific patterns that need custom handling
SPECIAL_PATTERNS = {
    "menerobos lampu merah": "Setiap orang yang mengemudikan Kendaraan Bermotor di Jalan yang melanggar aturan perintah atau larangan yang dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas dipidana",
    "parkir di trotoar": "fasilitas Pejalan Kaki trotoar parkir larangan",
    "bahu jalan saat macet": "bahu Jalan Kendaraan keadaan darurat lalu lintas",
    "helm": "Sepeda Motor helm standar nasional Indonesia dipidana",
    "sabuk pengaman": "sabuk keselamatan Pengemudi Penumpang dipidana",
    "menyalip zigzag": "gerakan lalu lintas melewati Kendaraan berbahaya dipidana",
}

SYSTEM_INSTRUCTION = """Anda adalah ahli hukum lalu lintas Indonesia yang sangat memahami UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan (LLAJ).

Tugas Anda adalah menganalisis pertanyaan pengguna dalam bahasa sehari-hari dan mengubahnya menjadi kalimat pencarian yang sangat mirip dengan teks dalam UU LLAJ.

PENTING: 
- "lampu merah" dalam bahasa sehari-hari = "Alat Pemberi Isyarat Lalu Lintas" dalam UU
- "menerobos lampu merah" = "melanggar aturan perintah atau larangan yang dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas"
- Jika tentang sanksi/denda, tambahkan kata "dipidana" dan "denda"

Berikan output dalam format JSON valid (tanpa markdown code block):
{
    "legal_search_query": "kalimat panjang yang mirip dengan teks dalam UU LLAJ untuk pencarian semantik",
    "key_legal_phrases": ["frasa hukum 1", "frasa hukum 2"]
}

Pastikan legal_search_query adalah kalimat yang PANJANG dan DETAIL, bukan hanya istilah-istilah terpisah."""


class QueryRewriter:
    """Service for rewriting user queries with legal terminology."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
    
    def _apply_term_mappings(self, query: str) -> list[str]:
        """Apply static term mappings to expand query.
        
        Args:
            query: Original user query
            
        Returns:
            List of expanded terms
        """
        query_lower = query.lower()
        expanded_terms = []
        
        for everyday_term, legal_terms in TERM_MAPPINGS.items():
            if everyday_term in query_lower:
                expanded_terms.extend(legal_terms)
        
        return expanded_terms
    
    def _get_special_pattern(self, query: str) -> str | None:
        """Check if query matches a special pattern that needs custom handling."""
        query_lower = query.lower()
        for pattern, legal_query in SPECIAL_PATTERNS.items():
            if pattern in query_lower:
                return legal_query
        return None
    
    async def rewrite_query(self, query: str) -> dict:
        """Rewrite user query with legal terminology.
        
        Args:
            query: Original user query in everyday language
            
        Returns:
            Dictionary with legal search query and key phrases
        """
        # Check for special patterns first
        special_query = self._get_special_pattern(query)
        
        # Apply static term mappings
        static_terms = self._apply_term_mappings(query)
        
        # Use LLM for dynamic rewriting
        prompt = (
            f"Pertanyaan pengguna: {query}"
            "\n\nUbah pertanyaan di atas menjadi kalimat pencarian yang sangat mirip dengan teks dalam UU LLAJ."
            "\nBuat kalimat yang PANJANG dan DETAIL."
        )

        try:
            response = await self.gemini_client.generate_text(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2
            )
            
            # Parse JSON response
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r'^```(?:json)?\n?', '', cleaned_response)
                cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
            
            result = json.loads(cleaned_response)
            
            legal_search_query = result.get("legal_search_query", "")
            key_phrases = result.get("key_legal_phrases", [])
            
            # Combine with special pattern if available
            if special_query:
                legal_search_query = f"{special_query} {legal_search_query}"
            
            return {
                "original_query": query,
                "legal_search_query": legal_search_query,
                "key_phrases": key_phrases,
                "static_terms": static_terms
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fallback
            return {
                "original_query": query,
                "legal_search_query": special_query or " ".join([query] + static_terms),
                "key_phrases": static_terms,
                "static_terms": static_terms
            }
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            return {
                "original_query": query,
                "legal_search_query": special_query or " ".join([query] + static_terms),
                "key_phrases": static_terms,
                "static_terms": static_terms
            }
    
    def build_expanded_query(self, rewrite_result: dict) -> str:
        """Build the primary search query from rewrite result."""
        # The legal_search_query is now the main search query
        return rewrite_result.get("legal_search_query", rewrite_result.get("original_query", ""))
    
    def get_additional_queries(self, rewrite_result: dict) -> list[str]:
        """Get additional queries for multi-query search."""
        queries = []
        
        # Add key phrases as separate queries
        for phrase in rewrite_result.get("key_phrases", []):
            if phrase:
                queries.append(phrase)
        
        # Add static terms as a combined query
        static_terms = rewrite_result.get("static_terms", [])
        if static_terms:
            queries.append(" ".join(static_terms[:5]))
        
        return queries[:5]  # Limit to 5 additional queries
