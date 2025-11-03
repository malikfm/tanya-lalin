from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base
from config import settings
from enums import LegalTextType


class LegalTextChunk(Base):
    __tablename__ = "legal_text_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Doc source, e.g., "UU_22_2009_LLAJ"
    source: Mapped[str] = mapped_column(String(64), nullable=False)

    # Article (pasal) number
    article_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Paragraph (ayat) number
    paragraph_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Content of an article/paragraph (body) or the elucidation
    chunk_type: Mapped[LegalTextType] = mapped_column(
        Enum(LegalTextType, name="chunk_type_enum"), nullable=False, default=LegalTextType.BODY
    )
    
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # chunk content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # pgvector column for embeddings
    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    
    doc_key: Mapped[Optional[str]] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        # Ensure uniqueness even with null paragraph
        UniqueConstraint(
            "source",
            "article_number",
            "paragraph_number",
            "chunk_type",
            "chunk_index",
            postgresql_nulls_not_distinct=True,
        ),
        # Cover article + paragraph lookups
        Index(None, "source", "article_number", "paragraph_number"),
        # For similarity search
        Index(
            None,
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 12, "ef_construction": 75},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        )
    )
