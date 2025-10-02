import enum
from app.db.base import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Enum, Integer, String, Text, UniqueConstraint


class LegalTextType(enum.Enum):
    BODY = "body"
    ELUCIDATION = "elucidation"


class LegalTextChunk(Base):
    __tablename__ = "legal_text_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)                             # e.g., "UU_22_2009_LLAJ"
    article = Column(String, nullable=False)                            # e.g., "Pasal 10"
    paragraph = Column(String, nullable=True)                           # e.g., "Ayat 1"
    chunk_type = Column(                                                # pasal or penjelasan
        Enum(LegalTextType, name="chunk_type_enum"),
        nullable=False,
        default=LegalTextType.BODY
    )
    chunk_index = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)                                 # chunk content
    embedding = Column(Vector(3072))                                    # pgvector column
    doc_key = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("source", "article", "paragraph", "chunk_type", "chunk_index"),
    )
