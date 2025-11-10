"""
SQLAlchemy models for pgvector RAG system.

These models correspond to the schema defined in schemas/pgvector-setup.sql
and are used by Alembic for database migrations.
"""

# Remove unused imports - datetime, Any, Dict, List, Optional not needed

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Import vector type from pgvector
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for environments without pgvector
    Vector = String


Base = declarative_base()


# Enum definitions matching the SQL schema
DOCUMENT_TYPE_ENUM = Enum(
    "schema", "sql_example", "documentation", name="document_type", create_type=True
)

CHUNK_TYPE_ENUM = Enum(
    "table_overview",
    "columns",
    "constraints",
    "example_overview",
    "example_ctes",
    "example_main",
    "doc_section",
    name="chunk_type",
    create_type=True,
)


class RagDocument(Base):
    """
    Main documents table storing full content.

    Corresponds to rag_documents table in pgvector-setup.sql
    """

    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_type = Column(DOCUMENT_TYPE_ENUM, nullable=False)
    source = Column(String(255), nullable=True)
    full_content = Column(Text, nullable=False)

    # Rich metadata for the entire document (using doc_metadata to avoid SQLAlchemy conflict)
    doc_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    # Tracking timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    chunks = relationship(
        "RagChunk", back_populates="document", cascade="all, delete-orphan"
    )

    # Indexes - use column object references for renamed columns
    __table_args__ = (
        Index("idx_documents_type", "doc_type"),
        Index("idx_documents_source", "source"),
        Index("idx_documents_created", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation of RagDocument."""
        return f"<RagDocument(id={self.id}, doc_type='{self.doc_type}', source='{self.source}')>"


class RagChunk(Base):
    """
    Chunks table with smart chunking metadata.

    Corresponds to rag_chunks table in pgvector-setup.sql
    """

    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(CHUNK_TYPE_ENUM, nullable=True)
    content = Column(Text, nullable=False)

    # Token tracking for prompt budgeting
    token_count = Column(Integer, nullable=True)

    # Chunk-specific metadata
    # For schema chunks: {"tables": ["users"], "columns": ["id", "name"]}
    # For SQL chunks: {"tables": ["users", "orders"], "complexity": "medium"}
    chunk_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    # Position tracking in original document
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    document = relationship("RagDocument", back_populates="chunks")
    embeddings = relationship(
        "RagEmbedding", back_populates="chunk", cascade="all, delete-orphan"
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_type", "chunk_type"),
        # Specialized GIN indexes for common queries will be added via migration
        # Full-text search index for hybrid retrieval
        Index(
            "idx_chunks_content_fts",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        """Return string representation of RagChunk."""
        return f"<RagChunk(id={self.id}, document_id={self.document_id}, chunk_type='{self.chunk_type}')>"


class RagEmbedding(Base):
    """
    Embeddings table for vector similarity search.

    Corresponds to rag_embeddings table in pgvector-setup.sql
    """

    __tablename__ = "rag_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(
        Integer, ForeignKey("rag_chunks.id", ondelete="CASCADE"), nullable=False
    )

    # Namespace for different embedding strategies
    # 'semantic': standard embeddings
    # 'keyword_enhanced': embeddings with keyword focus
    # 'structured': embeddings of structured representations
    namespace = Column(String, nullable=False, default="semantic")

    # Vector embedding (1024 dimensions for amazon.titan-embed-text-v2:0)
    embedding = Column(Vector(1024), nullable=False)

    # Embedding metadata (processing time, confidence, etc.)
    embedding_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    chunk = relationship("RagChunk", back_populates="embeddings")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("chunk_id", "namespace", name="uq_embeddings_chunk_namespace"),
        Index("idx_embeddings_chunk", "chunk_id"),
        Index("idx_embeddings_namespace", "namespace"),
        # HNSW index for vector similarity search (most efficient for pgvector)
        Index(
            "idx_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        """Return string representation of RagEmbedding."""
        return f"<RagEmbedding(id={self.id}, chunk_id={self.chunk_id}, namespace='{self.namespace}')>"


# Export all models for easy importing
__all__ = [
    "Base",
    "RagDocument",
    "RagChunk",
    "RagEmbedding",
    "DOCUMENT_TYPE_ENUM",
    "CHUNK_TYPE_ENUM",
]
