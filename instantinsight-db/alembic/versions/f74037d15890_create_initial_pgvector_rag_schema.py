"""
Create initial pgvector RAG schema.

Revision ID: f74037d15890
Revises:
Create Date: 2025-07-28 22:33:49.426037

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# Import pgvector types
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback if pgvector not available
    Vector = sa.String

# revision identifiers, used by Alembic.
revision: str = "f74037d15890"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema - Create pgvector RAG tables and types."""
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Create enum types
    document_type_enum = postgresql.ENUM(
        "schema", "sql_example", "documentation", name="document_type", create_type=True
    )
    chunk_type_enum = postgresql.ENUM(
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

    # Create rag_documents table
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("doc_type", document_type_enum, nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("full_content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            default={},
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for rag_documents
    op.create_index("idx_documents_type", "rag_documents", ["doc_type"])
    op.create_index("idx_documents_source", "rag_documents", ["source"])
    op.create_index("idx_documents_created", "rag_documents", ["created_at"])
    op.create_index(
        "idx_documents_metadata", "rag_documents", ["metadata"], postgresql_using="gin"
    )

    # Create rag_chunks table
    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_type", chunk_type_enum, nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            default={},
        ),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["rag_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_index"
        ),
    )

    # Create indexes for rag_chunks
    op.create_index("idx_chunks_document", "rag_chunks", ["document_id"])
    op.create_index("idx_chunks_type", "rag_chunks", ["chunk_type"])
    op.create_index(
        "idx_chunks_metadata", "rag_chunks", ["metadata"], postgresql_using="gin"
    )
    op.create_index(
        "idx_chunks_content_fts",
        "rag_chunks",
        [sa.text("to_tsvector('english', content)")],
        postgresql_using="gin",
    )

    # Specialized GIN indexes for common queries
    op.create_index(
        "idx_chunks_metadata_tables",
        "rag_chunks",
        [sa.text("(metadata->'tables')")],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_chunks_metadata_columns",
        "rag_chunks",
        [sa.text("(metadata->'columns')")],
        postgresql_using="gin",
    )

    # Create rag_embeddings table
    op.create_table(
        "rag_embeddings",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=False, default="semantic"),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            default={},
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["rag_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chunk_id", "namespace", name="uq_embeddings_chunk_namespace"
        ),
    )

    # Create indexes for rag_embeddings
    op.create_index("idx_embeddings_chunk", "rag_embeddings", ["chunk_id"])
    op.create_index("idx_embeddings_namespace", "rag_embeddings", ["namespace"])

    # HNSW index for vector similarity search
    op.create_index(
        "idx_embeddings_vector_hnsw",
        "rag_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # Create trigger function for updated_at
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """
    )

    # Create trigger for updated_at on documents
    op.execute(
        """
        CREATE TRIGGER update_rag_documents_updated_at
            BEFORE UPDATE ON rag_documents
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
    )


def downgrade() -> None:
    """Downgrade schema - Drop all pgvector RAG tables and types."""
    # Drop triggers first
    op.execute(
        "DROP TRIGGER IF EXISTS update_rag_documents_updated_at ON rag_documents"
    )
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop tables (foreign keys will cascade)
    op.drop_table("rag_embeddings")
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS chunk_type")
    op.execute("DROP TYPE IF EXISTS document_type")

    # Note: We don't drop the extensions as they might be used by other parts of the system
