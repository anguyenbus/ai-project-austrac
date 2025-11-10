"""
Add custom pgvector search functions.

Revision ID: d11b8ed78fb7
Revises: f74037d15890
Create Date: 2025-07-28 22:34:36.334217

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d11b8ed78fb7"
down_revision: str | Sequence[str] | None = "f74037d15890"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add custom pgvector search functions."""
    # Function for chunk-based similarity search
    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunk_similarity_search(
            query_embedding vector(1024),
            match_threshold float DEFAULT 0.7,
            match_count int DEFAULT 5,
            filter_types chunk_type[] DEFAULT NULL,
            filter_tables text[] DEFAULT NULL,
            namespace_filter text DEFAULT 'semantic'
        )
        RETURNS TABLE (
            chunk_id integer,
            document_id integer,
            content text,
            chunk_type chunk_type,
            doc_type document_type,
            metadata jsonb,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content,
                c.chunk_type,
                d.doc_type,
                c.metadata,
                1 - (e.embedding <=> query_embedding) as similarity
            FROM rag_chunks c
            JOIN rag_embeddings e ON c.id = e.chunk_id
            JOIN rag_documents d ON c.document_id = d.id
            WHERE 
                e.namespace = namespace_filter
                AND (filter_types IS NULL OR c.chunk_type = ANY(filter_types))
                AND (filter_tables IS NULL OR c.metadata->'tables' ?| filter_tables)
                AND 1 - (e.embedding <=> query_embedding) >= match_threshold
            ORDER BY e.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """
    )

    # Function for hybrid search (vector + keyword)
    op.execute(
        """
        CREATE OR REPLACE FUNCTION hybrid_chunk_search(
            query_embedding vector(1024),
            keyword_query text,
            vector_weight float DEFAULT 0.7,
            keyword_weight float DEFAULT 0.3,
            match_count int DEFAULT 5
        )
        RETURNS TABLE (
            chunk_id integer,
            content text,
            chunk_type chunk_type,
            metadata jsonb,
            combined_score float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            WITH vector_results AS (
                SELECT 
                    c.id,
                    c.content,
                    c.chunk_type,
                    c.metadata,
                    1 - (e.embedding <=> query_embedding) as vector_score
                FROM rag_chunks c
                JOIN rag_embeddings e ON c.id = e.chunk_id
                WHERE e.namespace = 'semantic'
                ORDER BY e.embedding <=> query_embedding
                LIMIT match_count * 2
            ),
            keyword_results AS (
                SELECT 
                    c.id,
                    c.content,
                    c.chunk_type,
                    c.metadata,
                    ts_rank(to_tsvector('english', c.content), 
                           plainto_tsquery('english', keyword_query)) as keyword_score
                FROM rag_chunks c
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', keyword_query)
                ORDER BY keyword_score DESC
                LIMIT match_count * 2
            )
            SELECT 
                COALESCE(v.id, k.id) as chunk_id,
                COALESCE(v.content, k.content) as content,
                COALESCE(v.chunk_type, k.chunk_type) as chunk_type,
                COALESCE(v.metadata, k.metadata) as metadata,
                COALESCE(v.vector_score * vector_weight, 0) + 
                COALESCE(k.keyword_score * keyword_weight, 0) as combined_score
            FROM vector_results v
            FULL OUTER JOIN keyword_results k ON v.id = k.id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $$;
    """
    )

    # Function to get chunk statistics
    op.execute(
        """
        CREATE OR REPLACE FUNCTION get_chunk_statistics()
        RETURNS TABLE (
            doc_type document_type,
            chunk_type chunk_type,
            chunk_count bigint,
            avg_token_count numeric,
            total_embeddings bigint
        )
        LANGUAGE sql
        AS $$
            SELECT 
                d.doc_type,
                c.chunk_type,
                COUNT(DISTINCT c.id) as chunk_count,
                AVG(c.token_count) as avg_token_count,
                COUNT(DISTINCT e.id) as total_embeddings
            FROM rag_documents d
            JOIN rag_chunks c ON d.id = c.document_id
            LEFT JOIN rag_embeddings e ON c.id = e.chunk_id
            GROUP BY d.doc_type, c.chunk_type
            ORDER BY d.doc_type, c.chunk_type;
        $$;
    """
    )

    # Create view for easy querying of chunks with full context
    op.execute(
        """
        CREATE OR REPLACE VIEW rag_chunks_view AS
        SELECT 
            c.id as chunk_id,
            c.document_id,
            c.chunk_index,
            c.chunk_type,
            c.content,
            c.token_count,
            c.metadata as chunk_metadata,
            d.doc_type,
            d.source,
            d.metadata as doc_metadata,
            CASE 
                WHEN EXISTS (SELECT 1 FROM rag_embeddings e WHERE e.chunk_id = c.id)
                THEN true 
                ELSE false 
            END as has_embedding,
            c.created_at
        FROM rag_chunks c
        JOIN rag_documents d ON c.document_id = d.id;
    """
    )

    # Insert initial system record
    op.execute(
        """
        INSERT INTO rag_documents (doc_type, source, full_content, metadata)
        VALUES (
            'documentation',
            'system',
            'pgvector RAG system V2 initialized with chunking support for nl2sql2vis project',
            '{"system": "pgvector", "version": "2.0", "features": ["chunking", "hybrid_search", "namespace_support"], "embedding_model": "amazon.titan-embed-text-v2:0", "vector_dimensions": 1024}'::jsonb
        ) ON CONFLICT DO NOTHING;
    """
    )


def downgrade() -> None:
    """Remove custom pgvector search functions."""
    # Drop view
    op.execute("DROP VIEW IF EXISTS rag_chunks_view")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS get_chunk_statistics()")
    op.execute(
        "DROP FUNCTION IF EXISTS hybrid_chunk_search(vector(1024), text, float, float, int)"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS chunk_similarity_search(vector(1024), float, int, chunk_type[], text[], text)"
    )

    # Remove initial system record
    op.execute(
        "DELETE FROM rag_documents WHERE source = 'system' AND doc_type = 'documentation'"
    )
