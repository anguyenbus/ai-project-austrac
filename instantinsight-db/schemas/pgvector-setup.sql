-- pgvector Setup V2 for nl2sql2vis RAG System with Chunking Support
-- This script sets up an improved vector storage with document chunking capabilities
-- Optimized for text-to-SQL using Vanna.ai patterns

\c instantinsight;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop old types if they exist (for clean migration)
DROP TYPE IF EXISTS document_type CASCADE;
DROP TYPE IF EXISTS chunk_type CASCADE;

-- Create enum for document types
CREATE TYPE document_type AS ENUM ('schema', 'sql_example', 'documentation');

-- Create enum for chunk types
CREATE TYPE chunk_type AS ENUM (
    'table_overview',    -- Table definition and primary info
    'columns',           -- Column definitions batch
    'constraints',       -- Foreign keys, checks, indexes
    'example_overview',  -- SQL example question + summary
    'example_ctes',      -- CTE definitions from complex queries
    'example_main',      -- Main query body
    'doc_section'        -- Documentation section
);

-- Main documents table (stores full content)
CREATE TABLE IF NOT EXISTS rag_documents (
    id SERIAL PRIMARY KEY,
    doc_type document_type NOT NULL,
    source VARCHAR(255),
    full_content TEXT NOT NULL,
    
    -- Rich metadata for the entire document
    metadata JSONB DEFAULT '{}',
    
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table with smart chunking metadata
CREATE TABLE IF NOT EXISTS rag_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_type chunk_type,
    content TEXT NOT NULL,
    
    -- Token tracking for prompt budgeting
    token_count INT,
    
    -- Chunk-specific metadata
    -- For schema chunks: {"tables": ["users"], "columns": ["id", "name"]}
    -- For SQL chunks: {"tables": ["users", "orders"], "complexity": "medium"}
    metadata JSONB DEFAULT '{}',
    
    -- Position tracking in original document
    char_start INT,
    char_end INT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- Embeddings table (simplified for Titan-only approach)
CREATE TABLE IF NOT EXISTS rag_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INT NOT NULL REFERENCES rag_chunks(id) ON DELETE CASCADE,
    
    -- Namespace for different embedding strategies
    -- 'semantic': standard embeddings
    -- 'keyword_enhanced': embeddings with keyword focus
    -- 'structured': embeddings of structured representations
    namespace TEXT NOT NULL DEFAULT 'semantic',
    
    -- Vector embedding (1024 dimensions for amazon.titan-embed-text-v2:0)
    embedding vector(1024) NOT NULL,
    
    -- Embedding metadata (processing time, confidence, etc.)
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, namespace)
);

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_type ON rag_documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_source ON rag_documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON rag_documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_created ON rag_documents(created_at);

-- Create indexes for chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document ON rag_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON rag_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON rag_chunks USING gin(metadata);

-- Specialized GIN indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_tables 
ON rag_chunks USING gin((metadata->'tables'));

CREATE INDEX IF NOT EXISTS idx_chunks_metadata_columns 
ON rag_chunks USING gin((metadata->'columns'));

-- Full-text search index for hybrid retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts 
ON rag_chunks USING gin(to_tsvector('english', content));

-- Create indexes for embeddings
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON rag_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_namespace ON rag_embeddings(namespace);

-- HNSW index for vector similarity search (most efficient for pgvector)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw
ON rag_embeddings USING hnsw (embedding vector_cosine_ops);

-- Function for chunk-based similarity search
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

-- Function for hybrid search (vector + keyword)
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

-- Function to get chunk statistics
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

-- View for easy querying of chunks with full context
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

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at on documents
CREATE TRIGGER update_rag_documents_updated_at
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (if needed for specific user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- Insert initial system record
INSERT INTO rag_documents (doc_type, source, full_content, metadata)
VALUES (
    'documentation',
    'system',
    'pgvector RAG system',
    jsonb_build_object(
        'system', 'pgvector',
        'version', '2.0',
        'features', ARRAY['chunking', 'hybrid_search', 'namespace_support'],
        'embedding_model', 'amazon.titan-embed-text-v2:0',
        'vector_dimensions', 1024
    )
) ON CONFLICT DO NOTHING;

COMMIT;