-- High-Cardinality Categorical Value Embeddings for RAG System
-- This schema extension adds support for storing and searching categorical values
-- from database columns to improve text-to-SQL query understanding
-- 
-- Usage:
--   psql -d instantinsight -f schemas/pgvector-cardinality.sql
--   python scripts/create_rag_cardinality.py --mode all

\c instantinsight;

-- Create enum for cardinality tiers
DROP TYPE IF EXISTS cardinality_tier CASCADE;
CREATE TYPE cardinality_tier AS ENUM ('binary', 'low', 'medium', 'high', 'very_high');

-- Create enum for detection methods
DROP TYPE IF EXISTS detection_method CASCADE;
CREATE TYPE detection_method AS ENUM ('explicit_type', 'partition_key', 'cardinality_ratio', 'manual');

-- Create enum for embedding status
DROP TYPE IF EXISTS embedding_status CASCADE;
CREATE TYPE embedding_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'skipped');

-- Table for tracking which columns have been analysed for categorical detection
-- This allows incremental processing and avoids re-analysis
CREATE TABLE IF NOT EXISTS rag_cardinality_columns (
    id                SERIAL PRIMARY KEY,
    schema_name       TEXT NOT NULL,
    table_name        TEXT NOT NULL,
    column_name       TEXT NOT NULL,
    column_type       TEXT NOT NULL,                    -- Athena/Glue data type
    is_categorical    BOOLEAN DEFAULT FALSE,
    cardinality_tier  cardinality_tier,
    detection_method  detection_method,
    confidence_score  FLOAT DEFAULT 0.0,               -- Detection confidence (0.0-1.0)
    distinct_count    BIGINT,                           -- Number of distinct values found
    total_rows        BIGINT,                           -- Total rows sampled
    distinct_ratio    FLOAT,                            -- distinct_count / total_rows
    last_analysed     TIMESTAMPTZ,
    metadata          JSONB DEFAULT '{}',               -- Additional analysis metadata
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(schema_name, table_name, column_name)
);

-- Main table for storing categorical values and their embeddings
CREATE TABLE IF NOT EXISTS rag_cardinality (
    id                BIGSERIAL PRIMARY KEY,
    column_id         INTEGER REFERENCES rag_cardinality_columns(id) ON DELETE CASCADE,
    schema_name       TEXT NOT NULL,
    table_name        TEXT NOT NULL,
    column_name       TEXT NOT NULL,
    category          TEXT NOT NULL,                    -- Original value as stored in database
    category_norm     TEXT NOT NULL,                    -- Normalised for matching (lowercase, trimmed)
    embedding         vector(1024),                     -- Amazon Titan embedding vector
    embedding_model   TEXT DEFAULT 'amazon.titan-embed-text-v2:0',
    embedding_status  embedding_status DEFAULT 'pending',
    frequency         BIGINT DEFAULT 0,                 -- How often this value appears in source table
    usage_count       BIGINT DEFAULT 0,                 -- How often it's been used in queries
    metadata          JSONB DEFAULT '{}',               -- Additional metadata (sampling info, etc.)
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure uniqueness at the exact category level per column (allows case variations)
    UNIQUE(schema_name, table_name, column_name, category)
);

-- Create indexes for rag_cardinality_columns (for fast column lookup)
CREATE INDEX IF NOT EXISTS idx_cardinality_columns_categorical 
    ON rag_cardinality_columns(is_categorical) 
    WHERE is_categorical = TRUE;

CREATE INDEX IF NOT EXISTS idx_cardinality_columns_tier 
    ON rag_cardinality_columns(cardinality_tier);

CREATE INDEX IF NOT EXISTS idx_cardinality_columns_schema_table 
    ON rag_cardinality_columns(schema_name, table_name);

CREATE INDEX IF NOT EXISTS idx_cardinality_columns_last_analysed 
    ON rag_cardinality_columns(last_analysed);

-- Create indexes for rag_cardinality (for fast category lookup and embedding search)
CREATE INDEX IF NOT EXISTS idx_cardinality_schema_table_column 
    ON rag_cardinality(schema_name, table_name, column_name);

CREATE INDEX IF NOT EXISTS idx_cardinality_column_id 
    ON rag_cardinality(column_id);

CREATE INDEX IF NOT EXISTS idx_cardinality_category_norm 
    ON rag_cardinality(category_norm);

CREATE INDEX IF NOT EXISTS idx_cardinality_embedding_status 
    ON rag_cardinality(embedding_status);

CREATE INDEX IF NOT EXISTS idx_cardinality_frequency_desc 
    ON rag_cardinality(frequency DESC);

-- HNSW index for efficient vector similarity search (core functionality)
CREATE INDEX IF NOT EXISTS idx_cardinality_embedding_hnsw
    ON rag_cardinality USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- Function for categorical value similarity search
-- This is the main function used by the RAG system to find matching categories
CREATE OR REPLACE FUNCTION search_categorical_values(
    query_embedding vector(1024),
    schema_filter TEXT DEFAULT NULL,
    table_filter TEXT[] DEFAULT NULL,
    column_filter TEXT[] DEFAULT NULL,
    tier_filter cardinality_tier[] DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    schema_name TEXT,
    table_name TEXT,
    column_name TEXT,
    category TEXT,
    category_norm TEXT,
    similarity FLOAT,
    frequency BIGINT,
    cardinality_tier cardinality_tier
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rc.id,
        rc.schema_name,
        rc.table_name,
        rc.column_name,
        rc.category,
        rc.category_norm,
        1 - (rc.embedding <=> query_embedding) as similarity,
        rc.frequency,
        cc.cardinality_tier
    FROM rag_cardinality rc
    JOIN rag_cardinality_columns cc ON rc.column_id = cc.id
    WHERE 
        rc.embedding IS NOT NULL
        AND rc.embedding_status = 'completed'
        AND (schema_filter IS NULL OR rc.schema_name = schema_filter)
        AND (table_filter IS NULL OR rc.table_name = ANY(table_filter))
        AND (column_filter IS NULL OR rc.column_name = ANY(column_filter))
        AND (tier_filter IS NULL OR cc.cardinality_tier = ANY(tier_filter))
        AND 1 - (rc.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY rc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to get cardinality statistics for monitoring and debugging
CREATE OR REPLACE FUNCTION get_cardinality_statistics()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    column_name TEXT,
    cardinality_tier cardinality_tier,
    total_categories BIGINT,
    embedded_categories BIGINT,
    pending_categories BIGINT,
    failed_categories BIGINT
)
LANGUAGE sql
AS $$
    SELECT 
        rc.schema_name,
        rc.table_name,
        rc.column_name,
        cc.cardinality_tier,
        COUNT(*) as total_categories,
        COUNT(*) FILTER (WHERE rc.embedding_status = 'completed') as embedded_categories,
        COUNT(*) FILTER (WHERE rc.embedding_status = 'pending') as pending_categories,
        COUNT(*) FILTER (WHERE rc.embedding_status = 'failed') as failed_categories
    FROM rag_cardinality rc
    JOIN rag_cardinality_columns cc ON rc.column_id = cc.id
    GROUP BY rc.schema_name, rc.table_name, rc.column_name, cc.cardinality_tier
    ORDER BY rc.schema_name, rc.table_name, rc.column_name;
$$;

-- Function to update category usage tracking (for query analytics)
CREATE OR REPLACE FUNCTION update_category_usage(
    p_schema_name TEXT,
    p_table_name TEXT,
    p_column_name TEXT,
    p_category_norm TEXT,
    p_increment BIGINT DEFAULT 1
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE rag_cardinality 
    SET 
        usage_count = usage_count + p_increment,
        updated_at = NOW()
    WHERE 
        schema_name = p_schema_name
        AND table_name = p_table_name
        AND column_name = p_column_name
        AND category_norm = p_category_norm;
END;
$$;

-- Function to get tier distribution for system monitoring
CREATE OR REPLACE FUNCTION get_tier_distribution()
RETURNS TABLE (
    tier cardinality_tier,
    column_count BIGINT,
    avg_categories BIGINT,
    total_embeddings BIGINT
)
LANGUAGE sql
AS $$
    SELECT 
        cc.cardinality_tier as tier,
        COUNT(DISTINCT cc.id) as column_count,
        AVG(subquery.category_count) as avg_categories,
        SUM(subquery.embedded_count) as total_embeddings
    FROM rag_cardinality_columns cc
    LEFT JOIN (
        SELECT 
            column_id,
            COUNT(*) as category_count,
            COUNT(*) FILTER (WHERE embedding_status = 'completed') as embedded_count
        FROM rag_cardinality
        GROUP BY column_id
    ) subquery ON cc.id = subquery.column_id
    WHERE cc.is_categorical = TRUE
    GROUP BY cc.cardinality_tier
    ORDER BY 
        CASE cc.cardinality_tier
            WHEN 'binary' THEN 1
            WHEN 'low' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'high' THEN 4
            WHEN 'very_high' THEN 5
        END;
$$;

-- Trigger function for updated_at timestamp
CREATE OR REPLACE FUNCTION update_cardinality_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update timestamps
CREATE TRIGGER trigger_rag_cardinality_columns_updated_at
    BEFORE UPDATE ON rag_cardinality_columns
    FOR EACH ROW EXECUTE FUNCTION update_cardinality_updated_at();

CREATE TRIGGER trigger_rag_cardinality_updated_at
    BEFORE UPDATE ON rag_cardinality
    FOR EACH ROW EXECUTE FUNCTION update_cardinality_updated_at();

-- View for easy querying of processed columns and their status
CREATE OR REPLACE VIEW v_cardinality_summary AS
SELECT 
    cc.schema_name,
    cc.table_name,
    cc.column_name,
    cc.column_type,
    cc.is_categorical,
    cc.cardinality_tier,
    cc.confidence_score,
    cc.distinct_count,
    cc.detection_method,
    cc.last_analysed,
    COALESCE(stats.stored_values, 0) as stored_values,
    COALESCE(stats.embedded_values, 0) as embedded_values,
    COALESCE(stats.pending_embeddings, 0) as pending_embeddings,
    COALESCE(stats.failed_embeddings, 0) as failed_embeddings,
    stats.last_embedding_update
FROM rag_cardinality_columns cc
LEFT JOIN (
    SELECT 
        column_id,
        COUNT(*) as stored_values,
        COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embedded_values,
        COUNT(*) FILTER (WHERE embedding_status = 'pending') as pending_embeddings,
        COUNT(*) FILTER (WHERE embedding_status = 'failed') as failed_embeddings,
        MAX(updated_at) as last_embedding_update
    FROM rag_cardinality
    GROUP BY column_id
) stats ON cc.id = stats.column_id
ORDER BY cc.schema_name, cc.table_name, cc.column_name;

-- Grant permissions (uncomment and adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_rag_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_rag_user;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO your_rag_user;

COMMIT;

-- Insert a system record to track schema version
INSERT INTO rag_cardinality_columns (
    schema_name, table_name, column_name, column_type, 
    is_categorical, detection_method, confidence_score, 
    last_analysed, metadata
)
VALUES (
    'system', 'cardinality_schema', 'version', 'varchar',
    false, 'manual', 1.0,
    NOW(),
    jsonb_build_object(
        'schema_version', '1.0.0',
        'features', ARRAY['categorical_detection', 'embedding_generation', 'similarity_search'],
        'embedding_model', 'amazon.titan-embed-text-v2:0',
        'vector_dimensions', 1024
    )
) ON CONFLICT (schema_name, table_name, column_name) DO NOTHING;