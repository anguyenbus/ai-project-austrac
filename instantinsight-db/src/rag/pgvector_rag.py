"""
PgvectorRAG: Enhanced PostgreSQL pgvector-based RAG with Chunking Support.

This is an enhanced version of the PgvectorRAG system that supports document chunking
for improved text-to-SQL generation. Features both chunked and legacy document-level modes.
"""

import json
import time
from typing import Any

# Use instructor with Bedrock for structured response
import psycopg
from langchain_aws import BedrockEmbeddings
from loguru import logger
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

# get database from anthena config
from src.config.database_config import ATHENA_CONFIG

# Import chunking strategies
from .components.chunking_strategies import ChunkData, chunk_document

DATABASE_NAME = ATHENA_CONFIG.get("database")


class PgvectorRAG:
    """
    Enhanced PostgreSQL pgvector-based RAG system with optimized chunked search.

    Features:
    - Document chunking for better granular retrieval
    - Hybrid search (vector + keyword)
    - Namespace support for different embedding strategies
    - Optimized database functions for better performance
    - Backward compatibility with legacy document-level mode
    - Optimized for text-to-SQL use cases
    """

    def __init__(
        self,
        connection_string: str,
        bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
        enable_chunking: bool = False,  # Disabled by default
        chunk_strategies: dict[str, dict] | None = None,
        default_namespace: str = "semantic",
        # pgvector optimization parameters
        enable_optimized_search: bool = True,
        search_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the enhanced pgvector RAG system with optimized chunked search.

        Args:
            connection_string: PostgreSQL connection string
            bedrock_embedding_model: AWS Bedrock embedding model ID
            dimension: Vector dimension (1024 for Titan v2)
            enable_chunking: Whether to use document chunking (default: False)
            chunk_strategies: Custom chunking strategy configurations
            default_namespace: Default namespace for embeddings
            enable_optimized_search: Whether to use optimized database functions
            search_config: Search optimization configuration

        """
        self.connection_string = connection_string
        self.bedrock_embedding_model = bedrock_embedding_model
        self.dimension = dimension
        self.enable_chunking = enable_chunking
        self.default_namespace = default_namespace

        # pgvector optimization settings
        self.enable_optimized_search = enable_optimized_search
        self.search_config = search_config or {
            "hnsw_ef_search": 100,
            "use_function_search": True,
            "parallel_workers": 4,
            "similarity_threshold": 0.3,
        }

        # Database connection
        self.db_connection = None

        # Embeddings model
        self.embeddings = None

        # Chunking configurations
        self.chunk_strategies = chunk_strategies or {
            "schema": {"max_columns_per_chunk": 12, "include_overview": True},
            "sql_example": {"include_ctes": True, "max_sql_length": 1000},
            "documentation": {"chunk_size": 1000, "overlap": 200},
        }

        # Statistics
        self.stats = {
            "documents_added": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "queries_processed": 0,
        }

        logger.info(
            f"🔧 Initializing PgvectorRAG with chunking={'enabled' if enable_chunking else 'disabled'}"
        )
        logger.info(
            f"🚀 Optimized search: {'enabled' if enable_optimized_search else 'disabled'}"
        )

    def connect_to_database(self) -> bool:
        """Connect to PostgreSQL database and verify pgvector setup."""
        try:
            self.db_connection = psycopg.connect(
                self.connection_string, row_factory=dict_row
            )
            register_vector(self.db_connection)

            # Test connection
            with self.db_connection.cursor() as cur:
                cur.execute("SELECT 1")

                # Verify pgvector optimization functions if enabled
                if self.enable_optimized_search:
                    self._verify_optimization_functions()

            logger.info("✓ Connected to PostgreSQL with pgvector support")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def _verify_optimization_functions(self) -> bool:
        """Verify that pgvector optimization functions are available."""
        try:
            with self.db_connection.cursor() as cur:
                # Check if pgvector extension exists
                cur.execute(
                    """
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname = 'vector'
                """
                )

                extensions = {
                    row["extname"]: row["extversion"] for row in cur.fetchall()
                }

                if "vector" not in extensions:
                    logger.warning("⚠️ pgvector extension not found")
                    self.enable_optimized_search = False
                    return False

                # Check for optimization functions
                cur.execute(
                    """
                    SELECT proname 
                    FROM pg_proc 
                    WHERE proname IN ('chunk_similarity_search', 'hybrid_chunk_search')
                """
                )

                functions = [row["proname"] for row in cur.fetchall()]

                if "chunk_similarity_search" not in functions:
                    logger.warning(
                        "⚠️ chunk_similarity_search function not found - falling back to basic search"
                    )
                    self.enable_optimized_search = False
                    return False

                logger.info(f"✓ pgvector verified: vector v{extensions['vector']}")
                logger.info(
                    f"✓ Optimization functions available: {', '.join(functions)}"
                )

                return True

        except Exception as e:
            logger.warning(f"Optimization verification failed: {e}")
            self.enable_optimized_search = False
            return False

    def initialize_embeddings(
        self, aws_region: str = "ap-southeast-2", aws_profile: str = None
    ) -> bool:
        """Initialize Bedrock embeddings."""
        try:
            self.embeddings = BedrockEmbeddings(
                model_id=self.bedrock_embedding_model,
                region_name=aws_region,
                credentials_profile_name=aws_profile,
            )

            # Test embeddings with basic call (avoiding circular dependency)
            try:
                test_embedding = self.embeddings.embed_query("test")
                if len(test_embedding) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(test_embedding)}, expected {self.dimension}"
                    )
            except RecursionError as e:
                logger.error(
                    f"Recursion error during embedding initialization test: {e}"
                )
                raise ValueError(
                    "Embedding model has recursion issues - initialization failed"
                ) from e

            logger.info(
                f"✓ Bedrock embeddings initialized: {self.bedrock_embedding_model}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False

    def _safe_embed_query(self, text: str, max_retries: int = 3) -> list[float] | None:
        """
        Safely generate embeddings with error handling and retries.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            List of embeddings or None if failed

        """
        import time

        for attempt in range(max_retries):
            try:
                if not self.embeddings:
                    logger.error("Embeddings not initialized")
                    return None

                # Truncate very long text to prevent issues
                if len(text) > 8000:
                    text = text[:8000] + "..."
                    logger.warning("Truncated text to 8000 characters for embedding")

                embedding = self.embeddings.embed_query(text)

                # Validate embedding
                if not embedding or len(embedding) != self.dimension:
                    logger.warning(
                        f"Invalid embedding: expected {self.dimension} dimensions, got {len(embedding) if embedding else 0}"
                    )
                    return None

                return embedding

            except RecursionError as e:
                logger.error(
                    f"Recursion error in embedding generation (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                return None

            except Exception as e:
                logger.error(f"Error generating embedding (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

        logger.error(f"Failed to generate embedding after {max_retries} attempts")
        return None

    def add_document(
        self,
        content: str,
        doc_type: str,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a document to the knowledge base with optional chunking.

        Args:
            content: Document content
            doc_type: Type of document ('schema', 'sql_example', 'documentation')
            source: Source of the document
            metadata: Additional metadata
            namespace: Embedding namespace (defaults to default_namespace)

        Returns:
            Dictionary with results including document_id, chunks_created, etc.

        """
        if not self.db_connection or not self.embeddings:
            return {"error": "Database or embeddings not initialized"}

        if metadata is None:
            metadata = {}

        if namespace is None:
            namespace = self.default_namespace

        try:
            with self.db_connection.cursor() as cur:
                # 1. Insert main document
                cur.execute(
                    """
                    INSERT INTO rag_documents (doc_type, source, full_content, metadata)
                    VALUES (%(doc_type)s, %(source)s, %(content)s, %(metadata)s)
                    RETURNING id
                """,
                    {
                        "doc_type": doc_type,
                        "source": source,
                        "content": content,
                        "metadata": json.dumps(metadata),
                    },
                )

                result = cur.fetchone()
                if not result:
                    return {"error": "Failed to insert document"}

                doc_id = result["id"]

                # 2. Create chunks if enabled
                chunks_created = 0
                embeddings_created = 0

                if self.enable_chunking:
                    chunks = self._create_chunks(content, doc_type, metadata)

                    for i, chunk in enumerate(chunks):
                        chunk_id = self._insert_chunk(cur, doc_id, i, chunk)
                        if chunk_id:
                            chunks_created += 1

                            # Generate and store embedding for chunk
                            try:
                                embedding = self._safe_embed_query(chunk.content)
                                if embedding:
                                    self._insert_embedding(
                                        cur, chunk_id, embedding, namespace
                                    )
                                    embeddings_created += 1
                                else:
                                    logger.warning(
                                        f"Failed to generate embedding for chunk {i}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error generating embedding for chunk {i}: {e}"
                                )
                                # Continue without this embedding
                else:
                    # No chunking: create single chunk with full content
                    # Map doc_type to appropriate chunk_type
                    chunk_type_mapping = {
                        "schema": "table_overview",
                        "sql_example": "example_overview",
                        "documentation": "doc_section",
                    }
                    chunk_type = chunk_type_mapping.get(doc_type, "doc_section")

                    chunk_data = ChunkData(
                        chunk_type=chunk_type, content=content, metadata=metadata
                    )

                    chunk_id = self._insert_chunk(cur, doc_id, 0, chunk_data)
                    if chunk_id:
                        chunks_created = 1
                        try:
                            embedding = self._safe_embed_query(content)
                            if embedding:
                                self._insert_embedding(
                                    cur, chunk_id, embedding, namespace
                                )
                                embeddings_created = 1
                            else:
                                logger.warning(
                                    "Failed to generate embedding for full document"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error generating embedding for document: {e}"
                            )
                            # Continue without embedding

                self.db_connection.commit()

                # Update statistics
                self.stats["documents_added"] += 1
                self.stats["chunks_created"] += chunks_created
                self.stats["embeddings_generated"] += embeddings_created

                logger.info(
                    f"✓ Added document: {chunks_created} chunks, {embeddings_created} embeddings"
                )

                return {
                    "success": True,
                    "document_id": doc_id,
                    "chunks_created": chunks_created,
                    "embeddings_created": embeddings_created,
                }

        except Exception as e:
            self.db_connection.rollback()
            error_msg = f"Failed to add document: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    # Legacy compatibility methods
    def add_sql_schema(self, schema_text: str, table_info: dict[str, Any] = None):
        """Add SQL schema to the knowledge base with table metadata."""
        metadata = table_info or {}

        # Ensure table_name is in metadata if available in table_info
        if table_info and "table_name" in table_info:
            metadata["table_name"] = table_info["table_name"]
            logger.debug(
                f"📋 Adding schema document with table_name: {table_info['table_name']}"
            )

        return self.add_document(
            content=schema_text, doc_type="schema", source="database", metadata=metadata
        )

    def add_sql_examples(self, examples: list[dict[str, str]]):
        """Add SQL question-answer examples with metadata support."""
        results = []
        for example in examples:
            # Extract tables from SQL if not provided in metadata
            tables = example.get("tables", [])
            if not tables and "sql" in example:
                tables = self._extract_tables_from_sql(example["sql"])

            # Build metadata including tables
            metadata = {"question": example["question"]}
            if tables:
                metadata["tables"] = tables

            # Add any additional metadata from example
            if "metadata" in example and isinstance(example["metadata"], dict):
                metadata.update(example["metadata"])

            question_sql_json = json.dumps(
                {
                    "question": example["question"],
                    "sql": example["sql"],
                    "tables": tables,  # Include tables in JSON content too
                },
                ensure_ascii=False,
            )

            result = self.add_document(
                content=question_sql_json,
                doc_type="sql_example",
                source="manual",
                metadata=metadata,
            )
            results.append(result)
        return results

    def _extract_tables_from_sql(self, sql: str) -> list[str]:
        """
        Extract table names from SQL query.

        Args:
            sql: SQL query string

        Returns:
            List of unique table names found in the query

        """
        import re

        if not sql:
            return []

        tables = set()

        # Normalize SQL - replace newlines with spaces for easier parsing
        sql_normalized = " ".join(sql.split())

        # Pattern to match table names in FROM and JOIN clauses
        patterns = [
            r"FROM\s+([a-zA-Z0-9_\.]+(?:\.[a-zA-Z0-9_]+)*)",
            r"(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+([a-zA-Z0-9_\.]+(?:\.[a-zA-Z0-9_]+)*)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql_normalized, re.IGNORECASE)
            for match in matches:
                # Clean the table name
                table = match.strip().replace("`", "").replace('"', "")

                # Handle fully qualified names
                if "." in table:
                    # Take the last part as the table name
                    table = table.split(".")[-1]

                # Add non-empty table names
                if table and table.upper() not in ["AS", "WHERE", "SET", "VALUES"]:
                    tables.add(table.lower())

        return list(tables)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.3,
        doc_types: list[str] | None = None,
        chunk_types: list[str] | None = None,
        table_filter: list[str] | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search with advanced filtering.

        Args:
            query: Search query
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            doc_types: Filter by document types
            chunk_types: Filter by chunk types
            table_filter: Filter by tables mentioned in metadata
            namespace: Embedding namespace to search

        Returns:
            List of search results with chunks and metadata

        """
        if chunk_types is None:
            chunk_types = ["example_overview", "example_ctes"]
        if doc_types is None:
            doc_types = ["sql_example", "documentation"]
        if not self.db_connection or not self.embeddings:
            return []

        if namespace is None:
            namespace = self.default_namespace

        try:
            # Generate query embedding
            query_embedding = self._safe_embed_query(query)
            if not query_embedding:
                return []  # Return empty results if embedding fails

            with self.db_connection.cursor() as cur:
                # Use optimized chunk_similarity_search function from pgvector schema
                if self.enable_chunking and self.enable_optimized_search:
                    return self._optimized_chunk_search(
                        cur,
                        query_embedding,
                        k,
                        similarity_threshold,
                        doc_types,
                        chunk_types,
                        table_filter,
                        namespace,
                    )
                else:
                    # Fallback to legacy query for non-chunked mode or disabled optimization
                    return self._legacy_similarity_search(
                        cur,
                        query_embedding,
                        k,
                        similarity_threshold,
                        doc_types,
                        chunk_types,
                        table_filter,
                        namespace,
                    )

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        keywords: str | None = None,
        k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.

        Args:
            query: Main search query
            keywords: Optional keyword query (defaults to query if not provided)
            k: Number of results to return
            vector_weight: Weight for vector similarity score
            keyword_weight: Weight for keyword score
            namespace: Embedding namespace to search

        Returns:
            List of search results with combined scores

        """
        if not self.db_connection or not self.embeddings:
            return []

        if namespace is None:
            namespace = self.default_namespace

        if keywords is None:
            keywords = query

        try:
            query_embedding = self._safe_embed_query(query)
            if not query_embedding:
                return []  # Return empty results if embedding fails

            with self.db_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM hybrid_chunk_search(
                        %(query_embedding)s::vector,
                        %(keywords)s,
                        %(vector_weight)s,
                        %(keyword_weight)s,
                        %(limit)s
                    )
                """,
                    {
                        "query_embedding": query_embedding,
                        "keywords": keywords,
                        "vector_weight": vector_weight,
                        "keyword_weight": keyword_weight,
                        "limit": k,
                    },
                )

                results = cur.fetchall()
                self.stats["queries_processed"] += 1

                logger.debug(f"Hybrid search returned {len(results)} results")
                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _optimized_chunk_search(
        self,
        cur,
        query_embedding,
        k,
        similarity_threshold,
        doc_types,
        chunk_types,
        table_filter,
        namespace,
    ) -> list[dict[str, Any]]:
        """Use optimized chunk_similarity_search function from pgvector schema."""
        try:
            # Convert doc_types to chunk_types filter if needed for compatibility
            if doc_types and not chunk_types:
                # Map document types to relevant chunk types
                chunk_type_mapping = {
                    "schema": ["table_overview", "columns", "constraints"],
                    "sql_example": ["example_overview", "example_ctes", "example_main"],
                    "documentation": ["doc_section"],
                }
                mapped_chunk_types = []
                for doc_type in doc_types:
                    mapped_chunk_types.extend(chunk_type_mapping.get(doc_type, []))
                chunk_types = (
                    list(set(mapped_chunk_types)) if mapped_chunk_types else None
                )

            # Call the optimized database function
            cur.execute(
                """
                SELECT * FROM chunk_similarity_search(
                    query_embedding := %s::vector,
                    match_threshold := %s,
                    match_count := %s,
                    filter_types := %s,
                    filter_tables := %s,
                    namespace_filter := %s
                )
                """,
                (
                    query_embedding,
                    similarity_threshold,
                    k,
                    chunk_types,
                    table_filter,
                    namespace,
                ),
            )

            results = cur.fetchall()
            self.stats["queries_processed"] += 1

            logger.debug(f"Optimized chunk search returned {len(results)} results")
            return [dict(row) for row in results]

        except Exception as e:
            logger.warning(f"Optimized search failed, falling back to legacy: {e}")
            return self._legacy_similarity_search(
                cur,
                query_embedding,
                k,
                similarity_threshold,
                None,
                chunk_types,
                table_filter,
                namespace,
            )

    def _legacy_similarity_search(
        self,
        cur,
        query_embedding,
        k,
        similarity_threshold,
        doc_types,
        chunk_types,
        table_filter,
        namespace,
    ) -> list[dict[str, Any]]:
        """Legacy similarity search using direct SQL queries."""
        try:
            # Build dynamic query with filters
            conditions = ["e.namespace = %(namespace)s"]
            params = {"namespace": namespace, "query_embedding": query_embedding}

            if doc_types:
                conditions.append("d.doc_type = ANY(%(doc_types)s)")
                params["doc_types"] = doc_types

            if chunk_types:
                conditions.append("c.chunk_type = ANY(%(chunk_types)s)")
                params["chunk_types"] = chunk_types

            if table_filter:
                conditions.append("c.metadata->'tables' ?| %(table_filter)s")
                params["table_filter"] = table_filter

            where_clause = " AND ".join(conditions)

            query_sql = f"""
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.chunk_type,
                    c.content,
                    c.token_count,
                    c.metadata as chunk_metadata,
                    d.doc_type,
                    d.source,
                    d.metadata as doc_metadata,
                    1 - (e.embedding <=> %(query_embedding)s::vector) as similarity
                FROM rag_chunks c
                JOIN rag_embeddings e ON c.id = e.chunk_id
                JOIN rag_documents d ON c.document_id = d.id
                WHERE {where_clause}
                    AND 1 - (e.embedding <=> %(query_embedding)s::vector) >= %(threshold)s
                ORDER BY e.embedding <=> %(query_embedding)s::vector
                LIMIT %(limit)s
            """

            params.update({"threshold": similarity_threshold, "limit": k})

            cur.execute(query_sql, params)
            results = cur.fetchall()

            logger.debug(f"Legacy search returned {len(results)} results")
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Legacy search failed: {e}")
            return []

    def get_document_chunks(self, document_id: int) -> list[dict[str, Any]]:
        """Get all chunks for a specific document."""
        if not self.db_connection:
            return []

        try:
            with self.db_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        c.*,
                        CASE WHEN e.id IS NOT NULL THEN true ELSE false END as has_embedding
                    FROM rag_chunks c
                    LEFT JOIN rag_embeddings e ON c.id = e.chunk_id
                    WHERE c.document_id = %(doc_id)s
                    ORDER BY c.chunk_index
                """,
                    {"doc_id": document_id},
                )

                return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []

    def reindex_document(
        self, document_id: int, namespace: str | None = None
    ) -> dict[str, Any]:
        """
        Reindex a document by regenerating its chunks and embeddings.

        Args:
            document_id: ID of document to reindex
            namespace: Embedding namespace

        Returns:
            Dictionary with reindexing results

        """
        if not self.db_connection or not self.embeddings:
            return {"error": "Database or embeddings not initialized"}

        if namespace is None:
            namespace = self.default_namespace

        try:
            with self.db_connection.cursor() as cur:
                # Get document
                cur.execute(
                    "SELECT * FROM rag_documents WHERE id = %(id)s", {"id": document_id}
                )
                doc = cur.fetchone()

                if not doc:
                    return {"error": f"Document {document_id} not found"}

                # Delete existing chunks and embeddings
                cur.execute(
                    "DELETE FROM rag_chunks WHERE document_id = %(id)s",
                    {"id": document_id},
                )

                # Recreate chunks
                chunks = self._create_chunks(
                    doc["full_content"], doc["doc_type"], doc.get("metadata", {})
                )

                chunks_created = 0
                embeddings_created = 0

                for i, chunk in enumerate(chunks):
                    chunk_id = self._insert_chunk(cur, document_id, i, chunk)
                    if chunk_id:
                        chunks_created += 1

                        # Generate and store embedding
                        embedding = self.embeddings.embed_query(chunk.content)
                        self._insert_embedding(cur, chunk_id, embedding, namespace)
                        embeddings_created += 1

                self.db_connection.commit()

                logger.info(
                    f"✓ Reindexed document {document_id}: {chunks_created} chunks, {embeddings_created} embeddings"
                )

                return {
                    "success": True,
                    "chunks_created": chunks_created,
                    "embeddings_created": embeddings_created,
                }

        except Exception as e:
            self.db_connection.rollback()
            error_msg = f"Failed to reindex document {document_id}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the knowledge base."""
        if not self.db_connection:
            return {"error": "Database not connected"}

        try:
            with self.db_connection.cursor() as cur:
                # Get chunk statistics using the database function
                cur.execute("SELECT * FROM get_chunk_statistics()")
                chunk_stats = [dict(row) for row in cur.fetchall()]

                # Get total counts
                cur.execute(
                    """
                    SELECT 
                        COUNT(DISTINCT d.id) as total_documents,
                        COUNT(DISTINCT c.id) as total_chunks,
                        COUNT(DISTINCT e.id) as total_embeddings,
                        COUNT(DISTINCT e.namespace) as total_namespaces
                    FROM rag_documents d
                    LEFT JOIN rag_chunks c ON d.id = c.document_id
                    LEFT JOIN rag_embeddings e ON c.id = e.chunk_id
                """
                )

                totals = dict(cur.fetchone())

                # Get namespace distribution
                cur.execute(
                    """
                    SELECT namespace, COUNT(*) as count
                    FROM rag_embeddings
                    GROUP BY namespace
                    ORDER BY count DESC
                """
                )

                namespace_stats = [dict(row) for row in cur.fetchall()]

                return {
                    "totals": totals,
                    "chunk_statistics": chunk_stats,
                    "namespace_distribution": namespace_stats,
                    "runtime_stats": self.stats,
                    "chunking_enabled": self.enable_chunking,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    def _create_chunks(
        self, content: str, doc_type: str, metadata: dict[str, Any]
    ) -> list[ChunkData]:
        """Create chunks for a document using appropriate strategy."""
        if not self.enable_chunking:
            # Map doc_type to appropriate chunk_type when chunking is disabled
            chunk_type_mapping = {
                "schema": "table_overview",
                "sql_example": "example_overview",
                "documentation": "doc_section",
            }
            chunk_type = chunk_type_mapping.get(doc_type, "doc_section")

            return [
                ChunkData(chunk_type=chunk_type, content=content, metadata=metadata)
            ]

        # Get chunking strategy configuration
        strategy_config = self.chunk_strategies.get(doc_type, {})

        try:
            chunks = chunk_document(doc_type, content, metadata, **strategy_config)
            logger.debug(f"Created {len(chunks)} chunks for {doc_type} document")
            return chunks

        except Exception as e:
            logger.warning(f"Chunking failed for {doc_type}: {e}, using fallback")
            return [
                ChunkData(
                    chunk_type="doc_section",  # Use valid chunk type for fallback
                    content=content[:2000],  # Truncate if too long
                    metadata=metadata,
                )
            ]

    def _insert_chunk(
        self, cur, doc_id: int, chunk_index: int, chunk: ChunkData
    ) -> int | None:
        """Insert a chunk into the database."""
        try:
            cur.execute(
                """
                INSERT INTO rag_chunks 
                (document_id, chunk_index, chunk_type, content, token_count, metadata, char_start, char_end)
                VALUES (%(doc_id)s, %(chunk_index)s, %(chunk_type)s, %(content)s, %(token_count)s, %(metadata)s, %(char_start)s, %(char_end)s)
                RETURNING id
            """,
                {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.content,
                    "token_count": chunk.estimate_tokens(),
                    "metadata": json.dumps(chunk.metadata),
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            )

            result = cur.fetchone()
            return result["id"] if result else None

        except Exception as e:
            logger.error(f"Failed to insert chunk: {e}")
            return None

    def _insert_embedding(
        self, cur, chunk_id: int, embedding: list[float], namespace: str
    ):
        """Insert an embedding into the database."""
        try:
            cur.execute(
                """
                INSERT INTO rag_embeddings (chunk_id, namespace, embedding, metadata)
                VALUES (%(chunk_id)s, %(namespace)s, %(embedding)s, %(metadata)s)
            """,
                {
                    "chunk_id": chunk_id,
                    "namespace": namespace,
                    "embedding": embedding,
                    "metadata": json.dumps({"created_at": time.time()}),
                },
            )

        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
            logger.info("Database connection closed")
