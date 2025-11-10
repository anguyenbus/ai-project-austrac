"""
PgvectorRAG: PostgreSQL pgvector-based RAG system for text-to-SQL generation.

Simple RAG implementation using pgvector for semantic search and retrieval.
"""

import time
from typing import Any

from langchain_aws import BedrockEmbeddings
from loguru import logger
from psycopg.rows import dict_row

from src.rag.backends import PgVectorBackend, PgVectorBackendConfig


class PgvectorRAG:
    """
    PostgreSQL pgvector-based RAG system for text-to-SQL generation.

    Features:
    - Vector similarity search using pgvector
    - AWS Bedrock embeddings integration
    - Namespace support for different embedding strategies
    - Designed for text-to-SQL use cases
    """

    def __init__(
        self,
        connection_string: str,
        bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
        default_namespace: str = "semantic",
        backend: PgVectorBackend | None = None,
        pool_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the pgvector RAG system.

        Args:
            connection_string: PostgreSQL connection string
            bedrock_embedding_model: AWS Bedrock embedding model ID
            dimension: Vector dimension (1024 for Titan v2)
            default_namespace: Default namespace for embeddings
            backend: Optional custom PgVectorBackend instance
            pool_config: Optional database connection pool configuration
            backend_config: Optional backend-specific configuration

        """
        self.connection_string = connection_string
        self.bedrock_embedding_model = bedrock_embedding_model
        self.dimension = dimension
        self.default_namespace = default_namespace

        # Backend / connection pooling
        self.backend: PgVectorBackend | None = backend
        self.backend_pool_config = pool_config or {}
        self.backend_config = backend_config or {}

        # Embeddings model
        self.embeddings = None

        logger.info("🔧 Initializing PgvectorRAG")

    def connect_to_database(self) -> bool:
        """Initialise the PgVector backend and connection pool."""
        try:
            if not self.backend:
                config_kwargs: dict[str, Any] = {
                    "db_url": self.backend_config.get("db_url", self.connection_string),
                    "table_name": self.backend_config.get(
                        "table_name", "instantinsight_pgvector_internal"
                    ),
                    "schema": self.backend_config.get("schema", "ai"),
                    "pool_size": self.backend_pool_config.get("pool_size", 5),
                    "max_overflow": self.backend_pool_config.get("max_overflow", 10),
                }

                if "search_type" in self.backend_config:
                    config_kwargs["search_type"] = self.backend_config["search_type"]
                if "vector_index" in self.backend_config:
                    config_kwargs["vector_index"] = self.backend_config["vector_index"]
                if "distance" in self.backend_config:
                    config_kwargs["distance"] = self.backend_config["distance"]
                if "prefix_match" in self.backend_config:
                    config_kwargs["prefix_match"] = self.backend_config["prefix_match"]
                if "vector_score_weight" in self.backend_config:
                    config_kwargs["vector_score_weight"] = self.backend_config[
                        "vector_score_weight"
                    ]

            if not self.backend:
                self.backend = PgVectorBackend(PgVectorBackendConfig(**config_kwargs))

            if not self.backend.ensure_initialized():
                logger.error("Failed to initialise PgVector backend")
                return False

            if not self.backend.ensure_initialized():
                return False

            logger.info("✓ Connected to PostgreSQL via PgVector backend")
            return True

        except Exception as error:  # pragma: no cover - defensive logging
            logger.error(f"Failed to set up PgVector backend: {error}")
            return False

    def is_connected(self) -> bool:
        """Check if backend is connected and initialized."""
        return bool(self.backend and self.backend.is_initialized)

    def cursor(self, row_factory=dict_row):
        """Get database cursor with specified row factory."""
        if not self.backend:
            raise RuntimeError("PgVector backend is not initialised")
        return self.backend.cursor(row_factory=row_factory)

    def session(self):
        """Get database session context manager."""
        if not self.backend:
            raise RuntimeError("PgVector backend is not initialised")
        return self.backend.session()

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
        if not self.is_connected() or not self.embeddings:
            return []

        if namespace is None:
            namespace = self.default_namespace

        try:
            # Generate query embedding
            query_embedding = self._safe_embed_query(query)
            if not query_embedding:
                return []  # Return empty results if embedding fails

            with self.cursor(row_factory=dict_row) as cur:
                # Use standard similarity search
                return self._standard_similarity_search(
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

    def _standard_similarity_search(
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
            try:
                cur.connection.rollback()
            except Exception as rollback_error:
                logger.debug(
                    f"Rollback after legacy search failure failed: {rollback_error}"
                )
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the knowledge base."""
        if not self.is_connected():
            return {"error": "Database not connected"}

        try:
            with self.cursor(row_factory=dict_row) as cur:
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
                    "chunking_enabled": False,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    def find_relevant_examples(self, question: str, **kwargs) -> dict[str, Any]:
        """
        Find relevant schema and example context for SQL generation.

        This method searches the knowledge base for relevant schema information
        and example queries, then delegates actual SQL generation to SQLWriterAgent.

        Args:
            question: Natural language question
            **kwargs: Additional parameters (selected_tables, k, etc.)

        Returns:
            Dictionary containing SQL and metadata

        """
        if not self.is_connected() or not self.embeddings:
            return {"error": "Database or embeddings not initialized"}

        try:
            # Extract relevant content for SQL generation context
            schema_context = kwargs.get("schema_context", [])
            # Get selected tables from kwargs (passed from TableAgent)
            selected_tables = kwargs.get("selected_tables", [])
            # Get filter context from FilteringAgent/ColumnAgent
            filter_context = kwargs.get("filter_context", None)
            example_context = []
            best_similarity = 0.0

            # Use similarity search to find relevant examples chunk
            search_results = self.similarity_search(
                query=question,
                k=kwargs.get("k", 5),
                similarity_threshold=kwargs.get("similarity_threshold", 0.2),
                doc_types=["sql_example", "documentation"],
                chunk_types=["example_overview", "example_ctes"],
                table_filter=selected_tables,
            )

            if not search_results:
                return {"error": "No relevant information found"}

            for result in search_results:
                chunk_type = result.get("chunk_type", "")
                content = result.get("content", "")
                similarity = result.get("similarity", 0.0)

                if similarity > best_similarity:
                    best_similarity = similarity

                if chunk_type in ["example_overview", "example_ctes"]:
                    example_context.append(content)

            # Return context and search results for SQL generation
            if schema_context or example_context:
                return {
                    "schema_context": schema_context,
                    "example_context": example_context,
                    "selected_tables": selected_tables,
                    "search_results": search_results,
                    "best_similarity": best_similarity,
                    "filter_context": filter_context,
                    "context_used": "\n".join(schema_context) if schema_context else "",
                    "sources": search_results[:5],
                    "metadata": {
                        "best_similarity": best_similarity,
                        "sources_count": len(search_results),
                    },
                }
            else:
                # Fallback - no context found
                return {
                    "schema_context": [],
                    "example_context": [],
                    "selected_tables": [],
                    "search_results": [],
                    "best_similarity": 0.0,
                    "filter_context": filter_context,
                    "context_used": "",
                    "sources": [],
                    "metadata": {
                        "best_similarity": 0.0,
                        "sources_count": 0,
                    },
                    "sql": f"-- No relevant context found for: {question}\nSELECT * FROM information_schema.tables LIMIT 10;",
                    "confidence": 0.1,
                    "error": "No relevant schema or examples found",
                }

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {"error": f"SQL generation failed: {str(e)}"}

    def close(self):
        """Close database connection."""
        if self.backend:
            self.backend.close_all()
            self.backend = None
            logger.info("PgVector backend closed")
