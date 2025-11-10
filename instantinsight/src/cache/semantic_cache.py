"""
Redis-based Semantic Cache Implementation.

This module provides semantic caching functionality using Redis Stack
with vector similarity search to reduce SQL generation latency.

Key Features:
- Redis Stack with native vector search
- Sub-millisecond latency for cache hits
- User isolation for MCP multi-tenancy
- Automatic TTL-based cache expiry
- Lightweight and scalable
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import redis.asyncio as redis
import xxhash
from langchain_aws import BedrockEmbeddings
from loguru import logger
from pydantic import BaseModel

from ..config.cache_config import CACHE_CONFIG
from ..config.database_config import BEDROCK_CONFIG


class CacheResult(BaseModel):
    """Result from semantic cache lookup."""

    sql: str
    confidence: float
    source: str = "semantic_cache"
    execution_time_ms: int | None = None
    data_row_count: int | None = None
    cache_entry_id: str | None = None
    visualization: dict[str, Any] | None = None


class CacheEntry(BaseModel):
    """Semantic cache entry structure."""

    id: str | None = None
    query_text: str
    query_embedding: list[float]
    generated_sql: str
    result_hash: str | None = None
    confidence_score: float = 1.0
    execution_time_ms: int | None = None
    data_row_count: int | None = None
    created_at: datetime | None = None
    cache_hits: int = 0
    user_id: str | None = None


class SemanticCache:
    """
    Redis-based semantic cache for SQL generation.

    This cache stores SQL generation results with their semantic embeddings
    in Redis, providing fast retrieval with native vector similarity search.
    """

    def __init__(
        self,
        connection_string: str | None = None,  # Keep for backward compatibility
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        enable_result_validation: bool = True,
    ):
        """
        Initialize Redis semantic cache.

        Args:
            connection_string: Ignored - kept for backward compatibility
            similarity_threshold: Minimum cosine similarity for cache hits (0.0-1.0)
            max_cache_size: Maximum number of cache entries
            enable_result_validation: Whether to validate cached results

        """
        # Load configuration
        config = CACHE_CONFIG
        redis_config = config["redis"]
        cache_settings = config["cache_settings"]

        # Override defaults with parameters
        self.similarity_threshold = (
            similarity_threshold or cache_settings["similarity_threshold"]
        )
        self.max_cache_size = max_cache_size or cache_settings["max_cache_size"]
        self.enable_result_validation = enable_result_validation
        self.ttl_seconds = cache_settings["ttl_hours"] * 3600
        self.max_entries_per_user = cache_settings["max_entries_per_user"]

        # MCP settings
        self.enable_user_isolation = config["mcp_settings"]["enable_user_isolation"]
        self.default_user_id = config["mcp_settings"]["default_user_id"]

        # Initialize Redis connection
        self.redis_client = None  # Will be initialized asynchronously
        self.redis_config = redis_config
        self.index_name = "idx:semantic_cache"
        self.index_created = False

        # Initialize Titan embeddings using Bedrock
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=BEDROCK_CONFIG["aws_region"],
            credentials_profile_name=BEDROCK_CONFIG["aws_profile"],
        )

        logger.info(
            f"🚀 Redis semantic cache initialized (threshold={self.similarity_threshold}, "
            f"max_size={self.max_cache_size}, ttl={cache_settings['ttl_hours']}h)"
        )

    async def _ensure_redis_connection(self):
        """Ensure Redis connection is established."""
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_config["host"],
                    port=self.redis_config["port"],
                    db=self.redis_config["db"],
                    password=self.redis_config["password"],
                    decode_responses=False,
                    max_connections=self.redis_config["max_connections"],
                    socket_keepalive=self.redis_config["socket_keepalive"],
                    health_check_interval=self.redis_config["health_check_interval"],
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Redis connection established")

                # Create index if needed
                await self._create_vector_index()

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
                raise

    async def _create_vector_index(self):
        """Create vector search index if it doesn't exist."""
        if self.index_created:
            return

        try:
            # Check if index exists
            try:
                await self.redis_client.ft(self.index_name).info()
                self.index_created = True
                logger.info(f"📊 Vector index '{self.index_name}' already exists")
                return
            except Exception:
                pass  # Index doesn't exist, create it

            # Create index using Redis Stack commands
            from redis.commands.search.field import (
                NumericField,
                TextField,
                VectorField,
            )
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            # Define schema
            schema = [
                TextField("user_id"),
                TextField("query_text"),
                TextField("sql"),
                NumericField("timestamp"),
                NumericField("hits"),
                VectorField(
                    "embedding",
                    "FLAT",  # Use HNSW for >1M vectors
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 1024,  # Titan embedding dimension
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ]

            # Create index
            definition = IndexDefinition(
                prefix=["cache:"],
                index_type=IndexType.HASH,
            )

            await self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )

            self.index_created = True
            logger.info(f"✨ Created vector index '{self.index_name}'")

        except Exception as e:
            logger.warning(f"Could not create vector index (may not be needed): {e}")
            # Continue without index - will use simpler key-based lookup

    def _generate_cache_key(
        self, user_id: str, query: str, prior_turns: list[dict] | None = None
    ) -> str:
        """
        Generate cache key including conversation context.

        Uses last 3 turns to balance cache hit rate with context awareness.
        """
        if prior_turns and len(prior_turns) > 0:
            # Use last 3 turns for context (enough for most scenarios)
            recent_context = prior_turns[-3:]
            context_parts = []

            for turn in recent_context:
                content = turn.get("content", "")
                sql = turn.get("sql", "")[:50]  # First 50 chars of SQL
                context_parts.append(f"{content}:{sql}")

            context_str = "|".join(context_parts)
            full_key = f"{query}|CTX:{context_str}"
        else:
            full_key = query

        query_hash = xxhash.xxh64(full_key.encode()).hexdigest()

        if self.enable_user_isolation and user_id:
            return f"cache:{user_id}:{query_hash}"
        return f"cache:global:{query_hash}"

    async def get_cached_result(
        self,
        query: str,
        user_id: str | None = None,
        prior_turns: list[dict] | None = None,
    ) -> CacheResult | None:
        """
        Look up cached result with conversation context.

        Args:
            query: Natural language query
            user_id: Optional user ID for MCP isolation
            prior_turns: Optional list of prior conversation turns

        Returns:
            CacheResult if similar query found, None otherwise

        """
        try:
            await self._ensure_redis_connection()

            if not self.redis_client:
                return None

            # Use default user if not provided
            user_id = user_id or self.default_user_id

            # Generate context-aware cache key
            cache_key = self._generate_cache_key(user_id, query, prior_turns)
            exact_match = await self.redis_client.hgetall(cache_key)

            if exact_match and b"sql" in exact_match:
                # Update hit counter
                await self.redis_client.hincrby(cache_key, "hits", 1)

                logger.info(f"🎯 Cache hit (exact match) for query: {query[:50]}...")

                # Parse visualization config if present
                visualization = None
                if b"visualization" in exact_match:
                    try:
                        viz_json = exact_match[b"visualization"].decode("utf-8")
                        visualization = json.loads(viz_json) if viz_json else None
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.warning("Failed to parse cached visualization config")
                        visualization = None

                return CacheResult(
                    sql=exact_match[b"sql"].decode("utf-8"),
                    confidence=1.0,
                    source="semantic_cache",
                    execution_time_ms=int(exact_match.get(b"exec_time", 0))
                    if exact_match.get(b"exec_time")
                    else None,
                    data_row_count=int(exact_match.get(b"row_count", 0))
                    if exact_match.get(b"row_count")
                    else None,
                    cache_entry_id=cache_key,
                    visualization=visualization,
                )

            # If no exact match and index exists, try vector similarity
            if self.index_created:
                # Generate embedding for incoming query
                query_embedding = await self._generate_embedding(query)

                if not query_embedding:
                    logger.warning("Failed to generate query embedding")
                    return None

                # Perform vector similarity search
                from redis.commands.search.query import Query

                # Convert embedding to bytes
                query_vec = np.array(query_embedding, dtype=np.float32).tobytes()

                # Build query with user filter if isolation is enabled
                if self.enable_user_isolation and user_id != "global":
                    search_query = (
                        Query(
                            f"(@user_id:{{{user_id}}})=>[KNN 3 @embedding $vec AS score]"
                        )
                        .return_fields(
                            "sql", "score", "exec_time", "row_count", "visualization"
                        )
                        .dialect(2)
                    )
                else:
                    search_query = (
                        Query("*=>[KNN 3 @embedding $vec AS score]")
                        .return_fields(
                            "sql", "score", "exec_time", "row_count", "visualization"
                        )
                        .dialect(2)
                    )

                results = await self.redis_client.ft(self.index_name).search(
                    search_query, query_params={"vec": query_vec}
                )

                if results.docs:
                    best_match = results.docs[0]
                    similarity = 1 - float(
                        best_match.score
                    )  # Convert distance to similarity

                    if similarity >= self.similarity_threshold:
                        # Update hit counter
                        await self.redis_client.hincrby(best_match.id, "hits", 1)

                        logger.info(
                            f"🎯 Cache hit (similarity: {similarity:.3f}) for query: {query[:50]}..."
                        )

                        # Parse visualization config if present
                        visualization = None
                        if (
                            hasattr(best_match, "visualization")
                            and best_match.visualization
                        ):
                            try:
                                visualization = json.loads(best_match.visualization)
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    "Failed to parse cached visualization config from vector search"
                                )
                                visualization = None

                        return CacheResult(
                            sql=best_match.sql,
                            confidence=similarity,
                            source="semantic_cache",
                            execution_time_ms=int(best_match.exec_time)
                            if hasattr(best_match, "exec_time")
                            else None,
                            data_row_count=int(best_match.row_count)
                            if hasattr(best_match, "row_count")
                            else None,
                            cache_entry_id=best_match.id,
                            visualization=visualization,
                        )

            logger.debug(f"💔 Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error in cache lookup: {e}")
            return None

    async def store_result(
        self,
        query: str,
        sql: str,
        result_data: pd.DataFrame | None = None,
        execution_time_ms: int | None = None,
        user_id: str | None = None,
        visualization: dict[str, Any] | None = None,
        prior_turns: list[dict] | None = None,
    ) -> bool:
        """
        Store result with context-aware cache key.

        Args:
            query: Original natural language query
            sql: Generated SQL
            result_data: Query execution result data
            execution_time_ms: SQL generation time in milliseconds
            user_id: Optional user ID for MCP isolation
            visualization: Optional visualization config (Plotly chart schema)
            prior_turns: Optional list of prior conversation turns

        Returns:
            True if stored successfully, False otherwise

        """
        try:
            # Skip caching for historical context queries
            if "Historical context" in query or "Current SQL query:" in query:
                logger.info("Skipping cache storage for historical context query")
                return True  # Return True to indicate no error, just skipped
            # Check if event loop is closing
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    logger.debug("Event loop is closed, skipping cache storage")
                    return False
            except RuntimeError:
                # No running loop, that's okay
                pass

            await self._ensure_redis_connection()

            if not self.redis_client:
                return False

            # Use default user if not provided
            user_id = user_id or self.default_user_id

            # Generate context-aware cache key
            cache_key = self._generate_cache_key(user_id, query, prior_turns)

            # Prepare data to store
            cache_data = {
                "user_id": user_id,
                "query_text": query,
                "sql": sql,
                "timestamp": int(time.time()),
                "hits": 0,
            }

            if execution_time_ms is not None:
                cache_data["exec_time"] = str(execution_time_ms)

            if result_data is not None and not result_data.empty:
                cache_data["row_count"] = str(len(result_data))
                # Optionally compute result hash
                if self.enable_result_validation:
                    cache_data["result_hash"] = self._compute_result_hash(result_data)

            # Store visualization config if provided
            if visualization is not None:
                try:
                    cache_data["visualization"] = json.dumps(visualization)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize visualization config: {e}")
                    # Continue without visualization

            # Store embedding if vector index is available
            if self.index_created:
                query_embedding = await self._generate_embedding(query)
                if query_embedding:
                    # Store embedding as binary
                    cache_data["embedding"] = np.array(
                        query_embedding, dtype=np.float32
                    ).tobytes()

            # Store in Redis with TTL - wrap in try-catch for connection issues
            try:
                await self.redis_client.hset(cache_key, mapping=cache_data)
                await self.redis_client.expire(cache_key, self.ttl_seconds)
            except Exception as redis_error:
                # Handle specific event loop closure gracefully
                if "Event loop is closed" in str(redis_error):
                    logger.debug("Skipping cache storage - event loop is closing")
                    return True  # Don't treat this as an error
                else:
                    logger.debug(f"Redis operation failed: {redis_error}")
                    return False

            # Enforce per-user cache limit
            if self.enable_user_isolation and user_id != "global":
                try:
                    user_pattern = f"cache:{user_id}:*"
                    user_keys = []
                    async for key in self.redis_client.scan_iter(match=user_pattern):
                        user_keys.append(key)

                    if len(user_keys) > self.max_entries_per_user:
                        # Remove oldest entries (simple FIFO for now)
                        excess = len(user_keys) - self.max_entries_per_user
                        keys_to_delete = user_keys[:excess]
                        if keys_to_delete:
                            await self.redis_client.delete(*keys_to_delete)
                            logger.debug(
                                f"Removed {excess} old cache entries for user {user_id}"
                            )
                except Exception as cleanup_error:
                    logger.debug(f"Cache cleanup failed: {cleanup_error}")

            logger.debug(
                f"💾 Stored cache entry {cache_key} for query: {query[:50]}..."
            )
            return True

        except Exception as e:
            logger.error(f"Error storing cache result: {e}")
            return False

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using Titan."""
        try:
            # BedrockEmbeddings is synchronous, so we run it in an executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embeddings.embed_query, text
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _compute_result_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of query result data."""
        if data is None or data.empty:
            return ""

        try:
            # Create a stable hash by sorting columns and converting to string
            sorted_data = data.reindex(sorted(data.columns), axis=1)
            data_string = sorted_data.to_string(index=False)
            return hashlib.sha256(data_string.encode()).hexdigest()[:16]  # Shorter hash
        except Exception as e:
            logger.warning(f"Error computing result hash: {e}")
            return ""

    async def get_cache_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_redis_connection()

            if not self.redis_client:
                return {}

            stats = {
                "total_entries": 0,
                "user_entries": 0,
                "total_hits": 0,
                "cache_size_bytes": 0,
            }

            # Count entries
            if user_id and self.enable_user_isolation:
                pattern = f"cache:{user_id}:*"
            else:
                pattern = "cache:*"

            total_hits = 0
            entry_count = 0

            async for key in self.redis_client.scan_iter(match=pattern):
                entry_count += 1
                hits = await self.redis_client.hget(key, "hits")
                if hits:
                    total_hits += int(hits)

            stats["total_entries"] = entry_count
            stats["total_hits"] = total_hits

            # Get Redis memory info
            info = await self.redis_client.info("memory")
            stats["cache_size_bytes"] = info.get("used_memory", 0)

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    async def clear_cache(self, user_id: str | None = None) -> bool:
        """Clear cache entries."""
        try:
            await self._ensure_redis_connection()

            if not self.redis_client:
                return False

            # Determine pattern to clear
            if user_id and self.enable_user_isolation:
                pattern = f"cache:{user_id}:*"
                logger.info(f"🧹 Clearing cache for user: {user_id}")
            else:
                pattern = "cache:*"
                logger.info("🧹 Clearing all cache entries")

            # Delete matching keys
            deleted = 0
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
                deleted += 1

            logger.info(f"✅ Cleared {deleted} cache entries")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    # Sync wrapper methods for backward compatibility
    def get_cached_result_sync(
        self,
        query: str,
        user_id: str | None = None,
        prior_turns: list[dict] | None = None,
    ) -> CacheResult | None:
        """Get cached result synchronously."""
        try:
            # Use direct synchronous Redis client to avoid event loop issues
            return self._get_cached_result_direct_sync(query, user_id, prior_turns)
        except Exception as e:
            logger.error(f"Error in sync cache lookup: {e}")
            return None

    def _get_cached_result_direct_sync(
        self,
        query: str,
        user_id: str | None = None,
        prior_turns: list[dict] | None = None,
    ) -> CacheResult | None:
        """Direct synchronous cache lookup using sync Redis client."""
        try:
            import redis

            # Create a synchronous Redis client
            sync_redis = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                db=self.redis_config["db"],
                password=self.redis_config["password"],
                decode_responses=False,
            )

            # Test connection
            sync_redis.ping()

            # Use default user if not provided
            user_id = user_id or self.default_user_id

            # Generate context-aware cache key
            cache_key = self._generate_cache_key(user_id, query, prior_turns)
            exact_match = sync_redis.hgetall(cache_key)

            if exact_match and b"sql" in exact_match:
                # Update hit counter
                sync_redis.hincrby(cache_key, "hits", 1)

                logger.info(f"🎯 Cache hit (exact match) for query: {query[:50]}...")

                # Parse visualization config if present
                visualization = None
                if b"visualization" in exact_match:
                    try:
                        viz_json = exact_match[b"visualization"].decode("utf-8")
                        visualization = json.loads(viz_json) if viz_json else None
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.warning("Failed to parse cached visualization config")
                        visualization = None

                return CacheResult(
                    sql=exact_match[b"sql"].decode("utf-8"),
                    confidence=1.0,
                    source="semantic_cache",
                    execution_time_ms=int(exact_match.get(b"exec_time", 0))
                    if exact_match.get(b"exec_time")
                    else None,
                    data_row_count=int(exact_match.get(b"row_count", 0))
                    if exact_match.get(b"row_count")
                    else None,
                    cache_entry_id=cache_key,
                    visualization=visualization,
                )

            # No exact match found - we skip vector similarity for sync version
            logger.debug(f"💔 Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.debug(f"Direct sync cache lookup failed: {e}")
            return None

    def store_result_sync(
        self,
        query: str,
        sql: str,
        result_data: pd.DataFrame | None = None,
        execution_time_ms: int | None = None,
        visualization: dict[str, Any] | None = None,
        prior_turns: list[dict] | None = None,
    ) -> bool:
        """Store result synchronously."""
        try:
            # Skip caching for historical context queries
            if "Historical context" in query or "Current SQL query:" in query:
                logger.info("Skipping cache storage for historical context query")
                return True  # Return True to indicate no error, just skipped

            # Always use the direct synchronous method to avoid event loop issues
            # This ensures the cache is stored immediately and reliably
            # return self._store_result_direct_sync(
            #     query, sql, result_data, execution_time_ms, visualization
            # )

        except Exception as e:
            logger.error(f"Error in sync cache storage: {e}")
            return False

    def _store_result_direct_sync(
        self,
        query: str,
        sql: str,
        result_data: pd.DataFrame | None = None,
        execution_time_ms: int | None = None,
        visualization: dict[str, Any] | None = None,
        prior_turns: list[dict] | None = None,
    ) -> bool:
        """Direct synchronous cache storage using sync Redis client."""
        try:
            import redis

            # Create a synchronous Redis client
            sync_redis = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                db=self.redis_config["db"],
                password=self.redis_config["password"],
                decode_responses=False,
            )

            # Test connection
            sync_redis.ping()

            # Use default user
            user_id = self.default_user_id

            # Generate context-aware cache key
            cache_key = self._generate_cache_key(user_id, query, prior_turns)

            # Prepare data to store
            cache_data = {
                "user_id": user_id,
                "query_text": query,
                "sql": sql,
                "timestamp": str(int(time.time())),
                "hits": "0",
            }

            if execution_time_ms is not None:
                cache_data["exec_time"] = str(execution_time_ms)

            if result_data is not None and not result_data.empty:
                cache_data["row_count"] = str(len(result_data))

            # Store visualization config if provided
            if visualization is not None:
                try:
                    cache_data["visualization"] = json.dumps(visualization)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize visualization config: {e}")

            # Store in Redis with TTL
            sync_redis.hset(cache_key, mapping=cache_data)
            sync_redis.expire(cache_key, self.ttl_seconds)

            logger.debug(
                f"💾 Stored cache entry {cache_key} (sync) for query: {query[:50]}..."
            )
            return True

        except Exception as e:
            logger.debug(f"Direct sync cache storage failed: {e}")
            return False

    # Compatibility methods (no-ops for Redis)
    async def invalidate_by_table(self, table_name: str) -> int:
        """Not needed for Redis - TTL handles invalidation."""
        logger.debug(
            f"Table invalidation not needed for Redis cache (table: {table_name})"
        )
        return 0

    async def invalidate_by_ttl(self, hours: int = 24) -> int:
        """Not needed for Redis - TTL is automatic."""
        logger.debug("TTL invalidation not needed for Redis cache (TTL set on storage)")
        return 0

    def _get_connection(self):
        """Compatibility method - not used in Redis implementation."""
        pass

    def _extract_table_dependencies(self, sql: str) -> dict[str, Any]:
        """Compatibility method - not used in Redis implementation."""
        return {}
