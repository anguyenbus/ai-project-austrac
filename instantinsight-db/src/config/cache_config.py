"""
Cache Configuration for Semantic Cache.

This module provides configuration for the semantic cache system,
supporting both Redis and PostgreSQL backends.
"""

import os
from typing import Any


def get_cache_config() -> dict[str, Any]:
    """
    Get cache configuration based on environment variables.

    Returns:
        Dictionary with cache configuration

    """
    return {
        "provider": os.getenv("CACHE_PROVIDER", "redis"),  # Default to Redis
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": int(os.getenv("REDIS_DB", 0)),
            "password": os.getenv("REDIS_PASSWORD", None),
            "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", 1000)),
            "socket_keepalive": True,
            "socket_keepalive_options": {
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            },
            "health_check_interval": 30,
            "decode_responses": False,  # Keep as bytes for embeddings
            "connection_pool_kwargs": {
                "connection_class": "redis.asyncio.Connection",
                "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", 1000)),
            },
        },
        "cache_settings": {
            "ttl_hours": int(os.getenv("CACHE_TTL_HOURS", 24)),
            "similarity_threshold": float(
                os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.92)
            ),
            "max_entries_per_user": int(os.getenv("CACHE_MAX_ENTRIES_PER_USER", 1000)),
            "max_cache_size": int(os.getenv("CACHE_MAX_SIZE", 10000)),
            "enable_metrics": os.getenv("CACHE_ENABLE_METRICS", "true").lower()
            == "true",
            "index_algorithm": os.getenv(
                "CACHE_INDEX_ALGORITHM", "FLAT"
            ),  # FLAT or HNSW
        },
        "mcp_settings": {
            "enable_user_isolation": os.getenv("MCP_USER_ISOLATION", "true").lower()
            == "true",
            "default_user_id": os.getenv("MCP_DEFAULT_USER", "default"),
        },
    }


# Export config as module-level constant
CACHE_CONFIG = get_cache_config()
