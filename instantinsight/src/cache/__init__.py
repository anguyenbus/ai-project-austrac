"""
Semantic caching module for instantinsight RAG system.

This module provides semantic caching functionality using:
- AWS Bedrock Titan embeddings for semantic similarity
- PostgreSQL pgvector for efficient vector storage and search
- Intelligent cache invalidation strategies
"""

from .semantic_cache import CacheEntry, CacheResult, SemanticCache

__all__ = ["SemanticCache", "CacheResult", "CacheEntry"]
