"""
Vector database configuration classes.

This module provides configuration types for vector database operations,
including distance metrics, search types, and index configurations.
These are simple replacements for the agno framework equivalents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class Distance(str, Enum):
    """
    Distance metrics for vector similarity calculations.

    Attributes:
        cosine: Cosine distance (1 - cosine similarity)
        l2: Euclidean (L2) distance
        max_inner_product: Maximum inner product (for normalized vectors)

    """

    cosine = "cosine"
    l2 = "l2"
    max_inner_product = "max_inner_product"


class SearchType(str, Enum):
    """
    Search types supported by vector database.

    Attributes:
        vector: Pure vector similarity search
        keyword: Full-text keyword search
        hybrid: Combined vector + keyword search

    """

    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"


class HNSW(BaseModel):
    """
    HNSW (Hierarchical Navigable Small World) index configuration.

    HNSW is an approximate nearest neighbor search algorithm that provides
    excellent performance for high-dimensional vector similarity search.

    Attributes:
        name: Optional index name (auto-generated if None)
        m: Maximum number of connections per layer (default: 16)
            Higher values = better recall, more memory, slower indexing
        ef_search: Size of dynamic candidate list for search (default: 5)
            Higher values = better recall, slower search
        ef_construction: Size of dynamic candidate list during construction (default: 200)
            Higher values = better quality index, slower construction
        configuration: PostgreSQL configuration for index creation

    Example:
        >>> index = HNSW(m=16, ef_search=10, ef_construction=200)

    """

    name: str | None = None
    m: int = 16
    ef_search: int = 5
    ef_construction: int = 200
    configuration: dict[str, Any] = {"maintenance_work_mem": "2GB"}


class Ivfflat(BaseModel):
    """
    IVFFlat (Inverted File with Flat Compression) index configuration.

    IVFFlat divides vectors into lists and then searches a subset of those lists
    that are closest to the query vector. Provides good performance with tuning.

    Attributes:
        name: Optional index name (auto-generated if None)
        lists: Number of inverted lists (default: 100)
            Rule of thumb: rows / 1000 for < 1M rows, sqrt(rows) for >= 1M rows
        probes: Number of lists to search (default: 10)
            Higher values = better recall, slower search
        dynamic_lists: Auto-calculate lists based on row count (default: True)
        configuration: PostgreSQL configuration for index creation

    Example:
        >>> index = Ivfflat(lists=100, probes=10, dynamic_lists=True)

    """

    name: str | None = None
    lists: int = 100
    probes: int = 10
    dynamic_lists: bool = True
    configuration: dict[str, Any] = {"maintenance_work_mem": "2GB"}
