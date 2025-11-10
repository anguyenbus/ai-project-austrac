"""
Configuration module for instantinsight project.

This module contains all configuration files including database settings,
DDL schemas, and application configuration.
"""

from .config import DDL
from .database_config import (
    ATHENA_CONFIG,
    AUTO_EXTRACT_SCHEMA,
    BEDROCK_CONFIG,
    FORCE_REBUILD_VECTOR_STORE,
    POSTGRES_CONFIG,
    RAG_CONFIG,
)

__all__ = [
    "DDL",
    "ATHENA_CONFIG",
    "POSTGRES_CONFIG",
    "RAG_CONFIG",
    "AUTO_EXTRACT_SCHEMA",
    "FORCE_REBUILD_VECTOR_STORE",
    "BEDROCK_CONFIG",
]
