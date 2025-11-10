"""
Shared utilities for the nl2vis project.

This module contains remaining utilities used across the codebase.

Note: Core connectors have been moved to src.connectors package.
LLM factory has been moved to src.llm package.
AthenaSchemaVectorizer should be imported directly to avoid circular imports.
"""

# Import remaining utility modules for easy access
from .error_handler import (
    ConfigurationError,
    DatabaseError,
    ErrorHandler,
    LLMError,
    ServiceError,
)

# Note: SchemaVectorizer not imported here to avoid circular imports
# Import it directly: from src.utils.schema_vectorizer import SchemaVectorizer
# Legacy: from src.utils.schema_vectorizer import AthenaSchemaVectorizer (backward compatible alias)

__all__ = [
    # Utility classes
    "ErrorHandler",
    # Exception classes
    "DatabaseError",
    "LLMError",
    "ServiceError",
    "ConfigurationError",
]
