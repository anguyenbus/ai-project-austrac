"""
Shared utilities for the nl2vis project.

This module contains remaining utilities used across the codebase.

Note: Core connectors have been moved to src.connectors package.
LLM factory has been moved to src.llm package.
"""

# Import remaining utility modules for easy access
from .error_handler import (
    ConfigurationError,
    DatabaseError,
    ErrorHandler,
    LLMError,
    ServiceError,
)

__all__ = [
    # Utility classes
    "ErrorHandler",
    # Exception classes
    "DatabaseError",
    "LLMError",
    "ServiceError",
    "ConfigurationError",
]
