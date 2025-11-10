"""
RAG (Retrieval-Augmented Generation) module for nl2sql2vis.

This module provides the core RAGEngine implementation with modular components.
"""

# Import modular components
from .components import (  # ConversationManager,
    DatabaseManager,
    EngineConfig,
    ErrorHandlingService,
    SQLGenerationService,
)

# Import the main engine
from .rag_engine import RAGEngine, create_rag_engine

# Default exports
__all__ = [
    "RAGEngine",
    "create_rag_engine",
    "DatabaseManager",
    "ConversationManager",
    "SQLGenerationService",
    "ErrorHandlingService",
    "EngineConfig",
]
