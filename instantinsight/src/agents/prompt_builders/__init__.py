"""
Prompt builders for various agents in the instantinsight system.

This module contains specialised prompt building utilities organised by agent type.
Each agent's prompt logic is separated for better maintainability and testability.
"""

from .output import VisPromptBuilders
from .query import ModificationDecisionPrompts, QueryClarificationPrompts
from .schema import (
    ColumnMappingPrompts,
    SchemaFilterPrompts,
    SchemaTableSelectorPrompts,
    SchemaValidatorPrompts,
)
from .sql import SQLCorrectorPrompts, SQLFormatterPrompts, SQLGeneratorPrompts

__all__ = [
    # Schema prompts
    "SchemaTableSelectorPrompts",
    "ColumnMappingPrompts",
    "SchemaFilterPrompts",
    "SchemaValidatorPrompts",
    # Query prompts
    "QueryClarificationPrompts",
    "ModificationDecisionPrompts",
    # SQL prompts
    "SQLGeneratorPrompts",
    "SQLCorrectorPrompts",
    "SQLFormatterPrompts",
    # Output prompts
    "VisPromptBuilders",
]
