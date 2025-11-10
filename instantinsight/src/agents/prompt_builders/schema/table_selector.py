"""
Schema table selector-specific prompt building utilities.

This module contains prompt builders for the TableAgent,
with structured prompt logic for intelligent table selection from database schemas.
"""

from ..prompts import Prompts


class SchemaTableSelectorPrompts:
    """Utility class for building schema table selector-specific LLM prompts."""

    @staticmethod
    def build_table_selection_prompt(query: str, schema_context: str) -> str:
        """
        Build prompt for LLM table selection from database schema.

        Args:
            query: Natural language query to find tables for
            schema_context: Formatted schema context with table information

        Returns:
            Formatted prompt for LLM table selection

        """
        # Use cached prompt and inject dynamic variables
        base_prompt = Prompts.SCHEMA_TABLE_SELECTOR

        return f"""{base_prompt}

USER QUERY: "{query}"

AVAILABLE ANALYSERS (shown as schemas with JOIN NOTES):
{schema_context}"""
