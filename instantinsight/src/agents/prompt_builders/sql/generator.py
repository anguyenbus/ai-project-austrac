"""
SQL generator-specific prompt building utilities.

This module contains prompt builders for the SQLWriterAgent,
with structured prompt logic for SQL query generation from natural language.
"""

from ..prompts import Prompts


class SQLGeneratorPrompts:
    """Utility class for building SQL generator-specific LLM prompts."""

    @staticmethod
    def build_sql_generation_prompt(
        question: str,
        context_text: str,
        selected_tables: list[str],
        filter_prompt: str,
    ) -> str:
        """
        Build prompt for LLM SQL generation from natural language.

        Args:
            question: Natural language query
            context_text: Schema and example context
            selected_tables: List of selected table names
            filter_prompt: Filter context prompt section

        Returns:
            Formatted prompt for LLM SQL generation

        """
        # Use cached prompt and inject dynamic variables
        base_prompt = Prompts.SQL_GENERATOR

        # Add selected tables info to the prompt
        selected_tables_info = f"Selected tables: {selected_tables if selected_tables else 'Auto-selected'}"

        return f"""{base_prompt}

{selected_tables_info}

SCHEMA INFORMATION (Format: Table Name followed by columns with * prefix):
{context_text}

Question: {question}{filter_prompt}"""
