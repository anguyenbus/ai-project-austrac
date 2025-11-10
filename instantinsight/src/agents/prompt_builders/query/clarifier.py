"""
Query clarification-specific prompt building utilities.

This module contains prompt builders for the ClarificationAgent,
organized by context type and use case.
"""

from typing import Any

from ..prompts import Prompts


class QueryClarificationPrompts:
    """Utility class for building query clarification-specific LLM prompts."""

    @staticmethod
    def build_clarification_prompt(
        context: dict[str, Any], max_examples: int = 3
    ) -> str:
        """
        Build structured prompt for LLM clarification generation based on context type.

        Args:
            context: Analysis context from various sources
            max_examples: Maximum number of example queries to include

        Returns:
            Formatted prompt for LLM

        """
        # Use cached prompt and inject dynamic context
        base_prompt = Prompts.QUERY_CLARIFIER

        # Extract relevant context fields
        question = context.get("question", "")
        reasoning = context.get("reasoning", "Query needs clarification")
        selected_tables = context.get("selected_tables", [])
        confidence_scores = context.get("confidence_scores", {})
        related_tables = context.get("related_tables", [])
        sql = context.get("sql", "")
        confidence = context.get("confidence", 0.0)

        # Build context section
        context_section = f"""USER QUESTION: {question}

ANALYSIS REASONING: {reasoning}"""

        # Add optional context information
        if selected_tables:
            context_section += f"\nSELECTED TABLES: {', '.join(selected_tables)}"

        if confidence_scores:
            context_section += f"\nCONFIDENCE SCORES: {confidence_scores}"

        if related_tables:
            context_section += f"\nRELATED TABLES: {', '.join(related_tables)}"

        if sql:
            sql_preview = sql[:100] + "..." if len(sql) > 100 else sql
            context_section += f"\nGENERATED SQL: {sql_preview}"

        if confidence is not None:
            context_section += f"\nCONFIDENCE: {confidence:.2f}"

        # Add instruction about examples
        context_section += f"\n\nPlease provide {max_examples} clear example queries that would resolve the ambiguity."

        return f"{base_prompt}\n\n{context_section}"
