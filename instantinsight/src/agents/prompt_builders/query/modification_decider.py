"""
Modification decision-specific prompt building utilities.

This module contains prompt builders for the ModificationDecisionAgent,
with structured prompt logic for analyzing visualization modification requests.
"""

import json
from typing import Any

from ..prompts import Prompts


class ModificationDecisionPrompts:
    """Utility class for building modification decision-specific LLM prompts."""

    @staticmethod
    def build_decision_prompt(
        historical_context: str,
        user_message: str,
        current_sql: str,
        current_plotly_schema: dict[str, Any],
    ) -> str:
        """
        Build prompt for LLM to analyze modification request.

        Args:
            historical_context: Full conversation history
            user_message: Current user modification request
            current_sql: Current SQL query generating the data
            current_plotly_schema: Current Plotly chart schema

        Returns:
            Formatted prompt for LLM decision making

        """
        # Use cached prompt and inject dynamic variables
        base_prompt = Prompts.QUERY_MODIFICATION_DECIDER

        # Convert schema to readable string
        schema_str = (
            json.dumps(current_plotly_schema, indent=2)
            if current_plotly_schema
            else "No schema available"
        )

        # Build context section for the prompt
        context_section = f"""HISTORICAL CONTEXT:
{historical_context}

CURRENT USER REQUEST:
{user_message}

CURRENT SQL QUERY:
{current_sql}

CURRENT PLOTLY SCHEMA:
{schema_str}"""

        return f"{base_prompt}\n\n{context_section}"
