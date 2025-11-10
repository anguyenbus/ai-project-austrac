"""
Schema filter-specific prompt building utilities.

This module contains prompt builders for the FilteringAgent,
with structured prompt logic for extracting filtering criteria from natural language queries.
"""

from datetime import datetime

from ..prompts.schema_filter import (
    FILTER_EXTRACTION_REQUEST_TEMPLATE,
    FILTER_EXTRACTION_SYSTEM_PROMPT,
)


class SchemaFilterPrompts:
    """Utility class for building schema filter-specific LLM prompts."""

    @staticmethod
    def build_extraction_prompt(
        query: str, normalized_hints: list[str] | None = None
    ) -> str:
        """
        Build prompt for LLM filter extraction.

        Args:
            query: Natural language query to extract filters from
            normalized_hints: Optional list of normalized hints guiding extraction

        Returns:
            Formatted prompt for LLM filter extraction

        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        previous_year = current_year - 1

        # Build hint section for template
        hint_section = ""
        if normalized_hints:
            formatted = "\n".join(
                f"- {hint}"
                for hint in normalized_hints
                if isinstance(hint, str) and hint.strip()
            )
            if formatted:
                hint_section = f"\n\nNORMALIZED FILTER HINTS:\n{formatted}"

        # Use cached system prompt with request template
        request_content = FILTER_EXTRACTION_REQUEST_TEMPLATE.format(
            current_date=current_date,
            current_year=current_year,
            previous_year=previous_year,
            hint_section=hint_section,
            query=query,
        )

        return f"{FILTER_EXTRACTION_SYSTEM_PROMPT}\n\n{request_content}"
