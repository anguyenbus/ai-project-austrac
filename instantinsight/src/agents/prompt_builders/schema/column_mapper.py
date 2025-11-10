"""
Column mapping-specific prompt building utilities.

This module contains prompt builders for the ColumnAgent,
with type-aware logic for better SQL value mapping.
"""

from typing import Any

from ..prompts.column_mapper import (
    CANDIDATE_REQUEST_TEMPLATE,
    CANDIDATE_SYSTEM_PROMPT,
    MAPPING_REQUEST_TEMPLATE,
    MAPPING_SYSTEM_PROMPT,
)


class ColumnMappingPrompts:
    """Utility class for building column mapping-specific LLM prompts."""

    @staticmethod
    def build_mapping_prompt(
        filters: list[dict[str, Any]],
        columns_info: dict[str, list[dict[str, str]]],
        categorical_mappings: dict[str, dict[str, list[str]]],
        tables: list[str],
    ) -> str:
        """
        Build type-aware prompt for column mapping that respects column data types.

        Args:
            filters: Original filters from FilteringAgent
            columns_info: Column information for tables (includes name and type)
            categorical_mappings: Distinct values for categorical columns
            tables: Selected table names

        Returns:
            Enhanced prompt for LLM with type awareness

        """
        # Build enhanced columns display with type information
        columns_display = []
        for table, cols in columns_info.items():
            if cols:
                col_info = [f"{c['name']} ({c['type']})" for c in cols]
                columns_display.append(f"{table}: {', '.join(col_info)}")

        # Build categorical values display with column-specific results
        categorical_display = []
        for filter_key, column_values_map in categorical_mappings.items():
            if column_values_map:
                categorical_display.append(
                    f"Filter '{filter_key}' candidate columns and values:"
                )
                for column_name, values in column_values_map.items():
                    if len(values) <= 10:
                        categorical_display.append(
                            f"  - Column '{column_name}': {values}"
                        )
                    else:
                        categorical_display.append(
                            f"  - Column '{column_name}': {values[:10]} ... ({len(values)} total)"
                        )

        # Build request sections for the template
        filters_section = str(filters)
        columns_section = "\n".join(columns_display)
        categorical_section = (
            "\n".join(categorical_display)
            if categorical_display
            else "None found via search"
        )
        tables_section = str(tables)

        # Use cached system prompt with request template
        request_content = MAPPING_REQUEST_TEMPLATE.format(
            filters=filters_section,
            columns_section=columns_section,
            categorical_section=categorical_section,
            tables_section=tables_section,
        )

        return f"{MAPPING_SYSTEM_PROMPT}\n\n{request_content}"

    @staticmethod
    def build_candidate_identification_prompt(
        filter_key: str,
        columns_info: dict[str, list[dict[str, str]]],
        schema_context: str = None,
        question: str = None,
    ) -> str:
        """
        Build prompt for identifying candidate columns that match a filter key.

        Args:
            filter_key: Filter key (e.g., 'document_description', 'city')
            columns_info: Column information for all tables
            schema_context: Optional schema DDL
            question: Original question for context

        Returns:
            Formatted prompt for LLM candidate identification

        """
        # Build columns display with type information for better matching
        columns_display = []
        for table, cols in columns_info.items():
            if cols:
                col_info = [f"{c['name']} ({c['type']})" for c in cols]
                columns_display.append(f"Table {table}:")
                columns_display.append(f"  Columns: {', '.join(col_info)}")

        # Build request sections for the template
        question_section = question or "Not provided"
        columns_section = "\n".join(columns_display)
        schema_context_section = schema_context or "Not provided"

        # Use cached system prompt with request template
        request_content = CANDIDATE_REQUEST_TEMPLATE.format(
            question_section=question_section,
            filter_key=filter_key,
            columns_section=columns_section,
            schema_context=schema_context_section,
        )

        return f"{CANDIDATE_SYSTEM_PROMPT}\n\n{request_content}"
