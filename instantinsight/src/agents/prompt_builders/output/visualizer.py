"""Visualization-specific prompt building utilities."""

import json
from typing import Any

import pandas as pd

from ..prompts import Prompts


class VisPromptBuilders:
    """Utility class for building visualization-specific LLM prompts."""

    @staticmethod
    def build_visualization_prompt(
        df: pd.DataFrame,
        query: str,
        analysis: dict[str, Any],
        retry_context: str = None,
        visual_hint: str | None = None,
    ) -> str:
        """Build prompt for LLM to analyze data and recommend visualization."""
        # Use cached prompt and inject dynamic variables
        base_prompt = Prompts.OUTPUT_VISUALIZER

        sample_df = df.head(min(10, len(df)))

        # Build hint section
        hint_section = ""
        if visual_hint:
            hint_section = (
                f"\nUSER VISUAL PREFERENCE:\n"
                f'The user mentioned: "{visual_hint}"\n'
                "Consider this as a suggested chart type, but prioritize data suitability."
            )

        # Build data context section
        data_context = f"""USER QUERY: {query}

DATA OVERVIEW:
- Rows: {analysis["row_count"]}
- Columns: {analysis["column_count"]}

COLUMN ANALYSIS:
{json.dumps(analysis["columns"], indent=2)}

SAMPLE DATA:
{sample_df.to_string()}"""

        # Combine all sections
        full_prompt = f"{base_prompt}\n\n{data_context}"

        if hint_section:
            full_prompt += f"\n\n{hint_section}"

        if retry_context:
            full_prompt += f"\n\nPREVIOUS ERROR:\n{retry_context}\n\nPlease correct the configuration to fix this error."

        return full_prompt
