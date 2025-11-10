"""
SQL formatter-specific prompt building utilities.

This module contains prompt builders for the SQLSpacingAgent,
with structured prompt logic for SQL spacing and formatting corrections.
"""


class SQLFormatterPrompts:
    """Utility class for building SQL formatter-specific LLM prompts."""

    @staticmethod
    def build_spacing_fix_prompt(sql: str) -> str:
        """
        Build prompt for LLM SQL spacing and formatting fix.

        Args:
            sql: SQL query with potential spacing issues

        Returns:
            Formatted prompt for LLM SQL spacing fix

        """
        return f"""You are a SQL formatting expert. Fix spacing issues in this SQL query for AWS Athena compatibility.

ORIGINAL SQL:
{sql}

SPACING RULES TO APPLY:
1. Add spaces before keywords: FROM, WHERE, GROUP BY, ORDER BY, JOIN, ON, AND, OR, HAVING, LIMIT
2. Add spaces around operators: =, <, >, <=, >=, !=, <>, +, -, *, /
3. Add space after commas
4. Convert backticks (`) to double quotes (") for Athena compatibility
5. Ensure proper spacing for parentheses in functions

IMPORTANT CONSTRAINTS:
- ONLY fix spacing and quote issues
- DO NOT change table names, column names, or query logic
- DO NOT add unnecessary line breaks (keep as single line unless extremely long)
- DO NOT change the semantic meaning of the query
- Preserve original case of identifiers

Analyze the SQL and provide:
1. fixed_sql: The corrected SQL with proper spacing
2. issues_found: List specific spacing issues found (with location and type)
3. confidence: Your confidence level (0.0-1.0) in the fixes
4. requires_fixes: Whether any fixes were needed
5. athena_compatible: Whether the result is Athena-compatible
6. fix_summary: Brief summary of what was fixed

If the SQL already has correct spacing, return it unchanged with requires_fixes=false."""
