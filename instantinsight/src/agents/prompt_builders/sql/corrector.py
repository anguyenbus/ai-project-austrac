"""
SQL corrector-specific prompt building utilities.

This module contains prompt builders for the SQLFixer,
with structured prompt logic for SQL error correction and refinement.
"""


class SQLCorrectorPrompts:
    """Utility class for building SQL corrector-specific LLM prompts."""

    @staticmethod
    def build_sql_fix_prompt(
        sql: str, error: str, schema_context: dict | None = None
    ) -> str:
        """
        Build prompt for LLM SQL error correction.

        Args:
            sql: Broken SQL query
            error: Database error message
            schema_context: Optional schema information

        Returns:
            Formatted prompt for LLM SQL error correction

        """
        # Build schema information if available
        schema_info = ""
        if schema_context:
            if "tables" in schema_context:
                tables = schema_context["tables"]
                schema_info += f"Available tables: {', '.join(tables[:10])}"
                if len(tables) > 10:
                    schema_info += f" ... (+{len(tables) - 10} more)"
                schema_info += "\n"

            if "columns" in schema_context:
                schema_info += "Column information available for referenced tables.\n"

        # Build the focused prompt
        return f"""Fix this SQL query that caused a database execution error.

ERROR MESSAGE:
{error}

BROKEN SQL:
{sql}

{schema_info}

INSTRUCTIONS:
1. Analyze the error message to understand what went wrong
2. Fix the specific issue mentioned in the error
3. Keep the original query logic and intent
4. Ensure the SQL is valid for AWS Athena (Presto dialect)
5. CRITICAL: Add proper spacing between ALL SQL keywords and clauses
6. Replace backticks with double quotes if needed
7. Only use tables and columns that exist in the schema

SPACING REQUIREMENTS (MANDATORY):
- ALWAYS add space after SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, etc.
- NEVER write: SELECT columnFROM table
- ALWAYS write: SELECT column FROM table
- NEVER write: WHEREcondition or FROMtable
- ALWAYS write: WHERE condition and FROM table
- Each SQL keyword must have spaces before and after it

CRITICAL RULES:
- Return the corrected SQL query with PROPER SPACING
- Every SQL keyword (SELECT, FROM, WHERE, etc.) MUST have spaces around it
- Use exact table and column names from the schema
- Maintain the original query's business logic

Provide the corrected SQL query that fixes the error with proper spacing."""
