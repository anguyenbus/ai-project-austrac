"""
Schema validator-specific prompt building utilities.

This module contains prompt builders for the SchemaValidatorAgent,
with structured prompt logic for SQL component extraction and validation.
"""


class SchemaValidatorPrompts:
    """Utility class for building schema validator-specific LLM prompts."""

    @staticmethod
    def build_sql_component_extraction_prompt(sql: str) -> str:
        """
        Build prompt for LLM SQL component extraction.

        Args:
            sql: SQL query string to analyze

        Returns:
            Formatted prompt for LLM SQL component extraction

        """
        return f"""
You are an expert SQL parser. Analyze the following SQL query and extract all table references and column references.

Find all table names that appear in:
- FROM clauses
- JOIN clauses (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, etc.)
- UPDATE statements
- INSERT INTO statements
- DELETE FROM statements
- Subqueries

Also extract all column references:
- In SELECT clause (before AS keyword, not the aliases)
- In WHERE clause
- In JOIN conditions
- In GROUP BY clause
- In ORDER BY clause
- Map each column to its source table

Important rules:
1. Extract only the table name, not aliases
2. Remove any database/schema prefixes (e.g., "database.table" becomes "table")
3. Remove quotes, backticks, or brackets around table names and columns
4. Ignore CTE (Common Table Expression) names - they are not real tables
5. Ignore temporary table references that start with "#" or "@"
6. For columns, extract the actual column name, not the alias after AS
7. If a column has table prefix (e.g., table.column), map it to that table
8. If a column has no prefix, try to determine which table it belongs to based on context
9. DO NOT extract COUNT(*) as columns

SQL Query:
```sql
{sql}
```

Extract all table names, column references, and provide analysis of the query structure.
"""
