"""
RAG evaluation metrics for text2SQL.

Focuses on practical metrics without requiring golden answers.
"""

import re

import sqlglot
from sqlglot import exp


class RAGMetrics:
    """Minimal, maintainable RAG evaluation metrics for text2SQL."""

    @staticmethod
    def extract_context_identifiers(context: str) -> set[str]:
        """
        Extract table and column identifiers from retrieved context.

        Handles structured format with 'Table Name:' and '* column_name:' patterns.
        Returns normalized lowercase identifiers for case-insensitive comparison.

        Args:
            context: Retrieved context string to extract identifiers from

        Returns:
            Set of normalized lowercase table and column identifiers

        """
        identifiers = set()
        lines = context.splitlines()

        current_table = ""
        in_sql_examples = False

        for line in lines:
            line = line.strip()

            # Stop processing when we hit SQL EXAMPLES section
            if line.startswith("SQL EXAMPLES:") or line.startswith("Example SQL"):
                in_sql_examples = True
                continue

            # Skip lines in SQL examples section
            if in_sql_examples:
                continue

            # Extract table name
            if line.startswith("Table Name:"):
                table_name = line.split("Table Name:", 1)[1].strip()
                # Handle fully qualified names
                if "." in table_name:
                    table_name = table_name.split(".")[-1]
                current_table = table_name.lower().replace(" ", "_")
                identifiers.add(current_table)

            # Extract column names
            elif line.startswith("* ") or line.startswith("- "):
                # Handle both '* column_name:' and '- column_name:' formats
                col_part = line[2:].strip()
                if ":" in col_part:
                    col_name = col_part.split(":", 1)[0].strip()
                else:
                    col_name = col_part.strip()

                # Normalize column name
                col_name = col_name.lower().replace(" ", "_")
                if col_name:
                    identifiers.add(col_name)
                    # Also add table.column format if we have current table
                    if current_table:
                        identifiers.add(f"{current_table}.{col_name}")

        # Also try to extract from any SQL statements in the context
        # Look for FROM and JOIN clauses
        sql_pattern = r"(?:FROM|JOIN)\s+([a-zA-Z0-9_\.]+)"
        for match in re.finditer(sql_pattern, context, re.IGNORECASE):
            table = match.group(1).lower()
            if "." in table:
                table = table.split(".")[-1]

            identifiers.add(table)

        return identifiers

    @staticmethod
    def extract_identifiers(sql: str) -> set[str]:
        """
        Extract all table and column identifiers from SQL.

        Returns normalized lowercase identifiers for case-insensitive comparison.
        Properly handles quoted identifiers with spaces.
        Filters out SQL keywords and aliases.
        """
        # Common SQL keywords to filter out
        sql_keywords = {
            "select",
            "from",
            "where",
            "and",
            "or",
            "group",
            "by",
            "order",
            "having",
            "limit",
            "as",
            "on",
            "join",
            "inner",
            "left",
            "right",
            "outer",
            "count",
            "sum",
            "avg",
            "max",
            "min",
            "distinct",
            "case",
            "when",
            "then",
            "else",
            "end",
            "null",
            "true",
            "false",
            "asc",
            "desc",
            "in",
            "not",
            "like",
            "between",
            "exists",
            "all",
            "any",
            "union",
            "intersect",
            "except",
            "create",
            "table",
            "external",
            "if",
            "with",
        }

        def normalize_identifier(identifier: str) -> str:
            """
            Normalize identifier by removing quotes and converting to lowercase.

            Also normalizes spaces to underscores for consistent matching.
            """
            if not identifier:
                return ""
            # Remove surrounding quotes (both single and double)
            normalized = identifier.strip()
            if (normalized.startswith('"') and normalized.endswith('"')) or (
                normalized.startswith("'") and normalized.endswith("'")
            ):
                normalized = normalized[1:-1]

            # Convert to lowercase and normalize spaces to underscores
            normalized = normalized.lower().replace(" ", "_")
            return normalized

        identifiers = set()

        # Parse SQL once
        parsed = sqlglot.parse_one(sql)

        # Extract aliases to exclude them
        aliases = set()
        for alias_node in parsed.find_all(exp.Alias):
            if alias_node.alias:
                alias_name = normalize_identifier(str(alias_node.alias))
                aliases.add(alias_name)

        # Extract tables
        for table in parsed.find_all(exp.Table):
            if table.this:
                table_name = normalize_identifier(str(table.this))
                if (
                    table_name
                    and table_name not in sql_keywords
                    and table_name not in aliases
                ):
                    identifiers.add(table_name)

        # Extract columns (including those with table prefixes)
        for column in parsed.find_all(exp.Column):
            if column.this:
                col_name = normalize_identifier(str(column.this))
                if (
                    col_name
                    and col_name not in sql_keywords
                    and col_name not in aliases
                ):
                    identifiers.add(col_name)

                # If column has table prefix, add both parts
                if column.table:
                    table_name = normalize_identifier(str(column.table))
                    if (
                        table_name
                        and table_name not in sql_keywords
                        and table_name not in aliases
                    ):
                        identifiers.add(table_name)
                        identifiers.add(f"{table_name}.{col_name}")

        return identifiers

    @staticmethod
    def sql_groundedness_score(generated_sql: str, retrieved_context: str) -> float:
        """
        Measure if generated SQL only uses identifiers from retrieved context.

        This is renamed from schema_adherence_score for clarity.

        Returns: 0.0 to 1.0 (1.0 = all SQL identifiers found in context)
        """
        sql_ids = RAGMetrics.extract_identifiers(generated_sql)
        context_ids = RAGMetrics.extract_context_identifiers(retrieved_context)

        if not sql_ids:
            return 1.0  # No identifiers = perfect groundedness

        # Check how many SQL identifiers are in the context
        grounded_ids = sql_ids.intersection(context_ids)
        return len(grounded_ids) / len(sql_ids)

    @staticmethod
    def schema_adherence_score(generated_sql: str, retrieved_context: str) -> float:
        """
        Legacy method - kept for backward compatibility.

        Use sql_groundedness_score instead.
        """
        return RAGMetrics.sql_groundedness_score(generated_sql, retrieved_context)

    @staticmethod
    def hallucination_check(
        generated_sql: str, all_valid_tables: set[str], all_valid_columns: set[str]
    ) -> tuple[bool, list[str]]:
        """
        Check if SQL references non-existent database objects.

        Returns: (has_hallucination, list_of_hallucinated_identifiers).
        """
        sql_ids = RAGMetrics.extract_identifiers(generated_sql)

        # Combine all valid identifiers
        all_valid = {id.lower() for id in all_valid_tables.union(all_valid_columns)}

        # Find identifiers not in valid set
        # Filter out common SQL keywords that might be picked up
        sql_keywords = {
            "select",
            "from",
            "where",
            "and",
            "or",
            "group",
            "by",
            "order",
            "having",
            "limit",
            "as",
            "on",
            "join",
            "inner",
            "left",
            "right",
            "outer",
            "count",
            "sum",
            "avg",
            "max",
            "min",
        }

        hallucinated = [
            id for id in sql_ids if id not in all_valid and id not in sql_keywords
        ]

        return len(hallucinated) > 0, hallucinated

    @staticmethod
    def context_precision_score(generated_sql: str, retrieved_context: str) -> float:
        """
        Measure how much of the retrieved context was actually useful.

        High score = context was focused and relevant.
        Low score = context contained many irrelevant identifiers.

        Returns: 0.0 to 1.0 (1.0 = all context identifiers were used)
        """
        sql_ids = RAGMetrics.extract_identifiers(generated_sql)
        context_ids = RAGMetrics.extract_context_identifiers(retrieved_context)

        if not context_ids:
            return 1.0  # No context provided = perfect precision (vacuously true)

        # Check how many context identifiers were actually used in SQL
        used_ids = sql_ids.intersection(context_ids)
        return len(used_ids) / len(context_ids)

    @staticmethod
    def context_recall_score(
        generated_sql: str,
        retrieved_context: str,
        expected_sql: str | None = None,
        successful_execution: bool = False,
    ) -> float:
        """
        Measure if retriever fetched all necessary schema components.

        If expected_sql is provided: measures against the golden standard
        If only successful_execution: uses generated SQL as proxy (less reliable)

        Returns: 0.0 to 1.0 (1.0 = perfect recall)
        """
        # Method 1: Use golden SQL if available (preferred)
        if expected_sql:
            expected_ids = RAGMetrics.extract_identifiers(expected_sql)
            context_ids = RAGMetrics.extract_context_identifiers(retrieved_context)

            if not expected_ids:
                return 1.0  # No identifiers needed = perfect recall

            recalled_ids = expected_ids.intersection(context_ids)
            return len(recalled_ids) / len(expected_ids)

        # Method 2: Fallback to generated SQL (less reliable)
        if not successful_execution:
            return 0.0  # Can't measure recall if query failed

        sql_ids = RAGMetrics.extract_identifiers(generated_sql)
        context_ids = RAGMetrics.extract_context_identifiers(retrieved_context)

        if not sql_ids:
            return 1.0  # No identifiers needed = perfect recall

        # Check how many required identifiers were in retrieved context
        recalled_ids = sql_ids.intersection(context_ids)
        return len(recalled_ids) / len(sql_ids)

    @staticmethod
    def context_f1_score(
        generated_sql: str,
        retrieved_context: str,
        expected_sql: str | None = None,
        successful_execution: bool = False,
    ) -> float:
        """
        Balanced F1-score combining context precision and recall.

        Returns: 0.0 to 1.0 (1.0 = perfect balance of precision and recall)
        """
        precision = RAGMetrics.context_precision_score(generated_sql, retrieved_context)
        recall = RAGMetrics.context_recall_score(
            generated_sql, retrieved_context, expected_sql, successful_execution
        )

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def sql_validity_score(generated_sql: str) -> float:
        """
        Check if the generated SQL is syntactically valid.

        Returns: 1.0 if valid, 0.0 if invalid
        """
        try:
            # Attempt to parse the SQL
            sqlglot.parse_one(generated_sql)
            return 1.0
        except Exception:
            return 0.0

    @staticmethod
    def consistency_score(sql_queries: list[str]) -> float:
        """
        Measure consistency across multiple SQL generations for same question.

        Compares normalized SQL structure.
        Returns: 0.0 to 1.0 (1.0 = perfectly consistent).
        """
        if len(sql_queries) < 2:
            return 1.0  # Single query = perfectly consistent

        # Normalize all queries
        normalized_queries = []
        for sql in sql_queries:
            try:
                # Parse and format to normalize
                parsed = sqlglot.parse_one(sql)
                normalized = parsed.sql(pretty=False, normalize=True)
                normalized_queries.append(normalized.lower())
            except Exception:
                # If parsing fails, use lowercase version
                normalized_queries.append(sql.lower().strip())

        # Count unique normalized queries
        unique_queries = len(set(normalized_queries))

        # Perfect consistency = 1 unique query, worst = all different
        consistency = 1.0 - ((unique_queries - 1) / len(sql_queries))
        return max(0.0, consistency)

    @staticmethod
    def evaluate_all(
        generated_sql: str,
        retrieved_context: str,
        all_valid_tables: set[str] | None = None,
        all_valid_columns: set[str] | None = None,
        successful_execution: bool = False,
        multiple_generations: list[str] | None = None,
        expected_sql: str | None = None,
    ) -> dict:
        """
        Run all metrics and return results dictionary.

        This is the main entry point for evaluation.
        """
        results = {
            # Core RAG metrics (improved)
            "sql_groundedness": RAGMetrics.sql_groundedness_score(
                generated_sql, retrieved_context
            ),
            "context_precision": RAGMetrics.context_precision_score(
                generated_sql, retrieved_context
            ),
            "context_recall": RAGMetrics.context_recall_score(
                generated_sql, retrieved_context, expected_sql, successful_execution
            ),
            "context_f1": RAGMetrics.context_f1_score(
                generated_sql, retrieved_context, expected_sql, successful_execution
            ),
            # SQL quality metrics
            "sql_validity": RAGMetrics.sql_validity_score(generated_sql),
            "execution_success": successful_execution,
            # Legacy metrics (for backward compatibility)
            "schema_adherence": RAGMetrics.schema_adherence_score(
                generated_sql, retrieved_context
            ),
        }

        # Hallucination check (if valid schema provided)
        if all_valid_tables and all_valid_columns:
            has_hallucination, hallucinated_ids = RAGMetrics.hallucination_check(
                generated_sql, all_valid_tables, all_valid_columns
            )
            results["has_hallucination"] = has_hallucination
            results["hallucinated_identifiers"] = hallucinated_ids

        # Consistency check (if multiple generations provided)
        if multiple_generations and len(multiple_generations) > 1:
            results["consistency"] = RAGMetrics.consistency_score(multiple_generations)

        return results
