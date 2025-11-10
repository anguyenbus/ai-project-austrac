"""
Advanced Chunking Strategies for Text-to-SQL RAG System.

This module provides intelligent chunking strategies for different types of content
in a text-to-SQL system, optimized for pgvector storage and retrieval.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ChunkData:
    """Represents a single chunk of content."""

    chunk_type: str
    content: str
    metadata: dict[str, Any]
    token_count: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def estimate_tokens(self) -> int:
        """Estimate token count using simple word-based approximation."""
        if self.token_count is not None:
            return self.token_count

        # Rough approximation: words * 1.3 (accounting for subword tokens)
        word_count = len(self.content.split())
        self.token_count = int(word_count * 1.3)
        return self.token_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_type": self.chunk_type,
            "content": self.content,
            "metadata": self.metadata,
            "token_count": self.estimate_tokens(),
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, content: str, metadata: dict[str, Any]) -> list[ChunkData]:
        """Chunk content into semantic pieces."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        pass


class DDLChunkingStrategy(ChunkingStrategy):
    """Chunking strategy for SQL DDL (Data Definition Language) content."""

    def __init__(self, max_columns_per_chunk: int = 12, include_overview: bool = True):
        """
        Initialize DDL chunking strategy.

        Args:
            max_columns_per_chunk: Maximum columns per chunk
            include_overview: Whether to include table overview chunk

        """
        self.max_columns_per_chunk = max_columns_per_chunk
        self.include_overview = include_overview

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "ddl_chunking"

    def chunk(self, content: str, metadata: dict[str, Any]) -> list[ChunkData]:
        """Chunk DDL content into semantic pieces."""
        chunks = []

        # Extract table name
        table_name = self._extract_table_name(content)
        if not table_name and "table_name" in metadata:
            table_name = metadata["table_name"]

        logger.debug(f"Chunking DDL for table: {table_name}")

        # Chunk 1: Table overview (if enabled)
        if self.include_overview:
            overview = self._create_table_overview(content, table_name)
            if overview:
                chunks.append(
                    ChunkData(
                        chunk_type="table_overview",
                        content=overview,
                        metadata={
                            "tables": [table_name] if table_name else [],
                            "chunk_focus": "overview",
                            "original_table": table_name,
                        },
                    )
                )

        # Chunk 2-N: Column groups
        columns = self._extract_columns(content)
        if columns:
            for i in range(0, len(columns), self.max_columns_per_chunk):
                chunk_cols = columns[i : i + self.max_columns_per_chunk]
                col_content = self._format_columns_chunk(table_name, chunk_cols)

                chunks.append(
                    ChunkData(
                        chunk_type="columns",
                        content=col_content,
                        metadata={
                            "tables": [table_name] if table_name else [],
                            "columns": [
                                col.get("name", "")
                                for col in chunk_cols
                                if col.get("name")
                            ],
                            "chunk_focus": "columns",
                            "column_batch": i // self.max_columns_per_chunk + 1,
                            "original_table": table_name,
                        },
                    )
                )

        # Final chunk: Constraints and relationships
        constraints = self._extract_constraints(content, table_name)
        if constraints:
            referenced_tables = self._extract_referenced_tables(content)
            chunks.append(
                ChunkData(
                    chunk_type="constraints",
                    content=constraints,
                    metadata={
                        "tables": (
                            [table_name] + referenced_tables
                            if table_name
                            else referenced_tables
                        ),
                        "referenced_tables": referenced_tables,
                        "chunk_focus": "relationships",
                        "original_table": table_name,
                    },
                )
            )

        # Fallback: if no chunks created, create a single chunk
        if not chunks:
            chunks.append(
                ChunkData(
                    chunk_type="table_overview",
                    content=content[:2000],  # Truncate if too long
                    metadata={"original_table": table_name, "chunk_focus": "fallback"},
                )
            )

        logger.debug(f"Created {len(chunks)} chunks for table {table_name}")
        return chunks

    def _extract_table_name(self, ddl: str) -> str | None:
        """Extract table name from DDL."""
        # Handle both regular and external tables, with and without backticks
        patterns = [
            r"CREATE\s+(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`([^`]+)`",
            r"CREATE\s+(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+\.\w+)",
            r"CREATE\s+(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, ddl, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                # Remove database prefix if present
                if "." in table_name:
                    table_name = table_name.split(".")[-1]
                return table_name

        return None

    def _create_table_overview(self, ddl: str, table_name: str) -> str:
        """Create a table overview chunk with essential information."""
        lines = ddl.split("\n")
        overview_lines = []
        in_table_def = False
        column_count = 0
        sample_columns = []

        for line in lines:
            stripped = line.strip()

            # Capture CREATE TABLE line
            if "CREATE" in line.upper() and "TABLE" in line.upper():
                overview_lines.append(line)
                in_table_def = True
                continue

            # Count columns and capture first few
            if in_table_def and stripped.startswith("`"):
                column_count += 1
                if len(sample_columns) < 5:  # Keep first 5 columns
                    # Extract column name and type
                    col_match = re.match(r"`([^`]+)`\s+(\w+)", stripped)
                    if col_match:
                        sample_columns.append(
                            f"  {col_match.group(1)} ({col_match.group(2)})"
                        )

            # Stop at end of table definition
            elif in_table_def and ")" in stripped and not stripped.startswith("`"):
                break

        # Build overview
        overview = f"Table: {table_name}\n"
        overview += f"Total Columns: {column_count}\n\n"

        if sample_columns:
            overview += "Key Columns:\n" + "\n".join(sample_columns)
            if column_count > 5:
                overview += f"\n  ... and {column_count - 5} more columns"

        # Add table properties if available
        properties = self._extract_table_properties(ddl)
        if properties:
            overview += f"\n\nTable Properties:\n{properties}"

        return overview

    def _extract_columns(self, ddl: str) -> list[dict[str, Any]]:
        """Extract column definitions from DDL."""
        columns = []
        in_table_def = False

        for line in ddl.split("\n"):
            stripped = line.strip()

            if "CREATE" in line.upper() and "TABLE" in line.upper():
                in_table_def = True
                continue

            if in_table_def and stripped.startswith("`"):
                # Parse column definition
                col_match = re.match(r"`([^`]+)`\s+(\w+)(.*)$", stripped)
                if col_match:
                    name = col_match.group(1)
                    data_type = col_match.group(2)
                    rest = col_match.group(3).strip()

                    columns.append(
                        {
                            "name": name,
                            "type": data_type,
                            "definition": stripped,
                            "nullable": "NOT NULL" not in rest.upper(),
                            "full_line": line,
                        }
                    )

            elif in_table_def and ")" in stripped and not stripped.startswith("`"):
                break

        return columns

    def _format_columns_chunk(
        self, table_name: str, columns: list[dict[str, Any]]
    ) -> str:
        """Format columns into a readable chunk."""
        lines = [f"Table: {table_name}"]
        lines.append(f"Columns ({len(columns)} columns):\\n")

        for col in columns:
            # Format with name, type, and key properties
            col_info = f"  {col['name']} - {col['type']}"
            if not col.get("nullable", True):
                col_info += " (NOT NULL)"
            lines.append(col_info)

            # Add full definition for reference
            lines.append(f"    SQL: {col['definition']}")

        return "\\n".join(lines)

    def _extract_constraints(self, ddl: str, table_name: str) -> str:
        """Extract constraints, indexes, and relationships."""
        constraint_lines = []
        table_properties = []

        # Look for constraint-related keywords
        for line in ddl.split("\n"):
            line_upper = line.upper()
            if any(
                keyword in line_upper
                for keyword in [
                    "CONSTRAINT",
                    "PRIMARY KEY",
                    "FOREIGN KEY",
                    "UNIQUE",
                    "INDEX",
                    "KEY",
                    "REFERENCES",
                ]
            ):
                constraint_lines.append(line.strip())

        # Look for table properties
        if "TBLPROPERTIES" in ddl.upper():
            props_match = re.search(
                r"TBLPROPERTIES\s*\((.*?)\)", ddl, re.DOTALL | re.IGNORECASE
            )
            if props_match:
                table_properties.append(
                    f"Table Properties: {props_match.group(1).strip()}"
                )

        # Look for STORED AS, LOCATION, etc.
        for keyword in ["STORED AS", "LOCATION", "ROW FORMAT"]:
            pattern = rf"{keyword}\s+[^\\n]+"
            matches = re.findall(pattern, ddl, re.IGNORECASE)
            table_properties.extend(matches)

        if constraint_lines or table_properties:
            result = f"Table: {table_name}\\n\\n"

            if constraint_lines:
                result += "Constraints and Keys:\\n"
                result += "\\n".join(f"  {line}" for line in constraint_lines)

            if table_properties:
                if constraint_lines:
                    result += "\\n\\n"
                result += "Storage Properties:\\n"
                result += "\\n".join(f"  {prop}" for prop in table_properties)

            return result

        return ""

    def _extract_referenced_tables(self, ddl: str) -> list[str]:
        """Extract referenced table names from foreign keys."""
        matches = re.findall(r"REFERENCES\s+`?(\w+)`?", ddl, re.IGNORECASE)
        return list(set(matches))

    def _extract_table_properties(self, ddl: str) -> str:
        """Extract table properties and metadata."""
        properties = []

        # Storage format
        if "STORED AS" in ddl.upper():
            match = re.search(r"STORED AS\s+(\w+)", ddl, re.IGNORECASE)
            if match:
                properties.append(f"Storage Format: {match.group(1)}")

        # Input/Output format
        if "INPUTFORMAT" in ddl.upper():
            match = re.search(r"INPUTFORMAT\s+'([^']+)'", ddl, re.IGNORECASE)
            if match:
                properties.append(f"Input Format: {match.group(1).split('.')[-1]}")

        return ", ".join(properties) if properties else ""


class SQLExampleChunkingStrategy(ChunkingStrategy):
    """Chunking strategy for SQL examples (question-SQL pairs)."""

    def __init__(self, include_ctes: bool = True, max_sql_length: int = 1000):
        """
        Initialize SQL example chunking strategy.

        Args:
            include_ctes: Whether to include CTEs in chunks
            max_sql_length: Maximum SQL length per chunk

        """
        self.include_ctes = include_ctes
        self.max_sql_length = max_sql_length

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "sql_example_chunking"

    def chunk(self, content: str, metadata: dict[str, Any]) -> list[ChunkData]:
        """Chunk SQL examples into semantic pieces."""
        chunks = []

        # Parse content (could be JSON or plain text)
        question, sql = self._parse_sql_example(content, metadata)

        if not question and not sql:
            # Fallback for unparseable content
            return [
                ChunkData(
                    chunk_type="example_overview",
                    content=content[:1000],
                    metadata={"chunk_focus": "fallback"},
                )
            ]

        logger.debug(f"Chunking SQL example: {question[:50]}...")

        # Chunk 1: Question + SQL overview
        overview = self._create_example_overview(question, sql)
        chunks.append(
            ChunkData(
                chunk_type="example_overview",
                content=overview,
                metadata={
                    "question": question,
                    "sql_length": len(sql),
                    "tables": self._extract_tables_from_sql(sql),
                    "chunk_focus": "overview",
                },
            )
        )

        # Chunk 2: CTEs (if any and enabled)
        if self.include_ctes and "WITH" in sql.upper():
            ctes = self._extract_ctes(sql)
            if ctes:
                chunks.append(
                    ChunkData(
                        chunk_type="example_ctes",
                        content=f"Question: {question}\\n\\nCTE Definitions:\\n{ctes}",
                        metadata={
                            "question": question,
                            "chunk_focus": "ctes",
                            "tables": self._extract_tables_from_ctes(ctes),
                        },
                    )
                )

        # Chunk 3: Main query (if different from full SQL)
        main_query = self._extract_main_query(sql)
        if (
            main_query and len(main_query) < len(sql) * 0.8
        ):  # Only if significantly different
            chunks.append(
                ChunkData(
                    chunk_type="example_main",
                    content=f"Question: {question}\\n\\nMain Query:\\n{main_query}",
                    metadata={
                        "question": question,
                        "chunk_focus": "main_query",
                        "tables": self._extract_tables_from_sql(main_query),
                    },
                )
            )

        # Chunk 4: Complex query breakdown (for very complex queries)
        if len(sql) > self.max_sql_length:
            breakdown = self._create_query_breakdown(sql)
            if breakdown:
                chunks.append(
                    ChunkData(
                        chunk_type="example_breakdown",
                        content=f"Question: {question}\\n\\nQuery Structure:\\n{breakdown}",
                        metadata={
                            "question": question,
                            "chunk_focus": "structure",
                            "complexity": "high",
                        },
                    )
                )

        logger.debug(f"Created {len(chunks)} chunks for SQL example")
        return chunks

    def _parse_sql_example(
        self, content: str, metadata: dict[str, Any]
    ) -> tuple[str, str]:
        """Parse SQL example content to extract question and SQL."""
        try:
            # Try parsing as JSON (Vanna format)
            data = json.loads(content)
            question = data.get("question", "")
            sql = data.get("sql", "")
        except json.JSONDecodeError:
            # Fallback: extract from metadata or use content as SQL
            question = metadata.get("question", "")
            sql = content

        return question, sql

    def _create_example_overview(self, question: str, sql: str) -> str:
        """Create an overview chunk for the SQL example."""
        overview = f"Question: {question}\\n\\n"

        # Add SQL summary
        if len(sql) <= 300:
            overview += f"SQL Query:\\n{sql}"
        else:
            overview += f"SQL Query (truncated):\\n{sql[:250]}\\n... [query continues]"

        # Add query analysis
        analysis = self._analyze_sql_complexity(sql)
        if analysis:
            overview += f"\\n\\nQuery Analysis:\\n{analysis}"

        return overview

    def _extract_ctes(self, sql: str) -> str:
        """Extract CTE definitions from SQL."""
        # Find WITH clause
        with_match = re.search(r"WITH\s+(.+?)\s+SELECT", sql, re.IGNORECASE | re.DOTALL)
        if with_match:
            cte_text = with_match.group(1).strip()

            # Clean up and format CTEs
            ctes = []
            # Split on comma that's not inside parentheses
            depth = 0
            current_cte = ""

            for char in cte_text:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    if current_cte.strip():
                        ctes.append(current_cte.strip())
                    current_cte = ""
                    continue

                current_cte += char

            if current_cte.strip():
                ctes.append(current_cte.strip())

            return "\\n\\n".join(f"CTE {i + 1}:\\n{cte}" for i, cte in enumerate(ctes))

        return ""

    def _extract_main_query(self, sql: str) -> str:
        """Extract the main query (after CTEs)."""
        if "WITH" in sql.upper():
            # Find the main SELECT after WITH clause
            parts = re.split(r"\\bSELECT\\b", sql, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Return the last SELECT (main query)
                return "SELECT" + parts[-1].strip()

        return sql

    def _extract_tables_from_sql(self, sql: str) -> list[str]:
        """Extract table names from SQL query."""
        tables = []

        # Look for FROM clauses
        from_matches = re.findall(r"FROM\\s+([\\w.`]+)", sql, re.IGNORECASE)
        tables.extend(from_matches)

        # Look for JOIN clauses
        join_matches = re.findall(r"JOIN\\s+([\\w.`]+)", sql, re.IGNORECASE)
        tables.extend(join_matches)

        # Clean up table names (remove backticks, database prefixes)
        cleaned_tables = []
        for table in tables:
            table = table.replace("`", "")
            if "." in table:
                table = table.split(".")[-1]
            cleaned_tables.append(table)

        return list(set(cleaned_tables))

    def _extract_tables_from_ctes(self, ctes: str) -> list[str]:
        """Extract table names from CTE definitions."""
        return self._extract_tables_from_sql(ctes)

    def _analyze_sql_complexity(self, sql: str) -> str:
        """Analyze SQL complexity and provide summary."""
        features = []

        sql_upper = sql.upper()

        if "WITH" in sql_upper:
            cte_count = len(re.findall(r"\\bWITH\\b", sql_upper))
            features.append(f"Uses {cte_count} CTE(s)")

        if any(
            join in sql_upper
            for join in ["JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN"]
        ):
            join_count = len(re.findall(r"\\bJOIN\\b", sql_upper))
            features.append(f"Has {join_count} join(s)")

        if "UNION" in sql_upper:
            features.append("Uses UNION")

        if "SUBQUERY" in sql_upper or sql_upper.count("SELECT") > 1:
            features.append("Contains subqueries")

        if "GROUP BY" in sql_upper:
            features.append("Uses aggregation")

        if "WINDOW" in sql_upper or "OVER(" in sql_upper:
            features.append("Uses window functions")

        return "; ".join(features) if features else "Simple query"

    def _create_query_breakdown(self, sql: str) -> str:
        """Create a structural breakdown of complex queries."""
        breakdown = []

        # Identify major components
        if "WITH" in sql.upper():
            breakdown.append("1. Common Table Expressions (CTEs)")

        breakdown.append("2. Main SELECT clause")

        if "FROM" in sql.upper():
            breakdown.append("3. Data sources (FROM clause)")

        if any(join in sql.upper() for join in ["JOIN", "LEFT JOIN", "RIGHT JOIN"]):
            breakdown.append("4. Table joins")

        if "WHERE" in sql.upper():
            breakdown.append("5. Filtering conditions (WHERE)")

        if "GROUP BY" in sql.upper():
            breakdown.append("6. Grouping and aggregation")

        if "ORDER BY" in sql.upper():
            breakdown.append("7. Result ordering")

        return "\\n".join(breakdown)


class DocumentationChunkingStrategy(ChunkingStrategy):
    """Chunking strategy for documentation content."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize documentation chunking strategy.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters

        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "documentation_chunking"

    def chunk(self, content: str, metadata: dict[str, Any]) -> list[ChunkData]:
        """Chunk documentation using sliding window approach."""
        chunks = []

        # First try semantic chunking (by paragraphs/sections)
        semantic_chunks = self._semantic_chunk(content)

        if semantic_chunks and all(
            len(chunk) <= self.chunk_size for chunk in semantic_chunks
        ):
            # Use semantic chunks if they're reasonable size
            for i, chunk_content in enumerate(semantic_chunks):
                chunks.append(
                    ChunkData(
                        chunk_type="doc_section",
                        content=chunk_content,
                        metadata={
                            "section_index": i,
                            "chunk_focus": "documentation",
                            "semantic_chunk": True,
                        },
                    )
                )
        else:
            # Fall back to sliding window chunking
            chunks = self._sliding_window_chunk(content)

        return chunks or [
            ChunkData(
                chunk_type="doc_section",
                content=content[: self.chunk_size],
                metadata={"chunk_focus": "documentation", "fallback": True},
            )
        ]

    def _semantic_chunk(self, content: str) -> list[str]:
        """Chunk by semantic boundaries (paragraphs, sections)."""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split("\\n\\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(para) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\\n\\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _sliding_window_chunk(self, content: str) -> list[ChunkData]:
        """Chunk using sliding window approach."""
        chunks = []
        words = content.split()

        # Estimate words per chunk
        words_per_chunk = self.chunk_size // 6  # Rough estimate: 6 chars per word
        overlap_words = self.overlap // 6

        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)

            chunks.append(
                ChunkData(
                    chunk_type="doc_section",
                    content=chunk_content,
                    metadata={
                        "section_index": chunk_index,
                        "chunk_focus": "documentation",
                        "sliding_window": True,
                        "word_start": start,
                        "word_end": end,
                    },
                )
            )

            start = end - overlap_words  # Create overlap
            chunk_index += 1

        return chunks


class ChunkingStrategyFactory:
    """Factory for creating appropriate chunking strategies."""

    _strategies = {
        "schema": DDLChunkingStrategy,
        "sql_example": SQLExampleChunkingStrategy,
        "documentation": DocumentationChunkingStrategy,
    }

    @classmethod
    def create_strategy(cls, doc_type: str, **kwargs) -> ChunkingStrategy:
        """Create chunking strategy based on document type."""
        strategy_class = cls._strategies.get(doc_type)

        if not strategy_class:
            logger.warning(
                f"No chunking strategy for doc_type '{doc_type}', using documentation strategy"
            )
            strategy_class = DocumentationChunkingStrategy

        return strategy_class(**kwargs)

    @classmethod
    def register_strategy(cls, doc_type: str, strategy_class: type):
        """Register a new chunking strategy."""
        cls._strategies[doc_type] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available document types."""
        return list(cls._strategies.keys())


def chunk_document(
    doc_type: str, content: str, metadata: dict[str, Any] = None, **strategy_kwargs
) -> list[ChunkData]:
    """
    Chunk a document using the appropriate strategy.

    Args:
        doc_type: Type of document ('schema', 'sql_example', 'documentation')
        content: Document content to chunk
        metadata: Document metadata
        **strategy_kwargs: Additional arguments for the chunking strategy

    Returns:
        List of ChunkData objects

    """
    if metadata is None:
        metadata = {}

    strategy = ChunkingStrategyFactory.create_strategy(doc_type, **strategy_kwargs)
    return strategy.chunk(content, metadata)


# Example usage and testing
if __name__ == "__main__":
    # Test DDL chunking
    ddl_content = """
    CREATE EXTERNAL TABLE `text_to_sql`.`sales_data` (
      `id` bigint,
      `customer_id` bigint,
      `product_id` bigint,
      `sale_date` date,
      `amount` double,
      `currency` string,
      `status` string
    )
    STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
    OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
    LOCATION 's3://my-bucket/sales/'
    TBLPROPERTIES ('has_encrypted_data'='false')
    """

    chunks = chunk_document("schema", ddl_content, {"table_name": "sales_data"})
    print(f"DDL chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  - {chunk.chunk_type}: {len(chunk.content)} chars")

    # Test SQL example chunking
    sql_example = json.dumps(
        {
            "question": "Show me total sales by customer for the last month",
            "sql": "WITH monthly_sales AS (SELECT customer_id, SUM(amount) as total FROM sales_data WHERE sale_date >= date_sub(current_date(), 30) GROUP BY customer_id) SELECT * FROM monthly_sales ORDER BY total DESC",
        }
    )

    chunks = chunk_document("sql_example", sql_example)
    print(f"\\nSQL example chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  - {chunk.chunk_type}: {len(chunk.content)} chars")
