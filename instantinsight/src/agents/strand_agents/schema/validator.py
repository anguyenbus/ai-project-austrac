"""
Schema Validation Agent for ensuring table existence in Athena using Strands framework.

Handles SQL component extraction, table validation, and column validation using Strands Agent
for structured LLM responses with better reliability and performance.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.schema.validator import SchemaValidatorPrompts
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ..model_config import get_agent_config

# Import for proper database cursor row factory
try:  # pragma: no cover - import guard for environments without psycopg3
    from psycopg.rows import dict_row

    DICT_ROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    dict_row = None  # type: ignore
    DICT_ROW_AVAILABLE = False


@dataclass
class TableValidationResult:
    """Results of table validation."""

    table_name: str
    exists: bool
    error: str | None = None


@dataclass
class ColumnValidationResult:
    """Results of column validation."""

    column_name: str
    table_name: str
    exists: bool
    error: str | None = None


@dataclass
class SchemaValidationResult:
    """Complete schema validation results."""

    valid_tables: list[str]
    invalid_tables: list[str]
    table_details: dict[str, TableValidationResult]
    validation_passed: bool
    # New fields for column validation
    column_validation: dict[str, list[ColumnValidationResult]] | None = None
    invalid_columns: list[str] | None = None


class SQLComponentExtraction(BaseModel):
    """Structured extraction of SQL components using Pydantic for Strands compatibility."""

    table_names: list[str] = Field(
        description="All table names referenced in the SQL query (from FROM, JOIN, UPDATE, INSERT INTO, DELETE FROM clauses)",
        default_factory=list,
    )
    query_type: str = Field(
        description="Type of SQL query (SELECT, INSERT, UPDATE, DELETE, etc.)"
    )
    has_joins: bool = Field(
        description="Whether the query contains JOIN operations", default=False
    )
    join_tables: list[str] = Field(
        description="Tables specifically referenced in JOIN clauses",
        default_factory=list,
    )
    qualified_mapping: dict[str, str] = Field(
        description="Mapping of clean table names to their fully qualified names",
        default_factory=dict,
    )
    # New fields for column extraction
    selected_columns: dict[str, list[str]] = Field(
        description="Mapping of table names to their selected columns (excluding aliases after AS)",
        default_factory=dict,
    )
    column_table_mapping: dict[str, str] = Field(
        description="Mapping of column names to their source tables",
        default_factory=dict,
    )
    confidence: float = Field(
        description="Confidence level (0.0-1.0) in the extraction accuracy",
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    extraction_notes: str = Field(
        description="Notes about the extraction process or any ambiguities found",
        default="",
    )


class SchemaValidationCore:
    """Core Strands-based schema validation implementation."""

    def __init__(
        self,
        model_id: str | None = None,
        aws_region: str | None = None,
        debug_mode: bool = False,
        session_id: str | None = None,
        rag_backend=None,
    ):
        """Initialize SchemaValidationCore with Strands Agent."""
        # Get configuration from centralized config
        config = get_agent_config("SchemaValidator", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or config["aws_region"]
        self.model_id = model_id or config["model_id"]
        self.debug_mode = debug_mode

        # Core validation components
        self.rag = rag_backend
        self.table_cache = {}
        self.schema_cache = {}
        self.cache_ttl = 300  # 5 minutes default
        self.prefer_local_validation = True

        # Create Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
            cache_prompt=config.get("cache_prompt"),
        )

        self._cache_prompt_type = config.get("cache_prompt")

        # Base instructions for SQL component extraction
        base_instructions = """You are an SQL component extraction specialist working with database schema validation.
        Extract table names, columns, and query structure from SQL queries with high accuracy.
        Return structured responses with table_names, selected_columns, and confidence fields."""

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        # Initialize Strands Agent for SQL component extraction
        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

        logger.info("✓ SchemaValidationCore (strand) initialized")

    @observe(as_type="generation")
    def extract_sql_components(self, sql: str) -> SQLComponentExtraction | None:
        """
        Extract SQL components using Strands Agent with structured analysis.

        Args:
            sql: SQL query string

        Returns:
            SQLComponentExtraction result or None if extraction fails

        """
        try:
            # Create comprehensive prompt for SQL component extraction
            prompt = SchemaValidatorPrompts.build_sql_component_extraction_prompt(sql)

            # Reset usage
            self._usage_container["last_usage"] = None

            # Call Strands Agent
            llm_result = self.agent.structured_output(SQLComponentExtraction, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "SchemaValidator",
                langfuse_context,
            )

            if self._cache_prompt_type:
                log_prompt_cache_status("SchemaValidator", self._usage_container)

            # Clean up table names while preserving mapping to original qualified names
            cleaned_tables = []
            qualified_mapping = {}

            for table in llm_result.table_names:
                original_qualified = table
                # Remove database/schema prefixes for validation
                clean_table = table.split(".")[-1]
                # Remove quotes and special characters
                clean_table = clean_table.strip("\"'`[]")
                # Skip empty names and common SQL keywords
                if clean_table and clean_table.upper() not in [
                    "SELECT",
                    "WHERE",
                    "GROUP",
                    "ORDER",
                    "HAVING",
                    "CTE",
                    "WITH",
                ]:
                    cleaned_tables.append(clean_table)
                    qualified_mapping[clean_table] = original_qualified

            # Update the result with cleaned table names and store mapping
            llm_result.table_names = list(set(cleaned_tables))
            llm_result.qualified_mapping = qualified_mapping

            logger.debug(
                f"Strand SQL extraction successful: {len(llm_result.table_names)} tables found"
            )
            return llm_result

        except Exception as e:
            logger.error(f"Strand SQL extraction failed: {e}")
            return None

    def validate_tables_exist(self, table_names: list[str]) -> dict[str, bool]:
        """
        Validate that all tables exist using local RAG first.

        Args:
            table_names: List of table names to validate

        Returns:
            Dictionary mapping table names to existence status

        """
        validation_results = {}

        for table in table_names:
            # Check cache first
            if table in self.table_cache:
                validation_results[table] = self.table_cache[table]
                logger.debug(f"Using cached validation for table: {table}")
                continue

            # Try local RAG validation first
            if self.prefer_local_validation and self.rag:
                exists = self._check_table_exists_in_rag(table)
                if exists is not None:  # RAG has definitive answer
                    self.table_cache[table] = exists
                    validation_results[table] = exists
                    logger.info(
                        f"Validated table {table} via RAG: {'EXISTS' if exists else 'NOT FOUND'}"
                    )
                    continue

            # No validation method available without schema context
            validation_results[table] = False
            logger.warning(
                f"Cannot validate table {table}: no RAG available and schema_context not provided"
            )

        return validation_results

    @observe(as_type="generation")
    def validate_sql_tables(
        self, sql: str, schema_context: str | None = None
    ) -> SchemaValidationResult:
        """
        Validate all table references and column references in a SQL query.

        Args:
            sql: SQL query to validate
            schema_context: Optional pre-fetched schema information from TableAgent

        Returns:
            SchemaValidationResult with validation details including column validation

        """
        # Extract SQL components using Strands agent
        sql_components = self.extract_sql_components(sql)

        # Extract table references
        if sql_components and sql_components.table_names:
            table_refs = sql_components.table_names
        else:
            # Fallback to regex extraction if Strands fails
            table_refs = self._extract_with_regex(sql)

        if not table_refs:
            logger.warning("No table references found in SQL")
            return SchemaValidationResult(
                valid_tables=[],
                invalid_tables=[],
                table_details={},
                validation_passed=True,  # No tables to validate
                column_validation=None,
                invalid_columns=None,
            )

        # Validate each table
        table_details = {}
        valid_tables = []
        invalid_tables = []

        logger.debug(f"🔍 Validating {len(table_refs)} tables: {table_refs}")

        for table in table_refs:
            logger.debug(f"🔍 Validating table: {table}")
            validation_result = self._validate_single_table(table)
            table_details[table] = validation_result

            logger.debug(
                f"Table '{table}': exists={validation_result.exists}, error={validation_result.error}"
            )

            if validation_result.exists:
                valid_tables.append(table)
                logger.debug(f"✅ Table '{table}' is valid")
            else:
                invalid_tables.append(table)
                logger.warning(
                    f"❌ Table '{table}' is invalid: {validation_result.error}"
                )

        # Validate columns if we have valid tables and column information
        column_validation = {}
        invalid_columns = []

        alias_columns = self._extract_select_aliases(sql)

        if valid_tables and sql_components and sql_components.selected_columns:
            logger.info("🔍 Validating columns from SQL query")

            for table_name, columns in sql_components.selected_columns.items():
                if table_name == "_unknown_":
                    # Skip unknown columns that couldn't be mapped
                    continue

                if table_name in valid_tables:
                    # Validate columns for this table using schema_context
                    column_results = self._validate_columns_for_table(
                        table_name, columns, schema_context, alias_columns
                    )
                    column_validation[table_name] = column_results

                    # Track invalid columns
                    for col_result in column_results:
                        if not col_result.exists:
                            invalid_columns.append(
                                f"{table_name}.{col_result.column_name}"
                            )
                            logger.warning(
                                f"❌ Column '{col_result.column_name}' not found in table '{table_name}'"
                            )

        validation_passed = len(invalid_tables) == 0 and len(invalid_columns) == 0

        if not validation_passed:
            if invalid_tables:
                logger.warning(
                    f"Schema validation FAILED. Invalid tables: {invalid_tables}"
                )
                for table in invalid_tables:
                    logger.warning(f"  - {table}: {table_details[table].error}")
            if invalid_columns:
                logger.warning(
                    f"Column validation FAILED. Invalid columns: {invalid_columns}"
                )
        else:
            logger.info("Schema validation PASSED. All tables and columns are valid")

        return SchemaValidationResult(
            valid_tables=valid_tables,
            invalid_tables=invalid_tables,
            table_details=table_details,
            validation_passed=validation_passed,
            column_validation=column_validation,
            invalid_columns=invalid_columns,
        )

    def _extract_with_regex(self, sql: str) -> list[str]:
        """Fallback regex-based table extraction from original implementation."""
        import re

        table_names = set()
        table_patterns = [
            r"FROM\s+(?:`|\")?([a-zA-Z_][\w.]*?)(?:`|\")?\s",
            r"FROM\s+([a-zA-Z_][\w.]*)",
            r"JOIN\s+(?:`|\")?([a-zA-Z_][\w.]*?)(?:`|\")?\s",
            r"JOIN\s+([a-zA-Z_][\w.]*)",
            r"UPDATE\s+(?:`|\")?([a-zA-Z_][\w.]*?)(?:`|\")?\s",
            r"UPDATE\s+([a-zA-Z_][\w.]*)",
            r"INSERT\s+INTO\s+(?:`|\")?([a-zA-Z_][\w.]*?)(?:`|\")?\s",
            r"INSERT\s+INTO\s+([a-zA-Z_][\w.]*)",
            r"DELETE\s+FROM\s+(?:`|\")?([a-zA-Z_][\w.]*?)(?:`|\")?\s",
            r"DELETE\s+FROM\s+([a-zA-Z_][\w.]*)",
        ]

        # Clean up the SQL for better pattern matching
        sql_clean = re.sub(r"--.*?\n", " ", sql)  # Remove comments
        sql_clean = re.sub(
            r"/\*.*?\*/", " ", sql_clean, flags=re.DOTALL
        )  # Remove block comments
        sql_clean = re.sub(r"\s+", " ", sql_clean)  # Normalize whitespace

        # Apply all table extraction patterns
        for pattern in table_patterns:
            matches = re.findall(pattern, sql_clean, re.IGNORECASE)
            for match in matches:
                # Clean up the table name
                table_name = match.strip()
                table_name = table_name.split()[0] if " " in table_name else table_name
                table_name = table_name.strip("\"'`[]")
                table_name = table_name.split(".")[-1]

                if table_name and table_name.upper() not in [
                    "SELECT",
                    "WHERE",
                    "GROUP",
                    "ORDER",
                    "HAVING",
                ]:
                    table_names.add(table_name)

        return list(table_names)

    def _validate_single_table(self, table_name: str) -> TableValidationResult:
        """
        Validate a single table and get its details using the hybrid validation approach.

        Args:
            table_name: Name of the table to validate

        Returns:
            TableValidationResult with validation details

        """
        logger.debug(f"🔍 Starting validation for table: {table_name}")

        try:
            # Try RAG validation first
            if self.prefer_local_validation and self.rag:
                logger.debug(f"🔍 Checking table '{table_name}' in RAG...")
                rag_result = self._check_table_exists_in_rag(table_name)
                logger.debug(f"RAG result for '{table_name}': {rag_result}")

                if rag_result is True:  # RAG says table exists
                    logger.debug(f"✅ RAG confirms table '{table_name}' exists")
                    return TableValidationResult(
                        table_name=table_name,
                        exists=True,
                        error=None,
                    )
                elif rag_result is False:  # RAG says table doesn't exist
                    logger.debug(f"❌ RAG says table '{table_name}' does not exist")
                    return TableValidationResult(
                        table_name=table_name,
                        exists=False,
                        error=f"Table '{table_name}' not found in RAG knowledge base",
                    )
                # If rag_result is None, we can't validate without schema_context
                logger.debug(
                    f"❌ Cannot validate table '{table_name}' - no definitive answer from RAG"
                )
                return TableValidationResult(
                    table_name=table_name,
                    exists=False,
                    error=f"Table '{table_name}' validation uncertain - no schema_context provided",
                )

        except Exception as e:
            logger.error(f"❌ Exception during validation of table '{table_name}': {e}")
            return TableValidationResult(
                table_name=table_name,
                exists=False,
                error=f"Validation failed: {str(e)}",
            )

    def _validate_columns_for_table(
        self,
        table_name: str,
        columns: list[str],
        schema_context: str | None = None,
        alias_columns: set | None = None,
    ) -> list[ColumnValidationResult]:
        """
        Validate that columns exist in the specified table using schema_context.

        Args:
            table_name: Name of the table
            columns: List of column names to validate
            schema_context: Pre-fetched schema information from TableAgent
            alias_columns: Set of column aliases to skip validation

        Returns:
            List of ColumnValidationResult for each column

        """
        results = []

        if not schema_context:
            # If no schema context provided, we can't validate columns
            logger.warning(
                f"No schema context provided for column validation of table '{table_name}'"
            )
            for column in columns:
                results.append(
                    ColumnValidationResult(
                        column_name=column,
                        table_name=table_name,
                        exists=False,
                        error="No schema context available for validation",
                    )
                )
            return results

        # Extract columns for this table from schema_context
        valid_columns = self._extract_columns_from_schema_context(
            table_name, schema_context
        )

        if not valid_columns:
            # Table not found in schema context
            logger.warning(f"Table '{table_name}' not found in schema context")
            for column in columns:
                results.append(
                    ColumnValidationResult(
                        column_name=column,
                        table_name=table_name,
                        exists=False,
                        error=f"Table '{table_name}' not found in schema context",
                    )
                )
            return results

        # Log available columns for debugging
        logger.debug(
            f"Available columns in '{table_name}': {valid_columns[:10]}..."
        )  # Show first 10

        # Validate each column
        for column in columns:
            column_lower = column.lower().strip()

            # Create a list of lowercased valid columns for exact matching
            valid_columns_lower = [col.lower().strip() for col in valid_columns]

            # Check for EXACT match (not substring match)
            if column_lower in valid_columns_lower:
                # Find the original cased version
                original_col = valid_columns[valid_columns_lower.index(column_lower)]
                results.append(
                    ColumnValidationResult(
                        column_name=column,
                        table_name=table_name,
                        exists=True,
                        error=None,
                    )
                )
                logger.debug(
                    f"✅ Column '{column}' exists in table '{table_name}' (matched: '{original_col}')"
                )
            elif alias_columns and column_lower in alias_columns:
                logger.debug(
                    f"🎭 Column '{column}' appears to be an alias; skipping strict validation"
                )
                results.append(
                    ColumnValidationResult(
                        column_name=column,
                        table_name=table_name,
                        exists=True,
                        error=None,
                    )
                )
            else:
                results.append(
                    ColumnValidationResult(
                        column_name=column,
                        table_name=table_name,
                        exists=False,
                        error=f"Column '{column}' not found in table '{table_name}'",
                    )
                )
                logger.debug(f"❌ Column '{column}' not found in table '{table_name}'")

                # Suggest similar column names if possible
                similar_columns = self._find_similar_columns(
                    column_lower, valid_columns
                )
                if similar_columns:
                    logger.info(
                        f"   Suggestion: Did you mean one of these columns? {similar_columns}"
                    )

        return results

    def _extract_select_aliases(self, sql: str) -> set:
        """Extract column aliases from the SELECT clause to avoid false column errors."""
        import re

        aliases: set = set()
        try:
            match = re.search(
                r"select\s+(.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL
            )
            if not match:
                return aliases

            select_clause = match.group(1)
            # Aliases using AS keyword
            for alias in re.findall(
                r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", select_clause, flags=re.IGNORECASE
            ):
                aliases.add(alias.lower())

            # Aliases without AS (e.g., "SUM(x) total")
            candidates = re.findall(r"[)\]]\s+([A-Za-z_][A-Za-z0-9_]*)", select_clause)
            sql_keywords = {
                "from",
                "where",
                "group",
                "order",
                "having",
                "limit",
                "and",
                "or",
                "when",
                "then",
                "else",
                "end",
            }
            for alias in candidates:
                lower_alias = alias.lower()
                if lower_alias not in sql_keywords:
                    aliases.add(lower_alias)

        except Exception as error:
            logger.debug(f"Alias extraction failed: {error}")

        return aliases

    def _extract_columns_from_schema_context(
        self, table_name: str, schema_context: str
    ) -> list[str]:
        """Extract column names for a specific table from cleaned schema context."""
        if not schema_context:
            return []

        try:
            columns = []
            lines = schema_context.split("\n")
            in_target_table = False

            for line in lines:
                line = line.strip()

                # Check if we're starting a new table section
                if line.startswith("Table Name:"):
                    current_table = line.replace("Table Name:", "").strip()
                    in_target_table = current_table.lower() == table_name.lower()

                # Extract columns for the target table
                elif in_target_table and line.startswith("*"):
                    # Format is: * column_name: type
                    # Extract just the column name
                    parts = line[1:].split(":")  # Remove the * and split by :
                    if parts:
                        column_name = parts[0].strip()
                        if column_name:
                            columns.append(column_name)

            return columns

        except Exception as e:
            logger.debug(f"Error extracting columns from schema context: {e}")
            return []

    def _find_similar_columns(
        self, column: str, valid_columns: list[str], threshold: float = 0.6
    ) -> list[str]:
        """Find similar column names using simple string similarity."""
        from difflib import SequenceMatcher

        similar = []
        for valid_col in valid_columns:
            ratio = SequenceMatcher(None, column, valid_col).ratio()
            if ratio >= threshold:
                similar.append(valid_col)

        return sorted(
            similar,
            key=lambda x: SequenceMatcher(None, column, x).ratio(),
            reverse=True,
        )[:3]

    def _check_table_exists_in_rag(self, table_name: str) -> bool | None:
        """Check if a table exists in the local RAG knowledge base using direct database lookup."""
        if not self.rag:
            return None

        try:
            # Method 1: Direct database lookup (most reliable)
            if self.rag and getattr(self.rag, "is_connected", lambda: False)():
                row_factory = dict_row if DICT_ROW_AVAILABLE else None
                with self.rag.cursor(row_factory=row_factory) as cur:
                    if DICT_ROW_AVAILABLE:
                        return self._check_table_with_cursor(cur, table_name)
                    return self._check_table_with_tuple_cursor(cur, table_name)

            # Method 2: Fallback to improved vector search (if direct lookup fails)
            search_results = self._improved_vector_search_for_table(table_name)
            if search_results is not None:
                return search_results

            # No schema information available, uncertain
            logger.debug(f"No schema information available for table: {table_name}")
            return None

        except Exception as e:
            logger.error(f"Error checking table in RAG: {e}")
            return None

    def _check_table_with_cursor(self, cur, table_name: str) -> bool | None:
        """Check table with dictionary cursor."""
        cur.execute(
            "SELECT full_content as content, metadata FROM rag_documents WHERE doc_type = 'schema'"
        )
        schema = cur.fetchall()
        # Check if table_name is in the content of the schema documents
        for table in schema:
            if table_name in table["content"]:
                return True
        return None

    def _check_table_with_tuple_cursor(self, cur, table_name: str) -> bool | None:
        """Check table with tuple cursor (fallback)."""
        # Check rag_schema_info table first (most direct)
        cur.execute(
            "SELECT table_name FROM rag_schema_info WHERE LOWER(table_name) = LOWER(%s)",
            [table_name],
        )
        schema_result = cur.fetchone()
        if schema_result:
            if isinstance(schema_result, Mapping):
                table_value = schema_result.get("table_name")
            else:
                table_value = schema_result[0] if len(schema_result) > 0 else None

            if table_value:
                logger.debug(f"Found table in schema_info: {table_name}")
                return True

        # Check rag_documents table for schema documents
        cur.execute(
            "SELECT full_content as content, metadata FROM rag_documents WHERE doc_type = 'schema'"
        )
        schema_docs = cur.fetchall()
        # Check if table_name is in the content of the schema documents
        for row in schema_docs:
            if isinstance(row, Mapping):
                content = row.get("content", "")
            else:
                content = row[0] if len(row) > 0 else ""

            if table_name in content:
                return True

        return None

    def _improved_vector_search_for_table(self, table_name: str) -> bool | None:
        """Improved vector search that tries multiple search strategies."""
        try:
            search_patterns = [
                table_name,  # Simple table name
                f"CREATE TABLE {table_name}",  # DDL pattern
                f"{table_name} (",  # Table with opening parenthesis
                f"TABLE {table_name}",  # Table keyword + name
            ]

            for pattern in search_patterns:
                search_results = self.rag.search_knowledge_base(
                    pattern, k=20
                )  # Increased k

                for result in search_results:
                    metadata = result.get("metadata", {})
                    content = result.get("content", "")

                    # Only check schema documents
                    if metadata.get("type") != "schema":
                        continue

                    # Check metadata table_name
                    stored_table_name = metadata.get("table_name", "")
                    if stored_table_name.lower() == table_name.lower():
                        logger.debug(
                            f"Vector search found table in metadata: {table_name}"
                        )
                        return True

                    # Check DDL content with multiple patterns
                    content_upper = content.upper()
                    table_upper = table_name.upper()

                    ddl_patterns = [
                        f"CREATE TABLE {table_upper}",
                        f"CREATE TABLE IF NOT EXISTS {table_upper}",
                        f"CREATE EXTERNAL TABLE {table_upper}",
                        f"{table_upper} (",
                        f"`{table_upper}`",
                        f'"{table_upper}"',
                        f"TABLE {table_upper}",
                    ]

                    for ddl_pattern in ddl_patterns:
                        if ddl_pattern in content_upper:
                            logger.debug(
                                f"Vector search found table in content: {table_name}"
                            )
                            return True

            # No matches found in vector search
            return None

        except Exception as e:
            logger.debug(f"Error in improved vector search: {e}")
            return None

    def clear_cache(self):
        """Clear all cached validation and schema data."""
        self.table_cache.clear()
        self.schema_cache.clear()
        logger.info("Cleared validation and schema caches")

    def _create_error_response(self, error_msg: str) -> dict[str, Any]:
        """Create standardized error response."""
        return {
            "error": error_msg,
            "validation_passed": False,
            "valid_tables": [],
            "invalid_tables": [],
        }


class SchemaValidatorAgent:
    """Compatibility wrapper maintaining original API."""

    def __init__(
        self,
        rag_backend=None,
        config: dict | None = None,
    ):
        """Initialize with Strands implementation."""
        self.config = config or {}

        requested_region = self.config.get("aws_region")
        requested_model = self.config.get("model_id")

        # Get model configuration from model_config.py
        agent_config = get_agent_config("SchemaValidator", requested_region)

        self.region = requested_region or agent_config["aws_region"]
        self.model_id = requested_model or agent_config["model_id"]

        # Initialize core Strands implementation
        self.core = SchemaValidationCore(
            model_id=self.model_id,
            aws_region=self.region,
            debug_mode=self.config.get("debug_mode", False),
            rag_backend=rag_backend,
        )

        # Maintain original attributes for backward compatibility
        self.rag = rag_backend
        self.table_cache = self.core.table_cache
        self.schema_cache = self.core.schema_cache
        self.cache_ttl = self.core.cache_ttl
        self.prefer_local_validation = self.core.prefer_local_validation

        logger.info("✓ SchemaValidatorAgent initialized")

    def validate_tables_exist(self, table_names: list[str]) -> dict[str, bool]:
        """Validate that all tables exist using local RAG first."""
        return self.core.validate_tables_exist(table_names)

    def get_available_tables(self, database: str | None = None) -> list[str]:
        """Get list of all available tables from RAG or analytics backend."""
        # Try RAG first
        if self.rag:
            try:
                rag_tables = self._get_tables_from_rag()
                if rag_tables:
                    logger.info(f"Found {len(rag_tables)} tables in RAG knowledge base")
                    return rag_tables
            except Exception as e:
                logger.debug(f"RAG table listing failed: {e}")

        # Fallback to analytics backend via AnalyticsConnector
        try:
            from src.config.database_config import ANALYTICS_DB_URL
            from src.connectors.analytics_backend import AnalyticsConnector

            connector = AnalyticsConnector(ANALYTICS_DB_URL)
            tables = connector.list_tables()
            logger.info(f"Found {len(tables)} tables via AnalyticsConnector")
            return tables
        except Exception as e:
            logger.error(f"Failed to get tables from analytics backend: {e}")

        return []

    def extract_table_references(self, sql: str) -> list[str]:
        """Extract table names from SQL using Strands agent with regex fallback."""
        sql_components = self.core.extract_sql_components(sql)
        if sql_components and sql_components.table_names:
            return sql_components.table_names
        else:
            # Fallback to regex extraction
            return self.core._extract_with_regex(sql)

    def get_enhanced_sql_analysis(self, sql: str) -> SQLComponentExtraction | None:
        """Get detailed SQL analysis using Strands agent for enhanced validation insights."""
        return self.core.extract_sql_components(sql)

    def validate_sql_tables(
        self, sql: str, schema_context: str | None = None
    ) -> SchemaValidationResult:
        """Validate all table references and column references in a SQL query."""
        return self.core.validate_sql_tables(sql, schema_context)

    def clear_cache(self):
        """Clear all cached validation and schema data."""
        self.core.clear_cache()

    def _get_tables_from_rag(self) -> list[str]:
        """Get all table names from the RAG knowledge base using direct database lookup."""
        if not self.rag:
            return []

        try:
            tables = set()

            # Method 1: Direct database lookup (most reliable)
            if self.rag and getattr(self.rag, "is_connected", lambda: False)():
                row_factory = dict_row if DICT_ROW_AVAILABLE else None
                with self.rag.cursor(row_factory=row_factory) as cur:
                    if DICT_ROW_AVAILABLE:
                        tables = self._get_tables_with_cursor(cur)
                    else:
                        tables = self._get_tables_with_tuple_cursor(cur)

                if tables:
                    logger.debug(
                        f"Direct lookup found {len(tables)} tables: {sorted(tables)}"
                    )
                    return sorted(list(tables))

            # Method 2: Fallback to vector search (if direct lookup fails)
            search_results = self.rag.search_knowledge_base(
                "schema DDL CREATE TABLE", k=100
            )

            for result in search_results:
                metadata = result.get("metadata", {})

                if metadata.get("type") == "schema":
                    # Get table name from metadata
                    table_name = metadata.get("table_name")
                    if table_name:
                        tables.add(table_name)

                    # Also parse table names from DDL content
                    content = result.get("content", "")
                    extracted_tables = self._extract_table_names_from_ddl(content)
                    tables.update(extracted_tables)

            return sorted(list(tables))

        except Exception as e:
            logger.error(f"Error getting tables from RAG: {e}")
            return []

    def _get_tables_with_cursor(self, cur) -> set:
        """Get tables with dictionary cursor."""
        tables = set()

        # Get tables from schema_info table
        cur.execute(
            "SELECT DISTINCT table_name FROM rag_schema_info WHERE table_name IS NOT NULL"
        )
        schema_info_tables = cur.fetchall()
        for row in schema_info_tables:
            if row["table_name"]:
                tables.add(row["table_name"])

        # Get tables from documents table
        cur.execute(
            """
            SELECT full_content as content, metadata FROM rag_documents
            WHERE doc_type = 'schema'
        """
        )
        schema_docs = cur.fetchall()

        for row in schema_docs:
            if isinstance(row, Mapping):
                content = row.get("content", "")
                metadata = row.get("metadata", {})
            else:
                content = row[0] if len(row) > 0 else ""
                metadata = row[1] if len(row) > 1 else {}

            if not isinstance(metadata, dict):
                metadata = {}

            # Get table name from metadata
            table_name = metadata.get("table_name")
            if table_name:
                tables.add(table_name)

            # Extract table names from DDL content
            extracted_tables = self._extract_table_names_from_ddl(content)
            tables.update(extracted_tables)

        return tables

    def _get_tables_with_tuple_cursor(self, cur) -> set:
        """Get tables with tuple cursor (fallback)."""
        tables = set()

        # Get tables from schema_info table
        cur.execute(
            "SELECT DISTINCT table_name FROM rag_schema_info WHERE table_name IS NOT NULL"
        )
        schema_info_tables = cur.fetchall()
        for row in schema_info_tables:
            table_name = row[0] if row else None  # Access by index
            if table_name:
                tables.add(table_name)

        # Get tables from documents table
        cur.execute(
            """
            SELECT full_content as content, metadata FROM rag_documents
            WHERE doc_type = 'schema'
        """
        )
        schema_docs = cur.fetchall()

        for row in schema_docs:
            content = row[0] if len(row) > 0 else ""  # First column: content
            metadata = row[1] if len(row) > 1 else {}  # Second column: metadata

            if not isinstance(metadata, dict):
                metadata = {}

            # Get table name from metadata
            table_name = metadata.get("table_name")
            if table_name:
                tables.add(table_name)

            # Extract table names from DDL content
            extracted_tables = self._extract_table_names_from_ddl(content)
            tables.update(extracted_tables)

        return tables

    def _extract_table_names_from_ddl(self, ddl_content: str) -> list[str]:
        """Extract table names from DDL content."""
        import re

        tables = []

        # Pattern to match CREATE TABLE statements
        create_table_patterns = [
            r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?([a-zA-Z_][\w.]*)",
            r"CREATE\s+(?:EXTERNAL\s+)?TABLE\s+(?:IF NOT EXISTS\s+)?([a-zA-Z_][\w.]*)",
        ]

        for pattern in create_table_patterns:
            matches = re.findall(pattern, ddl_content, re.IGNORECASE)
            for match in matches:
                # Clean up the table name
                table_name = match.strip().strip("`\"'")
                if table_name:
                    tables.append(table_name)

        return tables
