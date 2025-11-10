"""
Column mapping agent using Strand framework.

This agent maps filters to exact column names and values using LLM with structured outputs.
Handles high cardinality columns and complex database operations.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import boto3
import pandas as pd
import psycopg2
from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ...prompt_builders.prompts import Prompts
from ...prompt_builders.schema.column_mapper import ColumnMappingPrompts
from ..model_config import get_agent_config

# Import existing classes and utilities
from .column_helpers import (
    CategoricalValueRetriever,
    EmbeddingSearcher,
    FuzzyMatcher,
    NumericalColumnDetector,
)


# Define ColumnConfig for compatibility
class ColumnConfig:
    """Configuration for ColumnAgent."""

    def __init__(
        self,
        aws_region: str = None,
        model_id: str = None,
        embedding_model: str = None,
        max_distinct_values: int = None,
        enable_cardinality_cache: bool = True,
        postgres_config: dict[str, Any] | None = None,
        similarity_threshold: float = 0.2,
        top_k_similar: int = 20,
    ):
        """Initialize ColumnConfig with configuration parameters."""
        # Get model configuration from model_config.py
        agent_config = get_agent_config("SchemaColumnMapper", aws_region)

        # Load from model_config or fallback to parameters/environment
        self.aws_region = agent_config.get("aws_region", aws_region) or os.getenv(
            "AWS_REGION", "ap-southeast-2"
        )
        self.model_id = model_id or agent_config.get("model_id")
        self.temperature = agent_config.get("temperature", 0.1)
        self.max_tokens = agent_config.get("max_tokens", 3000)
        self.embedding_model = embedding_model or os.getenv(
            "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
        )
        self.max_distinct_values = max_distinct_values or int(
            os.getenv("MAX_DISTINCT_VALUES", "1000")
        )
        self.enable_cardinality_cache = enable_cardinality_cache
        self.postgres_config = postgres_config or {}
        self.similarity_threshold = similarity_threshold
        self.top_k_similar = top_k_similar


CARDINALITY_TABLE_NAME = os.getenv(
    "CARDINALITY_TABLE_NAME", "custom_analysers_cardinality_mapping"
)


class ColumnMapping(BaseModel):
    """Structured result for column mapping."""

    filterings: list[dict[str, Any]] = Field(
        description="List of column-value pairs. For text columns, use ONLY exact values from retrieved data (do NOT fabricate case variations). Example: [{'status': ['Commissioned']}, {'year': 2024}]"
    )
    mapping_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the column mappings"
    )
    analysis: str = Field(
        description="Brief explanation of how filters were mapped to columns"
    )


class CandidateColumns(BaseModel):
    """Structured result for candidate column identification."""

    candidates: dict[str, list[str]] = Field(
        description="Dictionary mapping table names to lists of candidate column names"
    )
    reasoning: str = Field(
        description="Explanation of the reasoning for the candidate columns"
    )


@dataclass
class ColumnMappingResult:
    """Result from column mapping agent - maintains backward compatibility."""

    filterings: list[dict[str, Any]]
    mapping_confidence: float
    analysis: str


class ColumnMapper:
    """Strand-based column mapper that maps filters to exact column names and values."""

    def __init__(
        self,
        aws_region: str = None,
        model_id: str = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the ColumnMapper.

        Args:
            aws_region: AWS region for Bedrock (used for config only)
            model_id: Bedrock model ID (optional, uses config default)
            debug_mode: Enable debug output

        """
        # Get configuration from centralized config
        agent_config = get_agent_config("SchemaColumnMapper", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]
        self.debug_mode = debug_mode

        # Create the Bedrock model (no aws_region parameter)
        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            streaming=False,
            cache_prompt=agent_config.get("cache_prompt"),
        )
        self._cache_prompt_type = agent_config.get("cache_prompt")

        # Initialize mapping and candidate agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the Strand agents for column mapping operations."""
        # Create callbacks for usage tracking
        mapping_callback, self._mapping_usage_container = create_usage_callback()
        candidate_callback, self._candidate_usage_container = create_usage_callback()

        # Agent for column mapping
        self.mapping_agent = Agent(
            model=self.model,
            system_prompt=Prompts.COLUMN_MAPPER_MAPPING,
            callback_handler=mapping_callback,
        )

        # Agent for candidate column identification
        self.candidate_agent = Agent(
            model=self.model,
            system_prompt=Prompts.COLUMN_MAPPER_CANDIDATE,
            callback_handler=candidate_callback,
        )

    def _log_prompt_cache_metrics(
        self, agent_label: str, usage_container: dict[str, Any]
    ) -> None:
        """Emit prompt cache diagnostics for the given column-mapper stage."""
        if not getattr(self, "_cache_prompt_type", None):
            logger.debug(
                f"{agent_label} prompt cache metrics skipped (cache_prompt disabled)"
            )
            return

        log_prompt_cache_status(agent_label, usage_container)

    def _extract_column_mapping(self, result) -> ColumnMapping | None:
        """Extract ColumnMapping from Strand result."""
        try:
            message = result.message
            content = message.get("content", [])

            if not content:
                return None

            # Look for JSON in content blocks
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text = block["text"]
                    # Try to parse JSON from text
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        return ColumnMapping(**data)

            return None
        except Exception as e:
            logger.warning(f"Failed to extract column mapping: {e}")
            return None

    def _extract_candidate_columns(self, result) -> CandidateColumns | None:
        """Extract CandidateColumns from Strand result."""
        try:
            message = result.message
            content = message.get("content", [])

            if not content:
                return None

            # Look for JSON in content blocks
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text = block["text"]
                    # Try to parse JSON from text
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        return CandidateColumns(**data)

            return None
        except Exception as e:
            logger.warning(f"Failed to extract candidate columns: {e}")
            return None

    @observe(as_type="generation")
    def map_columns(
        self,
        filters: list[dict[str, Any]],
        columns_info: dict[str, list[dict[str, str]]],
        categorical_mappings: dict[str, list[str]],
        tables: list[str],
    ) -> ColumnMappingResult:
        """
        Map filters to exact columns and values using Strand agent.

        Args:
            filters: Original filters from FilteringAgent
            columns_info: Column information for tables
            categorical_mappings: Distinct values for categorical columns
            tables: Selected table names

        Returns:
            ColumnMappingResult with mapped columns and metadata

        """
        if not filters:
            return ColumnMappingResult(
                filterings=[], mapping_confidence=1.0, analysis="No filters to map"
            )

        # Build prompt using existing prompt builder
        base_prompt = ColumnMappingPrompts.build_mapping_prompt(
            filters, columns_info, categorical_mappings, tables
        )

        # Add JSON schema request
        prompt = f"""{base_prompt}

Provide your response as a JSON object with these fields:
- filterings: List of column-value pairs as dictionaries
- mapping_confidence: Confidence score (0.0 to 1.0)
- analysis: Brief explanation of how filters were mapped
"""

        try:
            # Reset usage
            self._mapping_usage_container["last_usage"] = None

            # Call agent (using __call__, not .run())
            result = self.mapping_agent(prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._mapping_usage_container,
                self.model_id,
                "ColumnMapper",
                langfuse_context,
            )
            self._log_prompt_cache_metrics(
                "ColumnMapper", self._mapping_usage_container
            )

            # Extract structured output
            mapping = self._extract_column_mapping(result)

            if mapping and isinstance(mapping, ColumnMapping):
                result_obj = ColumnMappingResult(
                    filterings=mapping.filterings,
                    mapping_confidence=mapping.mapping_confidence,
                    analysis=mapping.analysis,
                )
                # Attach the LLM response for token tracking compatibility
                result_obj.llm_response = mapping
                return result_obj

            else:
                # Fallback: try to parse from text
                message = result.message
                content = message.get("content", [])
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_content += block["text"]

                return ColumnMappingResult(
                    filterings=filters,
                    mapping_confidence=0.5,
                    analysis=f"Response not in expected format. Raw: {text_content[:100]}",
                )

        except Exception as e:
            logger.error(f"Column mapping failed: {e}")
            return ColumnMappingResult(
                filterings=filters,
                mapping_confidence=0.5,
                analysis=f"Mapping error: {str(e)}",
            )

    @observe(as_type="generation")
    def identify_candidate_columns(
        self,
        filter_key: str,
        columns_info: dict[str, list[dict[str, str]]],
        schema_context: str | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        """
        Identify candidate columns using Strand agent.

        Args:
            filter_key: Filter key (e.g., 'document_description', 'city')
            columns_info: Column information for all tables
            schema_context: Optional schema DDL
            question: Question for context

        Returns:
            Dictionary with candidates and metadata

        """
        # Build prompt using existing prompt builder
        base_prompt = ColumnMappingPrompts.build_candidate_identification_prompt(
            filter_key, columns_info, schema_context, question
        )

        # Add JSON schema request
        prompt = f"""{base_prompt}

Provide your response as a JSON object with these fields:
- candidates: Dictionary mapping table names to lists of candidate column names
- reasoning: Explanation of the reasoning for the candidate columns
"""

        try:
            # Reset usage
            self._candidate_usage_container["last_usage"] = None

            # Call agent
            result = self.candidate_agent(prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._candidate_usage_container,
                self.model_id,
                "ColumnMapper_Candidate",
                langfuse_context,
            )
            self._log_prompt_cache_metrics(
                "ColumnMapper_Candidate", self._candidate_usage_container
            )

            # Extract structured output
            candidates = self._extract_candidate_columns(result)

            if candidates and isinstance(candidates, CandidateColumns):
                return {
                    "candidates": candidates.candidates,
                    "reasoning": candidates.reasoning,
                    "llm_response": candidates,  # For token tracking
                }

            else:
                # Fallback: return all columns
                fallback_candidates = {
                    table: [col["name"] for col in cols]
                    for table, cols in columns_info.items()
                }
                return {
                    "candidates": fallback_candidates,
                    "reasoning": "Response not in expected format - used fallback",
                    "llm_response": None,
                }

        except Exception as e:
            logger.error(f"Candidate identification failed: {e}")
            # Fallback: return all columns
            fallback_candidates = {
                table: [col["name"] for col in cols]
                for table, cols in columns_info.items()
            }
            return {
                "candidates": fallback_candidates,
                "reasoning": f"Agent error: {str(e)} - used fallback",
                "llm_response": None,
            }

    def get_config(self):
        """Get the current configuration."""
        return {
            "aws_region": self.aws_region,
            "model_id": self.model_id,
            "debug_mode": self.debug_mode,
        }

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        """
        for key, value in kwargs.items():
            if key == "model_id":
                self.model_id = value
                # Reinitialize agents with new model
                self._initialize_agents()
            elif key == "debug_mode":
                self.debug_mode = value


class ColumnAgent:
    """
    Column mapping agent using Strand framework.

    Maintains compatibility with existing code while using Strand under the hood.
    """

    def __init__(self, config: ColumnConfig | None = None, rag_instance=None):
        """
        Initialize the ColumnAgent.

        Args:
            config: Configuration object for the agent
            rag_instance: RAGEngine instance for database access

        """
        self.config = config or ColumnConfig()
        self.rag_engine = rag_instance
        self.candidate_columns = {}  # Will store filter_key -> list of candidate columns

        # Initialize the strand-based column mapper
        debug_mode = getattr(self.config, "debug_mode", True)

        self.column_mapper = ColumnMapper(
            aws_region=self.config.aws_region,
            model_id=self.config.model_id,
            debug_mode=debug_mode,
        )

        # Initialize database connections
        self._init_connections()

        # Initialize helper classes
        self.embedding_searcher = EmbeddingSearcher(
            self.pg_connection, self.bedrock_client, self.config
        )
        self.fuzzy_matcher = FuzzyMatcher()
        self.numerical_detector = NumericalColumnDetector()
        self.categorical_retriever = CategoricalValueRetriever(self)
        # Backward compatibility alias
        self.vanna = self.rag_engine

        # Compatibility attributes
        self.llm = None  # For compatibility
        self.instructor_client = self.column_mapper.mapping_agent  # For compatibility

        logger.info("ColumnAgent (strand) initialized")

    def _init_connections(self):
        """Initialize database connections."""
        self.pg_connection = None
        self.bedrock_client = None

        try:
            # PostgreSQL connection for RAG database
            if self.config.postgres_config:
                self.pg_connection = psycopg2.connect(
                    host=self.config.postgres_config.get("host", "localhost"),
                    port=self.config.postgres_config.get("port", 5432),
                    database=self.config.postgres_config.get("database", "pgvector"),
                    user=self.config.postgres_config.get("user", "admin"),
                    password=self.config.postgres_config.get("password", "admin"),
                )
                logger.debug("PostgreSQL connection established")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL: {e}")

        try:
            # Bedrock client for embeddings
            self.bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.config.aws_region
            )
            logger.debug("Bedrock client initialized for embeddings")
        except Exception as e:
            logger.warning(f"Could not initialize Bedrock client: {e}")

    def map_columns(
        self,
        filtering_result: dict[str, Any],
        selected_tables: list[str],
        schema_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Map filtering results to exact column names and values.

        Args:
            filtering_result: Output from FilteringAgent
            selected_tables: Tables selected by TableAgent
            schema_context: Optional schema information for the tables

        Returns:
            Dictionary with precise column-value mappings

        """
        logger.info(f"Mapping columns for {len(selected_tables)} tables")

        if not filtering_result.get("filterings"):
            return {"filterings": [], "analysis": "No filters to map"}

        # Extract column information from schema
        columns_info = self._extract_columns_from_tables(
            selected_tables, schema_context
        )

        # Separate filters that need categorical value retrieval from those that don't
        filters_needing_values = []
        filters_without_values = []

        for filter_dict in filtering_result["filterings"]:
            # Skip categorical retrieval for limit and not_null filters
            if "limit" in filter_dict or "not_null" in filter_dict:
                filters_without_values.append(filter_dict)
            else:
                filters_needing_values.append(filter_dict)

        # Only get categorical values for filters that need them
        categorical_mappings = {}
        if filters_needing_values:
            categorical_mappings = self._get_categorical_values(
                filters_needing_values,
                columns_info,
                selected_tables,
                schema_context,
                filtering_result["question"],
            )
        else:
            logger.info(
                "No filters requiring categorical value retrieval (only limit/not_null filters present)"
            )

        # Use strand-based LLM to map filters to exact columns and values
        mapped_result = self._llm_map_columns(
            filtering_result["filterings"],
            columns_info,
            categorical_mappings,
            selected_tables,
        )

        return mapped_result

    def _extract_columns_from_tables(
        self, tables: list[str], schema_context: str | None = None
    ) -> dict[str, list[dict[str, str]]]:
        """
        Extract column information from selected tables.

        Args:
            tables: List of table names
            schema_context: Optional schema DDL

        Returns:
            Dictionary mapping table names to column info

        """
        columns_info = {}

        for table in tables:
            columns = []

            # Try to get from schema context first
            if schema_context:
                columns = self._parse_columns_from_ddl(schema_context, table)

            columns_info[table] = columns

        return columns_info

    def _parse_columns_from_ddl(
        self, schema_ddl: str, table_name: str
    ) -> list[dict[str, str]]:
        """Parse columns from DDL schema."""
        columns = []

        # Find the table definition in the format "Table Name: table_name"
        table_pattern = rf"Table Name:\s+{re.escape(table_name)}\s*\n"
        match = re.search(table_pattern, schema_ddl, re.IGNORECASE)

        if match:
            # Get the text starting from after the table name
            start_pos = match.end()
            remaining_text = schema_ddl[start_pos:]

            # Find the end of this table's columns (next "Table Name:" or end of string)
            next_table_match = re.search(
                r"\nTable Name:", remaining_text, re.IGNORECASE
            )
            if next_table_match:
                columns_text = remaining_text[: next_table_match.start()]
            else:
                columns_text = remaining_text

            # Parse individual columns in format "* column name: type"
            column_pattern = r"^\*\s+([^:]+):\s+(\w+)"
            for line in columns_text.split("\n"):
                col_match = re.match(column_pattern, line.strip())
                if col_match:
                    column_name = col_match.group(1).strip()
                    column_type = col_match.group(2).strip()

                    columns.append({"name": column_name, "type": column_type})

        return columns

    def _get_categorical_values(
        self,
        filters: list[dict[str, Any]],
        columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
        schema_context: str | None = None,
        question: str | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """
        Get categorical values following 3-step process.

        1. Embedding search in rag_cardinality
        2. DB cardinality table lookup
        3. Direct SELECT DISTINCT query

        Args:
            filters: List of filter dictionaries
            columns_info: Column information
            tables: Selected table names
            schema_context: Optional schema DDL
            question: Question

        Returns:
            Dictionary mapping filter keys to column-value mappings
            e.g., {"city": {"city/town": ["Brisbane", ...], "location": [...]}, ...}

        """
        categorical_mappings = {}

        # Process each categorical filter
        for filter_dict in filters:
            for key, value in filter_dict.items():
                # Skip numeric filters
                if key in ["limit", "year", "month", "quarter"] or isinstance(
                    value, int | float
                ):
                    continue

                # Skip if likely target columns are numeric
                if self._should_skip_categorical_retrieval(key, columns_info):
                    logger.info(
                        f"Skipping categorical retrieval for '{key}' - likely numeric column"
                    )
                    continue

                # Handle both single values and lists of values
                if isinstance(value, str):
                    # Single string value
                    column_values_map = self.categorical_retriever.retrieve_values(
                        key, value, columns_info, tables, schema_context, question
                    )
                    if column_values_map:
                        categorical_mappings[key] = column_values_map
                elif isinstance(value, list):
                    # Multiple values - aggregate results from all values
                    aggregated_column_values = {}
                    for val in value:
                        if isinstance(val, str):
                            column_values_map = (
                                self.categorical_retriever.retrieve_values(
                                    key,
                                    val,
                                    columns_info,
                                    tables,
                                    schema_context,
                                    question,
                                )
                            )
                            # Merge results for this filter key
                            for col_name, col_values in column_values_map.items():
                                if col_name not in aggregated_column_values:
                                    aggregated_column_values[col_name] = []
                                # Add values, avoiding duplicates
                                for cv in col_values:
                                    if cv not in aggregated_column_values[col_name]:
                                        aggregated_column_values[col_name].append(cv)

                    if aggregated_column_values:
                        categorical_mappings[key] = aggregated_column_values

        return categorical_mappings

    def _should_skip_categorical_retrieval(
        self, filter_key: str, columns_info: dict[str, list[dict[str, str]]]
    ) -> bool:
        """
        Check if categorical value retrieval should be skipped for a filter key.

        This checks if the likely target columns for this filter are numeric,
        in which case we should use operators instead of categorical lists.

        Args:
            filter_key: The filter key (e.g., 'units_in_stock', 'price')
            columns_info: Column information for all tables

        Returns:
            True if categorical retrieval should be skipped

        """
        # Check if filter key matches any column names that are numeric
        for _table, cols in columns_info.items():
            for col_info in cols:
                col_name = col_info.get("name", "")
                col_type = col_info.get("type", "")

                # Check for exact or partial name match
                if (
                    filter_key.lower() in col_name.lower()
                    or col_name.lower() in filter_key.lower()
                ):
                    # Check if this column is numeric
                    if self.numerical_detector.is_numerical(col_name, col_type):
                        logger.debug(
                            f"Found numeric column '{col_name}' ({col_type}) matching filter '{filter_key}'"
                        )
                        return True

        # Check for common numeric filter patterns
        numeric_patterns = [
            "stock",
            "quantity",
            "count",
            "amount",
            "price",
            "cost",
            "units",
            "number",
            "total",
            "sum",
            "avg",
            "average",
            "min",
            "max",
            "score",
            "rating",
            "level",
            "rank",
        ]

        filter_key_lower = filter_key.lower()
        for pattern in numeric_patterns:
            if pattern in filter_key_lower:
                logger.debug(
                    f"Filter key '{filter_key}' matches numeric pattern '{pattern}'"
                )
                return True

        return False

    def _identify_candidate_columns(
        self,
        filter_key: str,
        columns_info: dict[str, list[dict[str, str]]],
        schema_context: str | None = None,
        question: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Use strand-based LLM to identify candidate columns.

        Args:
            filter_key: Filter key (e.g., 'document_description', 'city')
            columns_info: Column information for all tables
            schema_context: Optional schema DDL
            question: Question

        Returns:
            Dictionary mapping table names to list of candidate column names

        """
        try:
            result = self.column_mapper.identify_candidate_columns(
                filter_key, columns_info, schema_context, question
            )

            # Cache the result
            self.candidate_columns[filter_key] = result["candidates"]

            logger.info(
                f"LLM identified candidate columns for '{filter_key}': {result['candidates']}"
            )

            # Return format expected by existing code
            return result

        except Exception as e:
            logger.error(f"Error identifying candidate columns for '{filter_key}': {e}")
            # Fallback: return all columns
            candidates = {
                table: [col["name"] for col in cols]
                for table, cols in columns_info.items()
            }
            return {
                "candidates": candidates,
                "llm_response": None,  # No LLM response in fallback case
            }

    def _filter_columns_info_by_candidates(
        self,
        columns_info: dict[str, list[dict[str, str]]],
        candidate_columns: dict[str, list[str]],
    ) -> dict[str, list[dict[str, str]]]:
        """
        Filter columns_info to only include candidate columns that are categorical.

        Args:
            columns_info: Original column information for all tables
            candidate_columns: Dictionary mapping table names to candidate column names

        Returns:
            Filtered columns_info containing only categorical candidate columns

        """
        filtered_columns_info = {}

        for table, cols in columns_info.items():
            if table in candidate_columns:
                candidate_names = set(candidate_columns[table])
                filtered_cols = []

                for col_info in cols:
                    if col_info["name"] in candidate_names:
                        # Check if column is likely numerical - if so, skip it
                        col_type = col_info.get("type", "")
                        if not self.numerical_detector.is_numerical(
                            col_info["name"], col_type
                        ):
                            filtered_cols.append(col_info)
                        else:
                            logger.info(
                                f"Skipping numerical column '{col_info['name']}' from categorical value retrieval"
                            )

                if filtered_cols:
                    filtered_columns_info[table] = filtered_cols
                    logger.debug(
                        f"Table {table}: filtered from {len(cols)} to {len(filtered_cols)} categorical columns"
                    )
                else:
                    logger.debug(
                        f"Table {table}: no categorical candidate columns found"
                    )
            else:
                # Table not in candidates, filter for categorical columns only
                filtered_cols = []
                for col_info in cols:
                    col_type = col_info.get("type", "")
                    if not self.numerical_detector.is_numerical(
                        col_info["name"], col_type
                    ):
                        filtered_cols.append(col_info)
                if filtered_cols:
                    filtered_columns_info[table] = filtered_cols

        return filtered_columns_info

    def _check_column_cardinality(self, table: str, column: str) -> list[str]:
        """Check column cardinality table."""
        unique_value_column_name = "unique_value"
        try:
            query = f"""
            SELECT {unique_value_column_name}
            FROM {CARDINALITY_TABLE_NAME}
            WHERE LOWER(table_name) = LOWER('{table}') AND LOWER(column_name) = LOWER('{column}')
            """
            result = None
            if self.rag_engine:
                try:
                    result = self.rag_engine.execute_sql(query)
                except Exception as rag_error:
                    logger.debug(
                        f"RAG engine cardinality query failed for {table}.{column}: {rag_error}"
                    )

            if result is not None and not result.empty:
                # Extract all values from the column into a list
                raw_values = result[unique_value_column_name].dropna().tolist()
                # Flatten any JSON arrays and collect all values
                all_values = []
                for value in raw_values:
                    if isinstance(value, str) and value.startswith("["):
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, list):
                                all_values.extend(str(v) for v in parsed if v)
                            else:
                                all_values.append(str(value))
                        except (json.JSONDecodeError, TypeError):
                            all_values.append(str(value))
                    elif value:
                        all_values.append(str(value))

                # Remove duplicates while preserving order using pandas
                unique_values = pd.Series(all_values).drop_duplicates().tolist()

                logger.debug(
                    f"Found {len(unique_values)} unique values from cardinality table for {table}.{column}"
                )
                # Return all unique values up to max limit for IN clause support
                return unique_values[: self.config.max_distinct_values]
        except Exception as e:
            logger.debug(f"Error checking cardinality table: {e}")

        return []

    def _query_distinct_values(
        self, table: str, column: str, filter_value: str = None
    ) -> list[str]:
        """Query distinct values directly from database."""
        query = f"""
        SELECT DISTINCT "{column}" AS value
        FROM {table}
        WHERE "{column}" IS NOT NULL
        LIMIT {self.config.max_distinct_values}
        """

        result = None
        if self.rag_engine:
            try:
                result = self.rag_engine.execute_sql(query)
            except Exception as rag_error:
                logger.debug(
                    f"RAG engine distinct query failed for {table}.{column}: {rag_error}"
                )

        if result is None or result.empty:
            logger.debug(f"No distinct values retrieved for {table}.{column}")
            return []

        values = [str(v) for v in result["value"].dropna().tolist() if v]
        unique_values = list(dict.fromkeys(values))

        if filter_value:
            logger.debug(
                f"Retrieved {len(unique_values)} values for fuzzy matching '{filter_value}' from {table}.{column}"
            )
        else:
            logger.debug(
                f"Retrieved {len(unique_values)} distinct values from {table}.{column}"
            )

        return unique_values[: self.config.max_distinct_values]

    def _llm_map_columns(
        self,
        filters: list[dict[str, Any]],
        columns_info: dict[str, list[dict[str, str]]],
        categorical_mappings: dict[str, list[str]],
        tables: list[str],
    ) -> dict[str, Any]:
        """
        Use strand-based LLM to map filters to exact columns and values.

        Args:
            filters: Original filters from FilteringAgent
            columns_info: Column information for tables
            categorical_mappings: Distinct values for categorical columns
            tables: Selected table names

        Returns:
            Mapped column-value pairs

        """
        result = self.column_mapper.map_columns(
            filters, columns_info, categorical_mappings, tables
        )

        # Convert to expected return format
        return {
            "filterings": result.filterings,
            "mapping_confidence": result.mapping_confidence,
            "analysis": result.analysis,
            "llm_response": getattr(
                result, "llm_response", None
            ),  # Include for token tracking
        }

    def __del__(self):
        """Clean up database connections."""
        if self.pg_connection:
            self.pg_connection.close()

    def get_config(self) -> ColumnConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")


# Global agent instance
_column_agent = None


def get_column_agent(
    config: ColumnConfig | None = None, rag_instance=None
) -> ColumnAgent:
    """Get the global column agent instance."""
    global _column_agent
    if _column_agent is None:
        _column_agent = ColumnAgent(config, rag_instance)
    return _column_agent


def map_columns_with_llm(
    filtering_result: dict[str, Any],
    selected_tables: list[str],
    schema_context: str | None = None,
    config: ColumnConfig | None = None,
    rag_instance=None,
) -> dict[str, Any]:
    """Backward compatibility function for column mapping."""
    agent = get_column_agent(config, rag_instance)
    return agent.map_columns(filtering_result, selected_tables, schema_context)
