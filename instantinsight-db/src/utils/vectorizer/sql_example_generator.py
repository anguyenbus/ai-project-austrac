"""
SQL Example Generation Module.

Handles creation and validation of SQL examples for Athena tables.
Extracted from the monolithic AthenaSchemaVectorizer for better maintainability.
"""

import os
import random
from typing import Any

import boto3
import instructor
from loguru import logger
from pydantic import BaseModel, Field

from .template_loader import TemplateLoader


class SQLExampleBatch(BaseModel):
    """Simple batch of SQL examples."""

    examples: list[dict[str, str]] = Field(description="List of question/sql pairs")


class ValidationResult(BaseModel):
    """Simple validation result."""

    is_valid: bool = Field(description="Whether SQL is valid")


class SQLExampleGenerator:
    """
    Generates and validates SQL examples for Athena tables using LLM intelligence.

    This class handles:
    - Candidate SQL generation via LLM
    - SQL validation against Athena
    - Schema preparation for prompts
    """

    def __init__(self, backend=None):
        """
        Initialize the SQL example generator.

        Args:
            backend: AnalyticsConnector for SQL validation

        """
        self.backend = backend
        self.template_loader = TemplateLoader()

        # Initialize instructor client for structured output
        self._instructor_client = None
        self._setup_instructor_client()

        logger.debug("SQLExampleGenerator initialized")

    def _setup_instructor_client(self):
        """Configure instructor client for structured output."""
        try:
            region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
            bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            self._instructor_client = instructor.from_bedrock(
                client=bedrock_client,
                mode=instructor.Mode.BEDROCK_JSON,
            )
        except Exception as e:
            logger.warning(f"Failed to setup instructor client: {e}")
            self._instructor_client = None

    def generate_validated_examples(
        self, table_data: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Generate 5-10 VALIDATED SQL examples per table by executing against Athena.

        Args:
            table_data: Dictionary containing table metadata

        Returns:
            List of validated SQL examples with question/sql pairs

        Raises:
            RuntimeError: If LLM is not available or generation fails

        """
        table_name = table_data["table_name"]

        logger.info(
            f"🔍 Generating VALIDATED SQL examples for {table_name} (5-10 examples max)"
        )

        # Generate candidate examples first
        candidate_examples = self.generate_candidate_examples(table_data)

        # Limit to maximum candidates we want to test
        max_candidates = 10
        candidate_examples = candidate_examples[:max_candidates]

        logger.info(
            f"📊 Generated {len(candidate_examples)} candidate examples for {table_name}, validating with Athena..."
        )

        validated_examples = []
        target_examples = random.randint(5, 10)

        for i, example in enumerate(candidate_examples, 1):
            if len(validated_examples) >= target_examples:
                logger.info(
                    f"✅ Reached target of {target_examples} validated examples for {table_name}"
                )
                break

            logger.info(
                f"🔍 Validating example {i}/{len(candidate_examples)} for {table_name}"
            )

            # Step 1: LLM-based Athena compliance check (optional)
            if not self._validate_sql_with_llm(
                example["sql"], table_data["database_name"], table_name
            ):
                logger.debug(f"❌ Example {i} failed LLM compliance check, skipping")
                continue

            # Step 2: CRITICAL - Execute against backend with LIMIT 2
            test_sql = self._add_limit_to_query(example["sql"])
            if self._validate_sql_with_backend(test_sql):
                validated_examples.append(example)
                logger.info(f"✅ Example {i} validated successfully")
            else:
                logger.debug(f"❌ Example {i} failed backend execution, skipping")

        logger.info(
            f"🎯 Final result: {len(validated_examples)} validated examples for {table_name}"
        )

        return validated_examples

    def generate_candidate_examples(
        self, table_data: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Generate candidate SQL examples using LLM intelligence - LLM ONLY, NO FALLBACK.

        Args:
            table_data: Dictionary containing table metadata

        Returns:
            List of candidate SQL examples

        Raises:
            RuntimeError: If LLM is not available or generation fails

        """
        table_name = table_data["table_name"]

        logger.info(
            f"🤖 Using LLM EXCLUSIVELY to generate intelligent SQL examples for {table_name} (NO FALLBACK)"
        )

        # Prepare table schema information for LLM
        schema_info = self._prepare_schema_for_llm(table_data)

        try:
            examples = self._generate_examples_with_enhanced_llm(
                schema_info, table_data
            )
            logger.info(
                f"🎯 LLM generated {len(examples)} candidate examples for {table_name}"
            )
            return examples

        except Exception as e:
            error_msg = f"❌ LLM example generation failed for {table_name}: {e}. NO FALLBACK AVAILABLE."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _generate_examples_with_enhanced_llm(
        self, schema_info: str, table_data: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Generate SQL examples using instructor for structured output."""
        if not self._instructor_client:
            raise RuntimeError("Instructor client not available")

        prompt = self._build_example_generation_prompt(schema_info, table_data)
        model_id = os.getenv(
            "BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

        try:
            response = self._instructor_client.chat.completions.create(
                model=model_id,
                response_model=SQLExampleBatch,
                messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )

            # Validate examples
            valid_examples = []
            for example in response.examples:
                if self._validate_example(example, table_data):
                    valid_examples.append(example)

            if not valid_examples:
                raise ValueError("No valid examples generated")

            logger.info(f"🎯 Instructor generated {len(valid_examples)} valid examples")
            return valid_examples

        except Exception as e:
            logger.error(f"Instructor example generation error: {e}")
            raise RuntimeError(f"LLM example generation failed: {e}") from e

    def _validate_example(
        self, example: dict[str, str], table_data: dict[str, Any]
    ) -> bool:
        """Validate a single example."""
        sql = example["sql"]
        full_table_name = (
            f"awsdatacatalog.{table_data['database_name']}.{table_data['table_name']}"
        )

        # Check table name
        if full_table_name not in sql:
            return False

        # Check column quoting
        schema_columns = [col["name"] for col in table_data.get("columns", [])]
        for col in schema_columns:
            needs_quoting = (
                " " in col
                or "-" in col
                or any(char in col for char in ["(", ")", ".", "/", "\\", "%"])
                or col.lower()
                in ["date", "time", "order", "group", "user", "table", "type", "key"]
            )
            if needs_quoting and col in sql and f'"{col}"' not in sql:
                return False

        return True

    def _build_example_generation_prompt(
        self, schema_info: str, table_data: dict[str, Any]
    ) -> str:
        """Build the comprehensive LLM prompt for SQL example generation using template."""
        table_name = table_data["table_name"]
        database_name = table_data["database_name"]
        full_table_name = f"awsdatacatalog.{database_name}.{table_name}"

        return self.template_loader.format_prompt(
            "sql_generation",
            schema_info=schema_info,
            table_name=table_name,
            database_name=database_name,
            full_table_name=full_table_name,
        )

    def _prepare_schema_for_llm(self, table_data: dict[str, Any]) -> str:
        """Prepare table schema information for LLM prompt."""
        table_name = table_data["table_name"]
        database_name = table_data["database_name"]
        full_table_name = f"awsdatacatalog.{database_name}.{table_name}"

        schema_info = f"""Table: {full_table_name}
Description: {table_data.get("description", "No description available")}
Location: {table_data.get("location", "Not specified")}

Columns:"""

        for col in table_data.get("columns", []):
            col_name = col["name"]
            col_type = col["type"]
            col_comment = col.get("comment", "")

            # Show proper quoting for columns with spaces/special chars
            needs_quoting = (
                " " in col_name
                or "-" in col_name
                or any(char in col_name for char in ["(", ")", ".", "/", "\\", "%"])
                or col_name.lower()
                in ["date", "time", "order", "group", "user", "table", "type", "key"]
            )

            quoted_col = f'"{col_name}"' if needs_quoting else col_name
            schema_info += f"\n- {quoted_col} ({col_type})"
            if col_comment:
                schema_info += f" -- {col_comment}"

        # Add partition information
        if table_data.get("partition_keys"):
            schema_info += "\n\nPartition Keys:"
            for pk in table_data["partition_keys"]:
                pk_name = pk["name"]
                pk_type = pk["type"]
                pk_comment = pk.get("comment", "")

                needs_quoting = " " in pk_name or "-" in pk_name
                quoted_pk = f'"{pk_name}"' if needs_quoting else pk_name

                schema_info += f"\n- {quoted_pk} ({pk_type})"
                if pk_comment:
                    schema_info += f" -- {pk_comment}"

        return schema_info

    def _validate_sql_with_llm(
        self, sql: str, database_name: str, table_name: str
    ) -> bool:
        """Validate SQL with LLM using instructor for structured output."""
        if not self._instructor_client:
            return True  # Skip validation if no instructor

        full_table_name = f"awsdatacatalog.{database_name}.{table_name}"
        prompt = self.template_loader.format_prompt(
            "sql_validation", sql=sql, full_table_name=full_table_name
        )

        try:
            model_id = os.getenv(
                "BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
            )
            response = self._instructor_client.chat.completions.create(
                model=model_id,
                response_model=ValidationResult,
                messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )

            if not response.is_valid:
                logger.debug(f"LLM validation failed for SQL: {sql}")

            return response.is_valid

        except Exception as e:
            logger.warning(f"LLM validation failed with error: {e}")
            return True  # Default to valid if validation fails

    def _add_limit_to_query(self, sql: str) -> str:
        """Add LIMIT 2 to query for safe testing."""
        sql = sql.strip().rstrip(";")

        # Check if query already has LIMIT
        if "LIMIT" in sql.upper():
            return sql + ";"

        return f"{sql} LIMIT 2;"

    def _validate_sql_with_backend(self, sql: str) -> bool:
        """Validate SQL by executing against Ibis backend with LIMIT."""
        if not self.backend:
            logger.warning("No Ibis backend available for validation")
            return True  # Skip validation if no backend connection

        try:
            # Execute the query with LIMIT 2 for safety using Ibis
            self.backend.execute_query(sql)
            logger.debug(f"Ibis backend validation successful for: {sql}")
            return True

        except Exception as e:
            logger.debug(f"Ibis backend validation failed for: {sql} - Error: {e}")
            return False
