"""
Schema Vectorizer Orchestrator.

Coordinates all vectorization components for clean separation of concerns.
This orchestrator replaces the monolithic AthenaSchemaVectorizer methods.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from .athena_ddl_generator import AthenaDDLGenerator
from .sql_example_generator import SQLExampleGenerator
from .yaml_data_exporter import YAMLDataExporter


class SchemaVectorizerOrchestrator:
    """
    Orchestrates the vectorization process using modular components.

    This class coordinates:
    - DDL generation for schema documentation
    - SQL example generation and validation
    - YAML export of training data
    - RAG system integration
    """

    def __init__(
        self,
        rag_system=None,
        analytics_backend=None,
        training_data_dir: Path | None = None,
    ):
        """
        Initialize the orchestrator with all required components.

        Args:
            rag_system: RAG system for storing vectorized data
            analytics_backend: AnalyticsConnector for SQL validation (preferred)
            training_data_dir: Directory for YAML training data

        """
        self.rag_system = rag_system
        self.analytics_backend = analytics_backend

        # Initialize modular components
        self.ddl_generator = AthenaDDLGenerator()
        # Use AnalyticsConnector for SQL validation
        self.sql_generator = SQLExampleGenerator(analytics_backend)
        self.yaml_exporter = YAMLDataExporter(training_data_dir)

        # Track validated examples for export
        self.validated_examples_for_export = []

        logger.info(
            "🚀 SchemaVectorizerOrchestrator initialized with modular components"
        )

    def process_database_schema(
        self, schema_data: dict[str, Any], generate_examples: bool = False
    ) -> bool:
        """
        Process complete database schema data.

        Args:
            schema_data: Database schema metadata
            generate_examples: Whether to generate SQL examples

        Returns:
            True if processing succeeded

        """
        try:
            database_name = schema_data["database_name"]
            logger.info(f"🔄 Processing database schema: {database_name}")

            # Create database documentation (basic DDL only for now)
            tables_processed = 0
            for _table_name, table_data in schema_data.get("tables", {}).items():
                if self.process_table_schema(table_data, generate_examples):
                    tables_processed += 1

            logger.info(
                f"✅ Processed {tables_processed} tables for database {database_name}"
            )

            # Export validated examples if any were generated
            if self.validated_examples_for_export:
                self.yaml_exporter.export_all_validated_examples(
                    self.validated_examples_for_export
                )

            return True

        except Exception as e:
            logger.error(f"Failed to process database schema: {e}")
            return False

    def process_table_schema(
        self, table_data: dict[str, Any], generate_examples: bool = False
    ) -> bool:
        """
        Process individual table schema data.

        Args:
            table_data: Table schema metadata
            generate_examples: Whether to generate SQL examples

        Returns:
            True if processing succeeded

        """
        try:
            table_name = table_data["table_name"]
            database_name = table_data["database_name"]

            logger.info(f"📊 Processing table: {database_name}.{table_name}")

            # Generate DDL
            table_ddl = self.ddl_generator.generate_ddl(table_data)

            # Add DDL to RAG system
            if self.rag_system:
                self._add_to_rag_system(
                    table_ddl, "schema", f"{database_name}.{table_name}"
                )

            # Generate SQL examples if requested
            if generate_examples:
                try:
                    sql_examples = self.sql_generator.generate_validated_examples(
                        table_data
                    )

                    if sql_examples:
                        # Add to RAG system
                        for example in sql_examples:
                            if self.rag_system:
                                self._add_sql_example_to_rag(example)

                        # Export to YAML
                        self.yaml_exporter.export_table_examples(
                            sql_examples, database_name, table_name
                        )

                        # Track for bulk export
                        self.validated_examples_for_export.extend(sql_examples)

                        logger.info(
                            f"✅ Generated {len(sql_examples)} validated examples for {table_name}"
                        )
                    else:
                        logger.warning(f"No valid examples generated for {table_name}")

                except Exception as e:
                    logger.error(f"SQL example generation failed for {table_name}: {e}")

            return True

        except Exception as e:
            logger.error(
                f"Failed to process table {table_data.get('table_name', 'unknown')}: {e}"
            )
            return False

    def load_existing_training_data(self) -> list[dict[str, str]]:
        """
        Load existing training data from YAML files.

        Returns:
            List of existing SQL examples

        """
        return self.yaml_exporter.load_existing_examples()

    def _create_basic_database_ddl(self, schema_data: dict[str, Any]) -> str:
        """Create basic database DDL documentation."""
        database_name = schema_data["database_name"]

        ddl = f"""-- AWS Athena Database: {database_name}
-- Description: {schema_data.get("description", "No description available")}
-- Location: {schema_data.get("location_uri", "Not specified")}
-- Total Tables: {schema_data.get("table_count", 0)}
-- Extracted: {schema_data.get("extracted_at", "Unknown")}

-- CRITICAL: ALL table references MUST use format: awsdatacatalog.{database_name}.table_name

-- Available Tables:"""

        for table_name in schema_data.get("tables", {}).keys():
            full_table_name = f"awsdatacatalog.{database_name}.{table_name}"
            ddl += f"\n-- - {full_table_name}"

        return ddl

    def _add_to_rag_system(self, content: str, doc_type: str, source: str):
        """Add content to RAG system using clean public interface."""
        logger.info(f"🔄 Adding {doc_type} to RAG system: {source}")
        logger.debug(f"Content preview: {content[:100]}...")

        try:
            # Call specific methods directly instead of generic router
            if doc_type == "documentation":
                result = self.rag_system.add_documentation_text(content)
            elif doc_type == "schema":
                result = self.rag_system.add_schema(content)
            else:
                logger.error(f"Unsupported doc_type: {doc_type}")
                return False

            if result:
                logger.info(f"✅ Successfully added {doc_type}: {source}")
                return True
            else:
                logger.error(f"❌ Failed to add {doc_type}: {source}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to add {doc_type} to RAG system: {e}")
            return False

    def _add_sql_example_to_rag(self, example: dict[str, str]):
        """Add SQL example to RAG system using public interface."""
        try:
            # Use the clean public interface
            result = self.rag_system.add_training_example(
                question=example["question"], sql=example["sql"]
            )

            if result:
                logger.debug(f"✅ Added SQL example: {example['question'][:50]}...")
            else:
                logger.warning(
                    f"⚠️ Failed to add SQL example: {example['question'][:50]}..."
                )

        except Exception as e:
            logger.error(f"Failed to add SQL example to RAG system: {e}")

    def get_validated_examples_count(self) -> int:
        """Get count of validated examples ready for export."""
        return len(self.validated_examples_for_export)

    def export_all_validated_examples(self) -> Path | None:
        """Export all validated examples to YAML."""
        if not self.validated_examples_for_export:
            logger.warning("No validated examples to export")
            return None

        return self.yaml_exporter.export_all_validated_examples(
            self.validated_examples_for_export
        )
