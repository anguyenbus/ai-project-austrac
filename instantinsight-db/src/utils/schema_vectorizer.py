"""
Database Schema Processing Pipeline.

This module processes database schema data for RAG training.
It integrates with the existing PgVectorRAG system to add schema knowledge
from any supported database backend (Athena, PostgreSQL, Snowflake, etc.).

Key Features:
- Modular component architecture
- DDL generation and schema documentation
- SQL example generation and validation
- YAML export for training data
- Template-based LLM prompts
- Universal database support via Ibis
"""

from pathlib import Path
from typing import Any

from loguru import logger

from ..rag.rag_engine import RAGEngine
from .error_handler import ErrorHandler
from .vectorizer import SchemaVectorizerOrchestrator

# NOTE: Legacy AthenaConnectionManager removed - use AnalyticsConnector for all database operations
ATHENA_LEGACY_AVAILABLE = False


class SchemaVectorizer:
    """
    Universal schema processing pipeline using modular components.

    This class handles the process of converting database schema metadata
    into RAG training data using clean, modular components. Supports any
    database backend via Ibis.
    """

    def __init__(self, rag_config: dict[str, Any], db_config: dict[str, Any]):
        """
        Initialize the schema processor.

        Args:
            rag_config: RAG system configuration
            db_config: Database connection configuration (legacy Athena config for backward compatibility)

        Note:
            The db_config parameter supports legacy Athena configuration format
            but the actual database connection is now handled via ANALYTICS_DB_URL.

        """
        self.rag_config = rag_config
        self.db_config = db_config  # Renamed from athena_config
        self.error_handler = ErrorHandler()

        # Initialize RAG system
        from ..rag.components.engine_config import EngineConfig

        engine_config = EngineConfig()
        self.rag_system = RAGEngine(config=engine_config)

        logger.info("✓ RAG system initialized with modular components")

        # Legacy connection manager removed - SQL validation now handled by AnalyticsConnector
        self.legacy_manager = None
        logger.debug(
            "SQL validation handled by AnalyticsConnector (legacy manager removed)"
        )

        # Initialize the modular orchestrator
        training_data_dir = Path(__file__).parent.parent / "training" / "data"
        self.orchestrator = SchemaVectorizerOrchestrator(
            rag_system=self.rag_system,
            llm_client=getattr(self, "llm", None),
            training_data_dir=training_data_dir,
        )

        logger.info("✓ Schema processor initialized with modular architecture")

    def vectorize_database_schema(
        self,
        schema_data: dict[str, Any],
        update_mode: str = "incremental",
        generate_examples: bool = False,
    ) -> bool:
        """
        Process complete database schema for RAG training.

        Args:
            schema_data: Extracted database schema data
            update_mode: Update strategy (unused in new architecture)
            generate_examples: Whether to generate SQL examples (default: False)

        Returns:
            bool: True if successful, False otherwise

        """
        database_name = schema_data["database_name"]
        logger.info(
            f"🚀 Processing database schema: {database_name} "
            f"(modular architecture, generate_examples={generate_examples})"
        )

        # Load existing training data first
        existing_examples = self.orchestrator.load_existing_training_data()
        logger.info(f"📖 Loaded {len(existing_examples)} existing training examples")

        # Use the orchestrator for all processing
        success = self.orchestrator.process_database_schema(
            schema_data, generate_examples=generate_examples
        )

        if success:
            logger.info(f"🎉 Successfully processed database: {database_name}")
        else:
            logger.error(f"❌ Failed to process database: {database_name}")

        return success

    def vectorize_table_data(
        self, table_data: dict[str, Any], generate_examples: bool = False
    ) -> bool:
        """
        Process individual table schema data.

        Args:
            table_data: Table metadata dictionary
            generate_examples: Whether to generate SQL examples

        Returns:
            True if processing succeeded

        """
        return self.orchestrator.process_table_schema(table_data, generate_examples)

    def load_existing_sql_examples(self) -> list[dict[str, str]]:
        """
        Load existing SQL examples from training data files.

        Returns:
            List of existing SQL examples

        """
        return self.orchestrator.load_existing_training_data()

    def _load_existing_training_examples(self) -> int:
        """
        Load existing SQL examples from YAML files and store them in RAG database.

        Returns:
            Number of examples loaded and stored

        """
        try:
            # Load examples from YAML files
            existing_examples = self.orchestrator.load_existing_training_data()

            if not existing_examples:
                logger.warning("⚠️ No existing SQL examples found in training/data")
                return 0

            logger.info(
                f"📥 Storing {len(existing_examples)} existing SQL examples in RAG database..."
            )

            # Store each example in the RAG database
            stored_count = 0
            for example in existing_examples:
                try:
                    # Use the orchestrator's method to add SQL examples
                    if self._add_sql_example_to_rag(example):
                        stored_count += 1
                except Exception as e:
                    logger.error(f"Failed to store example: {e}")
                    continue

            logger.info(
                f"✅ Successfully stored {stored_count}/{len(existing_examples)} SQL examples in RAG database"
            )
            return stored_count

        except Exception as e:
            logger.error(f"Failed to load existing training examples: {e}")
            return 0

    def _add_sql_example_to_rag(self, example: dict[str, str]) -> bool:
        """Add SQL example to RAG system using public interface."""
        try:
            # Use the clean public interface
            result = self.rag_system.add_training_example(
                question=example["question"], sql=example["sql"]
            )

            if result:
                logger.debug(f"✅ Stored SQL example: {example['question'][:50]}...")
                return True
            else:
                logger.error("❌ Failed to add SQL example")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to add SQL example to RAG: {e}")
            return False

    def export_validated_examples_to_yaml(self):
        """Export all validated examples to YAML files."""
        examples_count = self.orchestrator.get_validated_examples_count()

        if examples_count == 0:
            logger.warning("No validated examples to export")
            return

        output_file = self.orchestrator.export_all_validated_examples()

        if output_file:
            logger.info(
                f"💾 Successfully exported {examples_count} examples to {output_file}"
            )
        else:
            logger.error("Failed to export validated examples")

    def get_validated_examples_count(self) -> int:
        """Get count of validated examples."""
        return self.orchestrator.get_validated_examples_count()

    def cleanup(self):
        """Clean up resources."""
        # The modular components handle their own cleanup
        if self.legacy_manager:
            self.legacy_manager.close()
        logger.info("✓ Schema processor cleanup completed")

    # Properties for backward compatibility
    @property
    def validated_examples_for_export(self) -> list[dict[str, str]]:
        """Get validated examples (backward compatibility)."""
        return self.orchestrator.validated_examples_for_export

    @validated_examples_for_export.setter
    def validated_examples_for_export(self, value: list[dict[str, str]]):
        """Set validated examples (backward compatibility)."""
        self.orchestrator.validated_examples_for_export = value

    # Legacy method stubs for backward compatibility
    def _add_documentation_modern(self, documentation: str) -> bool:
        """Legacy method - functionality moved to orchestrator."""
        logger.debug("Using legacy _add_documentation_modern method")
        try:
            self.orchestrator._add_to_rag_system(
                documentation, "documentation", "legacy"
            )
            return True
        except Exception:
            return False

    def update_schema_incrementally(self, new_schema_data: dict[str, Any]) -> bool:
        """Update schema data incrementally."""
        return self.vectorize_database_schema(
            new_schema_data, update_mode="incremental"
        )

    def get_vectorization_stats(self) -> dict[str, Any]:
        """Get statistics about processed schema data."""
        try:
            stats = self.rag_system.get_statistics()
            return {
                "total_documents": stats.get("total_documents", 0),
                "total_sql_examples": stats.get("document_types", {})
                .get("sql_example", {})
                .get("count", 0),
                "validated_examples": len(self.validated_examples_for_export),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Backward compatibility alias
AthenaSchemaVectorizer = SchemaVectorizer


# Utility functions for integration
def vectorize_database_schemas(
    schema_files: list[str], rag_config: dict[str, Any], db_config: dict[str, Any]
) -> bool:
    """
    Process multiple schema files.

    Args:
        schema_files: List of schema JSON file paths
        rag_config: RAG system configuration
        db_config: Database configuration

    Returns:
        bool: True if all schemas processed successfully

    """
    processor = SchemaVectorizer(rag_config, db_config)

    try:
        success_count = 0

        for schema_file in schema_files:
            try:
                import json

                with open(schema_file) as f:
                    schema_data = json.load(f)

                if isinstance(schema_data, dict) and "database_name" in schema_data:
                    success = processor.vectorize_database_schema(schema_data)
                    if success:
                        success_count += 1
                        logger.info(f"✓ Processed schema from {schema_file}")
                    else:
                        logger.error(f"✗ Failed to process schema from {schema_file}")
                elif isinstance(schema_data, dict):
                    # Handle multiple databases in one file
                    for db_schema in schema_data.values():
                        if isinstance(db_schema, dict) and "database_name" in db_schema:
                            success = processor.vectorize_database_schema(db_schema)
                            if success:
                                success_count += 1

            except Exception as e:
                logger.error(f"Failed to process schema file {schema_file}: {e}")

        logger.info(
            f"Successfully processed {success_count}/{len(schema_files)} schema files"
        )
        return success_count == len(schema_files)

    finally:
        processor.cleanup()


# Backward compatibility function alias
vectorize_athena_schemas = vectorize_database_schemas
