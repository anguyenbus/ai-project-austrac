"""Modern RAGEngine with clean, modular component architecture."""

import os
from typing import Any

import pandas as pd
from loguru import logger

from src.connectors.analytics_backend import AnalyticsConnector

# Import modular components
from .components.database_manager import DatabaseManager
from .components.engine_config import EngineConfig
from .components.error_handling_service import ErrorHandlingConfig, ErrorHandlingService
from .components.sql_generation_service import SQLGenerationConfig, SQLGenerationService


class RAGEngine:
    """
    Modern RAGEngine with clean, modular component architecture.

    This class provides a clean, maintainable implementation using specialized
    components for different aspects of functionality. Focused on SQL generation
    for AWS Athena with advanced agent-based pipeline.
    """

    def __init__(self, config: EngineConfig | None = None):
        """
        Initialize the modular RAGEngine.

        Args:
            config: Engine configuration object

        Raises:
            RuntimeError: If RAG instance cannot be properly initialized

        """
        # Create default config if not provided
        self.config = config or EngineConfig()
        self.is_initialized = False

        # Initialize components first
        self._initialize_components()

        # Initialize RAG instance immediately - fail fast if not possible
        try:
            from src.config.database_config import DATABASE_URL
            from src.rag.pgvector_rag import PgvectorRAG

            logger.info("Creating PgvectorRAG instance...")
            self._rag_instance = PgvectorRAG(connection_string=DATABASE_URL)

            # Connect to database - critical operation
            if not self._rag_instance.connect_to_database():
                raise RuntimeError("Failed to connect PgvectorRAG to database")

            # Initialize embeddings - critical operation
            if not self._rag_instance.initialize_embeddings():
                self._rag_instance.close()
                raise RuntimeError("Failed to initialize PgvectorRAG embeddings")

            logger.info("✅ PgvectorRAG instance created successfully")

        except ImportError as e:
            raise RuntimeError(f"Missing required dependencies: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG engine: {e}") from e

        # Set initialized flag only after successful setup
        self.is_initialized = True
        logger.info("✅ RAGEngine initialized with clean modular architecture")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _initialize_components(self):
        """Initialize all modular components."""
        try:
            # Error handling (initialized first)
            self.error_handler = ErrorHandlingService(ErrorHandlingConfig())

            # Core components
            self.database_manager = DatabaseManager(self.config.database)

            # SQL generation
            self.sql_generation_service = SQLGenerationService(
                rag_engine=self, config=SQLGenerationConfig()
            )

            # Training statistics
            self.training_stats = {
                "total_items_added": 0,
                "question_sql_pairs": 0,
                "ddl_statements": 0,
                "documentation_items": 0,
                "failed_additions": 0,
            }

            logger.info("✅ All modular components initialized")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Continue with partial initialization

    def train(  # noqa: PLR0913
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: str = None,
        **kwargs,
    ):
        """Add training data to the system."""
        try:
            if question and sql:
                return self.add_training_example(question, sql, **kwargs)
            elif ddl:
                return self.add_schema(ddl)
            elif documentation:
                return self.add_documentation_text(documentation)
            return False
        except Exception as e:
            logger.error(f"Error in train method: {e}")
            return False

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL query on analytics database."""
        try:
            logger.info(f"Executing SQL on analytics database: {sql[:100]}...")

            # Get analytics DB URL
            analytics_db_url = os.getenv("ANALYTICS_DB_URL")
            if not analytics_db_url:
                raise ValueError("ANALYTICS_DB_URL not configured")

            # Initialize analytics connector
            backend = AnalyticsConnector(analytics_db_url)

            # Execute SQL query
            df = backend.execute_query(sql)

            logger.info(f"✅ Query executed successfully, returned {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            # Re-raise to allow Pipeline to handle error and trigger refinement
            raise

    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, "database_manager"):
                self.database_manager.close_all()

            # Close cached RAG instance
            if self._rag_instance is not None:
                if hasattr(self._rag_instance, "close"):
                    self._rag_instance.close()
                self._rag_instance = None

            logger.info("✅ RAGEngine closed successfully")
        except Exception as e:
            logger.error(f"Error closing RAGEngine: {e}")

    # Modern API Methods
    def add_training_example(self, question: str, sql: str, **kwargs) -> bool:
        """Add a single question-SQL training pair directly to RAG."""
        try:
            if hasattr(self._rag_instance, "add_document"):
                import json

                # Extract tables from SQL if not provided
                tables = kwargs.get("tables", [])
                if not tables and sql:
                    # Use the _extract_tables_from_sql method from RAG instance
                    if hasattr(self._rag_instance, "_extract_tables_from_sql"):
                        tables = self._rag_instance._extract_tables_from_sql(sql)
                        logger.debug(f"Extracted tables from SQL: {tables}")

                # Build metadata for the document
                # Always include tables key (empty list if no tables found)
                metadata = {
                    "question": question,
                    "tables": tables,  # Always include, even if empty
                }
                if kwargs:
                    for key in ["explanation", "tag"]:
                        if key in kwargs and kwargs[key]:
                            metadata[key] = kwargs[key]

                # Format content as JSON (same format as add_sql_examples uses)
                content = json.dumps(
                    {
                        "question": question,
                        "sql": sql,
                        "tables": tables,  # Include extracted tables in JSON content
                    },
                    ensure_ascii=False,
                )

                logger.debug(f"Adding training example to RAG: {question[:50]}...")
                result = self._rag_instance.add_document(
                    content=content,
                    doc_type="sql_example",
                    source="training",
                    metadata=metadata,
                )

                # Check if document was added successfully
                success = result and "document_id" in result and not result.get("error")
                if success:
                    self.training_stats["total_items_added"] += 1
                    self.training_stats["question_sql_pairs"] += 1
                    logger.debug(
                        f"✅ Successfully added training example to RAG (doc_id: {result.get('document_id')})"
                    )
                else:
                    logger.warning(
                        f"❌ Failed to add training example to RAG: {result.get('error', 'Unknown error')}"
                    )
                return success
            else:
                logger.error(
                    "❌ RAG instance not available or missing add_document method"
                )
                return False
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            self.training_stats["failed_additions"] += 1
            return False

    def add_documentation_text(self, documentation: str) -> bool:
        """Add documentation text directly to RAG."""
        try:
            if hasattr(self._rag_instance, "add_document"):
                success = self._rag_instance.add_document(
                    content=documentation, doc_type="documentation", source="manual"
                )
                if success:
                    self.training_stats["total_items_added"] += 1
                    self.training_stats["documentation_items"] += 1
                return success
            return False
        except Exception as e:
            logger.error(f"Error adding documentation: {e}")
            self.training_stats["failed_additions"] += 1
            return False

    def add_schema(
        self, ddl_text: str, table_info: dict[str, Any] | None = None
    ) -> bool:
        """Add DDL schema directly without circular calls."""
        try:
            # Store in RAG instance for persistence
            if hasattr(self._rag_instance, "add_sql_schema"):
                success = self._rag_instance.add_sql_schema(ddl_text)

                # 2. Update stats directly
                if success:
                    self.training_stats["total_items_added"] += 1
                    self.training_stats["ddl_statements"] += 1

                return success
            return False
        except Exception as e:
            logger.error(f"Error adding schema: {e}")
            return False

    def get_training_data(self) -> pd.DataFrame:
        """Get training data directly from RAG instance."""
        try:
            # Get data directly from RAG instance
            if hasattr(self._rag_instance, "get_training_data"):
                data = self._rag_instance.get_training_data()
                if isinstance(data, list):
                    # Convert list of dicts to DataFrame
                    return pd.DataFrame(data)
                elif isinstance(data, pd.DataFrame):
                    return data
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()

    def get_statistics(self) -> dict[str, Any]:
        """Get RAG statistics."""
        try:
            return {
                "total_documents": self.training_stats.get("documentation_items", 0),
                "total_sql_examples": self.training_stats.get("question_sql_pairs", 0),
                "total_ddl_items": self.training_stats.get("ddl_statements", 0),
                "total_items": self.training_stats.get("total_items_added", 0),
                "failed_additions": self.training_stats.get("failed_additions", 0),
                "document_types": {
                    "question_sql": self.training_stats.get("question_sql_pairs", 0),
                    "ddl": self.training_stats.get("ddl_statements", 0),
                    "documentation": self.training_stats.get("documentation_items", 0),
                },
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    # Additional modern methods for complete API coverage
    def add_sql_examples(self, examples: list[dict]) -> bool:
        """Add multiple SQL examples at once."""
        try:
            success_count = 0
            for example in examples:
                if (
                    isinstance(example, dict)
                    and "question" in example
                    and "sql" in example
                ):
                    if self.add_training_example(
                        question=example["question"],
                        sql=example["sql"],
                        explanation=example.get("explanation", ""),
                        tag="bulk_examples",
                    ):
                        success_count += 1

            logger.info(f"Added {success_count}/{len(examples)} SQL examples")
            return success_count > 0
        except Exception as e:
            logger.error(f"Error adding SQL examples: {e}")
            return False

    def add_sql_schema(
        self, schema=None, schema_text=None, table_info=None, **kwargs
    ) -> bool:
        """Add SQL schema with multiple input formats (for script compatibility)."""
        ddl_text = schema or schema_text
        if not ddl_text:
            logger.warning("No schema provided")
            return False
        return self.add_schema(ddl_text, table_info=table_info)

    def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if self._rag_instance and hasattr(self._rag_instance, "close"):
                self._rag_instance.close()
                self._rag_instance = None

            # Reset training stats
            self.training_stats = {
                "total_items_added": 0,
                "question_sql_pairs": 0,
                "ddl_statements": 0,
                "documentation_items": 0,
                "failed_additions": 0,
            }

            logger.info("✅ RAG engine cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function
def create_rag_engine(**kwargs) -> RAGEngine:
    """Create a RAGEngine instance."""
    return RAGEngine(**kwargs)
