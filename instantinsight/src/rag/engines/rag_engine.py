"""
RAGEngine using factory pattern and dependency injection.

Clean interface to SQL generation and execution with separated concerns.
"""

import time
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.utils.langfuse_client import observe

from .engine_factory import EngineFactory
from .query_executor import QueryExecutor
from .sql_engine import SQLEngine


class RAGEngine:
    """
    Clean RAGEngine with separated concerns.

    Responsibilities:
    - Provide unified interface to SQL generation and execution
    - Delegate to specialized components
    - Maintain backward compatibility

    Does NOT handle:
    - Configuration loading (EngineFactory)
    - Agent initialization (EngineFactory)
    - Direct SQL generation (SQLEngine)
    - Query execution (QueryExecutor)
    """

    # Global cached instance for singleton pattern
    _instance = None

    def __init__(self, sql_engine: SQLEngine, query_executor: QueryExecutor):
        """
        Initialize with injected dependencies.

        Args:
            sql_engine: Configured SQL generation engine
            query_executor: Configured query executor

        """
        self.sql_engine = sql_engine
        self.query_executor = query_executor
        self.is_initialized = True

    @classmethod
    def create_instance(
        cls, force_new: bool = False, test_mode: bool = False
    ) -> tuple[Optional["RAGEngine"], str | None]:
        """
        Create a configured engine instance.

        Args:
            force_new: Force creation of new instance instead of cached
            test_mode: Use test executor

        Returns:
            Tuple of (RAGEngine, error_message)

        """
        # Return cached instance if available and not forcing new
        if not force_new and not test_mode and cls._instance is not None:
            return cls._instance, None

        try:
            # Create SQL engine
            sql_engine, sql_error = EngineFactory.create_sql_engine()
            if not sql_engine:
                return None, sql_error

            # Create query executor
            executor, exec_error = EngineFactory.create_query_executor(
                test_mode=test_mode
            )
            if not executor:
                return None, exec_error

            # Create engine instance
            engine = cls(sql_engine=sql_engine, query_executor=executor)

            # Cache instance only if not in test mode
            if not test_mode:
                cls._instance = engine

            logger.info("✅ RAGEngine created successfully")
            return engine, None

        except Exception as e:
            error_msg = f"Failed to create RAGEngine: {e}"
            logger.error(error_msg)
            return None, error_msg

    @observe(name="rag_generate_sql")
    def generate_sql(
        self,
        question: str,
        return_context: bool = False,
        prior_turns: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> Any:
        """
        Generate SQL using the SQL engine.

        Args:
            question: Natural language question
            return_context: If True, returns dict with context
            prior_turns: Optional list of prior conversation turns
            **kwargs: Additional parameters (for compatibility)

        Returns:
            SQL string or dict with context based on return_context

        """
        start_time = time.time()

        try:
            logger.info(f"🔄 Generating SQL for: {question[:50]}...")

            # Generate SQL using injected engine
            result = self.sql_engine.generate_sql(question, prior_turns=prior_turns)
            total_time = (time.time() - start_time) * 1000

            # Handle different result types
            if result.success:
                sql = result.sql
                logger.info(f"✅ SQL generated in {total_time:.1f}ms: {sql[:100]}...")

                if return_context:
                    return {
                        "sql": sql,
                        "schema_context": result.schema_context or "",
                        "normalized_query": result.normalized_query,
                    }
                return sql

            elif result.needs_clarification:
                clarification_sql = (
                    f"CANNOT FIND TABLES: {result.clarification_message}"
                )
                logger.info(f"⚠️ Clarification needed: {result.clarification_message}")

                if return_context:
                    return {
                        "sql": clarification_sql,
                        "schema_context": "",
                        "normalized_query": result.normalized_query,
                    }
                return clarification_sql

            else:
                # Error case
                error_msg = result.error or "Unknown SQL generation error"
                logger.error(f"❌ SQL generation failed: {error_msg}")
                raise RuntimeError(f"Failed to generate SQL: {error_msg}")

        except Exception as e:
            logger.error(f"Error in generate_sql: {e}")
            raise RuntimeError(f"Failed to generate SQL: {e}") from e

    @observe(name="rag_execute_sql")
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query using the query executor.

        Args:
            sql: SQL query to execute

        Returns:
            Query results as DataFrame

        Raises:
            Exception: If query execution fails

        """
        try:
            logger.info(f"Executing SQL: {sql[:100]}...")

            # Execute using injected executor
            result = self.query_executor.execute(sql)

            if result.success:
                logger.info(
                    f"✅ Query executed successfully, returned {result.rows_affected} rows"
                )
                return result.data
            else:
                error_msg = result.error or "Unknown execution error"
                logger.error(f"❌ Query execution failed: {error_msg}")
                # Raise exception so Pipeline can catch and trigger refinement
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            # Re-raise to allow Pipeline to handle error and trigger refinement
            raise

    def get_component_status(self) -> dict[str, Any]:
        """Get component status for monitoring."""
        return {
            "initialized": self.is_initialized,
            "sql_engine": self.sql_engine is not None,
            "query_executor": self.query_executor is not None,
        }

    def close(self):
        """Clean up resources."""
        try:
            # Clean up SQL engine resources
            if (
                hasattr(self.sql_engine, "rag_instance")
                and self.sql_engine.rag_instance
            ):
                if hasattr(self.sql_engine.rag_instance, "close"):
                    self.sql_engine.rag_instance.close()

            logger.info("✅ RAGEngine closed successfully")
        except Exception as e:
            logger.error(f"Error closing RAGEngine: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Compatibility function
def create_rag_engine(**kwargs) -> RAGEngine:
    """
    Create a RAGEngine instance (legacy compatibility).

    Note: kwargs are ignored as configuration is handled by factory.
    """
    engine, error = RAGEngine.create_instance()
    if not engine:
        raise RuntimeError(f"Failed to create engine: {error}")
    return engine
