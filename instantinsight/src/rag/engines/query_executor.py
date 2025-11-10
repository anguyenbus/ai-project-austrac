"""
Query execution service with configurable backends.

Handles SQL execution with proper error handling and structured results.
"""

from typing import Protocol

import pandas as pd
from loguru import logger

from .types import ExecutionResult, ExecutionStatus


class QueryExecutor(Protocol):
    """Protocol for SQL query execution backends."""

    def execute(self, sql: str) -> ExecutionResult:
        """Execute SQL query and return structured result."""
        ...


class UniversalExecutor:
    """Universal query executor using Ibis framework."""

    def __init__(self, analytics_db_url: str):
        """
        Initialize UniversalExecutor with connection string.

        Args:
            analytics_db_url: Database URL (e.g., athena://..., postgres://...)

        Raises:
            ConnectionError: If connection fails

        """
        from src.connectors.analytics_backend import AnalyticsConnector

        self.analytics_db_url = analytics_db_url
        self.connector = AnalyticsConnector(analytics_db_url)
        logger.info(
            f"✅ UniversalExecutor initialized with {self.connector.backend_type} backend"
        )

    def execute(self, sql: str) -> ExecutionResult:
        """
        Execute SQL query using Ibis backend.

        Args:
            sql: SQL query to execute (read-only)

        Returns:
            ExecutionResult with data or error

        """
        try:
            logger.info(
                f"Executing SQL ({self.connector.backend_type}): {sql[:100]}..."
            )

            # NOTE: AnalyticsConnector.execute_query enforces read-only via icontract
            df = self.connector.execute_query(sql)

            logger.info(f"✅ Query executed successfully, returned {len(df)} rows")
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=df,
                rows_affected=len(df),
            )

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e),
            )

    def close(self):
        """Close Ibis connection."""
        if self.connector:
            self.connector.close()


# NOTE: AthenaExecutor class completely removed - replaced by UniversalExecutor


class TestExecutor:
    """Test executor for unit testing."""

    def __init__(self, mock_data: pd.DataFrame = None):
        """
        Initialize TestExecutor with optional mock data.

        Args:
            mock_data: Optional mock DataFrame to return for testing

        """
        self.mock_data = mock_data or pd.DataFrame()

    def execute(self, sql: str) -> ExecutionResult:
        """Return mock data for testing."""
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            data=self.mock_data,
            rows_affected=len(self.mock_data),
        )
