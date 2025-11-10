"""
Universal database backend using Ibis Framework.

Supports: Athena, PostgreSQL, Snowflake, BigQuery, Redshift, etc.
"""

from typing import Any, Final

import ibis
import pandas as pd
import sqlglot
from beartype import beartype
from loguru import logger


@beartype
class AnalyticsConnector:
    """
    Universal database connector for analytics databases.

    This class provides a unified interface for connecting to and querying
    multiple database backends (Athena, PostgreSQL, Databricks, Snowflake,
    BigQuery, etc.) through the Ibis framework.

    Attributes:
        connection_string: Database URL in standard format (ANALYTICS_DB_URL)
        backend_type: Detected backend (athena, postgres, snowflake, databricks, etc.)
        _conn: Active Ibis connection

    Examples:
        >>> backend = AnalyticsConnector("athena://awsdatacatalog?region=ap-southeast-2&database=mydb")
        >>> backend = AnalyticsConnector("postgres://user:pass@host:5432/database")
        >>> backend = AnalyticsConnector("snowflake://account/database?warehouse=compute")
        >>> backend = AnalyticsConnector("databricks://workspace/catalog")

    """

    __slots__ = ("connection_string", "backend_type", "_conn", "_default_database")

    # Dangerous SQL statement types for security (using sqlglot)
    # NOTE: Not all statement types exist in all sqlglot versions
    _DANGEROUS_STMT_TYPES: Final[tuple] = (
        sqlglot.exp.Drop,
        sqlglot.exp.Create,
        sqlglot.exp.Alter,
        sqlglot.exp.Delete,
        sqlglot.exp.Insert,
        sqlglot.exp.Update,
        sqlglot.exp.Merge,
    )

    def __init__(self, connection_string: str) -> None:
        """
        Initialize Ibis backend.

        Args:
            connection_string: Database URL in standard format
                - athena://catalog?region=ap-southeast-2&database=mydb
                - postgres://user:pass@host:5432/database
                - snowflake://account/database?warehouse=compute

        Raises:
            ValueError: If connection string is invalid
            ConnectionError: If connection fails

        """
        if "://" not in connection_string:
            raise ValueError("Connection string must contain '://'")

        self.connection_string = connection_string
        self.backend_type = self._detect_backend_type(connection_string)
        self._default_database = None  # Will be set by helper functions if needed

        try:
            self._conn = ibis.connect(connection_string)
            logger.info(f"✓ Connected to {self.backend_type} backend")
        except Exception as e:
            logger.error(f"Failed to connect to {self.backend_type}: {e}")
            raise ConnectionError(f"Connection failed: {e}") from e

    @staticmethod
    @beartype
    def _detect_backend_type(url: str) -> str:
        """
        Extract backend type from connection string.

        Args:
            url: Connection string URL

        Returns:
            Backend type (e.g., 'athena', 'postgres')

        Examples:
            >>> AnalyticsConnector._detect_backend_type("athena://catalog?region=ap-southeast-2")
            'athena'
            >>> AnalyticsConnector._detect_backend_type("postgres://host/db")
            'postgres'

        """
        if not url or "://" not in url:
            return "unknown"
        backend_type = url.split("://")[0]
        # Normalize postgresql to postgres for consistency
        return "postgres" if backend_type == "postgresql" else backend_type

    @beartype
    def list_databases(self) -> list[str]:
        """
        List available databases.

        Returns:
            List of database names

        Raises:
            RuntimeError: If listing fails

        """
        try:
            databases = self._conn.list_databases()
            logger.debug(f"Found {len(databases)} databases: {databases}")
            return databases
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            raise RuntimeError(f"Database listing failed: {e}") from e

    @beartype
    def list_tables(self, database: str | None = None) -> list[str]:
        """
        List tables in database.

        Args:
            database: Optional database name filter

        Returns:
            List of table names

        """
        try:
            tables = self._conn.list_tables(database=database)
            logger.debug(
                f"Found {len(tables)} tables in {database or 'default database'}"
            )
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    @beartype
    def get_table_schema(
        self, table_name: str, database: str | None = None
    ) -> dict[str, Any]:
        """
        Get table schema information.

        Args:
            table_name: Name of the table
            database: Optional database name

        Returns:
            Schema dictionary with columns and metadata

        """
        try:
            table = self._conn.table(table_name, database=database)
            schema = table.schema()

            columns = [
                {"name": name, "type": str(dtype)} for name, dtype in schema.items()
            ]

            return {
                "columns": columns,
                "column_count": len(schema),
            }
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            return {"columns": [], "column_count": 0}

    @beartype
    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.

        Args:
            sql: SQL query string (SELECT only)

        Returns:
            Query results as pandas DataFrame

        Raises:
            ValueError: If query contains dangerous operations
            RuntimeError: If query execution fails

        Examples:
            >>> backend = AnalyticsConnector("postgres://user:pass@host/db")
            >>> df = backend.execute_query("SELECT * FROM users LIMIT 10")
            >>> len(df)
            10

        """
        # Validate SQL is read-only before execution
        if not self._is_read_only(sql):
            raise ValueError("Only SELECT queries allowed")

        try:
            logger.debug(f"Executing query: {sql[:100]}...")

            # NOTE: Ibis 11.0+ supports Athena sql() method without temporary view issues
            result = self._conn.sql(sql).to_pandas()
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise RuntimeError(f"Query failed: {e}") from e

    @beartype
    def _is_read_only(self, sql: str) -> bool:
        """
        Validate SQL is read-only using sqlglot parsing.

        Args:
            sql: SQL query to validate

        Returns:
            True if query is safe (SELECT only), False otherwise

        """
        try:
            # Parse SQL with sqlglot
            parsed = sqlglot.parse_one(sql)

            # Check if statement type is dangerous
            for dangerous_type in self._DANGEROUS_STMT_TYPES:
                if isinstance(parsed, dangerous_type):
                    logger.warning(
                        f"Dangerous SQL operation detected: {type(parsed).__name__}"
                    )
                    return False

            # Also check for CTEs and subqueries recursively
            for node in parsed.walk():
                for dangerous_type in self._DANGEROUS_STMT_TYPES:
                    if isinstance(node, dangerous_type):
                        logger.warning(
                            f"Dangerous SQL operation in subquery: {type(node).__name__}"
                        )
                        return False

            return True
        except Exception as e:
            # If parsing fails, be conservative and reject
            logger.warning(f"Failed to parse SQL, rejecting as unsafe: {e}")
            return False

    def close(self) -> None:
        """Close connection and clean up resources."""
        if hasattr(self._conn, "close"):
            try:
                self._conn.close()
                logger.info(f"✓ {self.backend_type} connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalyticsConnector(backend='{self.backend_type}')"


@beartype
def create_postgres_backend(
    host: str, database: str, user: str, password: str, port: int = 5432
) -> AnalyticsConnector:
    """
    Create PostgreSQL backend with common parameters.

    Args:
        host: PostgreSQL host
        database: Database name
        user: Username
        password: Password
        port: Port number

    Returns:
        Configured AnalyticsConnector for PostgreSQL

    """
    connection_string = f"postgres://{user}:{password}@{host}:{port}/{database}"
    return AnalyticsConnector(connection_string)
