"""
Database Manager for handling multiple database connections and query routing.

This component centralizes all database operations that were previously
scattered throughout VannaRAGEngine.
"""

import re
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

import pandas as pd
import psycopg
from loguru import logger

try:
    from psycopg_pool import ConnectionPool

    POOL_AVAILABLE = True
except ImportError:
    logger.warning("psycopg_pool not available, using single connections")
    POOL_AVAILABLE = False
# Import Ibis backend for universal database support
import os

from psycopg.rows import dict_row

try:
    from src.connectors.analytics_backend import AnalyticsConnector

    IBIS_AVAILABLE = True
except ImportError:
    logger.warning("AnalyticsConnector not available")
    IBIS_AVAILABLE = False

from ..components.engine_config import AthenaConfig, DatabaseConfig, PostgresConfig


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""

    @abstractmethod
    def execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is working."""
        pass

    @abstractmethod
    def get_table_list(self) -> list[str]:
        """Get list of available tables."""
        pass

    @abstractmethod
    def close(self):
        """Close the connection."""
        pass


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL connection with pgvector support."""

    def __init__(self, config: PostgresConfig, use_pool: bool = True):
        """
        Initialize PostgreSQL connection.

        Args:
            config: PostgreSQL configuration
            use_pool: Whether to use connection pooling

        """
        self.config = config
        self.use_pool = use_pool
        self._pool = None
        self._conn = None
        # No lock needed - connection pool handles thread safety
        # For single connection mode, use only in single-threaded contexts

        if use_pool:
            self._init_pool()
        else:
            self._init_connection()

    def _init_pool(self):
        """Initialize connection pool."""
        if not POOL_AVAILABLE:
            logger.info("Pool not available, using single connection")
            self._init_connection()
            return

        try:
            # Get timeout settings from config
            pool_timeout = getattr(self.config, "pool_timeout", 30)

            self._pool = ConnectionPool(
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                conninfo=self.config.to_connection_string(),
                timeout=pool_timeout,
                kwargs={"row_factory": dict_row, "autocommit": True},
            )
            logger.info(
                f"PostgreSQL connection pool initialized (min={self.config.min_connections}, max={self.config.max_connections})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            # Fallback to single connection
            self.use_pool = False
            self._init_connection()

    def _init_connection(self):
        """Initialize single connection."""
        try:
            self._conn = psycopg.connect(
                self.config.to_connection_string(),
                row_factory=dict_row,
                autocommit=True,
            )
            logger.info("PostgreSQL connection initialized")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a connection from pool or return single connection."""
        if self.use_pool and self._pool:
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                self._pool.putconn(conn)
        else:
            yield self._conn

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Execute SQL query.

        Thread-safe when using connection pool (default).
        For single connection mode, use only in single-threaded contexts.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Set statement timeout if configured
                    if hasattr(self.config, "statement_timeout_ms"):
                        cursor.execute(
                            f"SET statement_timeout = {self.config.statement_timeout_ms}"
                        )

                    cursor.execute(sql, params)

                    # Check if query returns results
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()

                        # Convert to DataFrame
                        if rows:
                            df = pd.DataFrame(rows, columns=columns)
                        else:
                            df = pd.DataFrame(columns=columns)

                        return df
                    else:
                        # No results (e.g., INSERT, UPDATE)
                        return pd.DataFrame()

        except Exception as e:
            logger.error(f"PostgreSQL query failed: {e}")
            raise

    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {e}")
            return False

    def get_table_list(self) -> list[str]:
        """Get list of tables."""
        sql = """
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
        df = self.execute(sql)
        return df["tablename"].tolist() if not df.empty else []

    def close(self):
        """Close connection(s)."""
        if self._pool:
            self._pool.close()
            logger.info("PostgreSQL connection pool closed")
        elif self._conn:
            self._conn.close()
            logger.info("PostgreSQL connection closed")


class AnalyticsConnection(DatabaseConnection):
    """Universal analytics database connection using Ibis."""

    def __init__(self, config: AthenaConfig):
        """
        Initialize analytics connection using Ibis backend.

        Args:
            config: AthenaConfig (for backward compatibility) or connection string

        Note:
            This class now uses AnalyticsConnector and ANALYTICS_DB_URL instead of
            the legacy AthenaConnectionManager. It supports any Ibis backend
            (Athena, PostgreSQL, Snowflake, etc.).

        """
        self.config = config
        self._backend = None

        if not IBIS_AVAILABLE:
            raise ImportError("AnalyticsConnector not available")

        self._init_backend()

    def _init_backend(self):
        """Initialize Ibis backend from ANALYTICS_DB_URL."""
        try:
            # Get analytics DB URL from environment
            analytics_db_url = os.getenv("ANALYTICS_DB_URL")

            if not analytics_db_url:
                # Fallback: construct from AthenaConfig
                logger.warning(
                    "ANALYTICS_DB_URL not set, constructing from AthenaConfig"
                )
                analytics_db_url = self._construct_url_from_config()

                # For Athena, set database via environment variable
                if hasattr(self.config, "database") and self.config.database:
                    os.environ["ATHENA_DATABASE"] = self.config.database

                logger.info(f"✓ Constructed URL from config: {analytics_db_url}")

            # Create analytics connector directly
            self._backend = AnalyticsConnector(analytics_db_url)
            logger.info(
                f"✓ Analytics connector initialized: {self._backend.backend_type}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Ibis backend: {e}")
            raise

    def _construct_url_from_config(self) -> str:
        """
        Construct Athena URL from legacy AthenaConfig.

        Note: Database should be set via ATHENA_DATABASE env var instead of
        including it in URL query parameters to avoid Ibis connection issues.
        """
        params = [f"region={self.config.region}"]

        if hasattr(self.config, "workgroup") and self.config.workgroup:
            params.append(f"work_group={self.config.workgroup}")

        if hasattr(self.config, "s3_staging_dir") and self.config.s3_staging_dir:
            params.append(f"s3_staging_dir={self.config.s3_staging_dir}")

        return f"athena://awsdatacatalog?{'&'.join(params)}"

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute SQL query using Ibis backend."""
        if params:
            logger.warning(
                "Ibis does not support parameterized queries, ignoring params"
            )

        try:
            return self._backend.execute_query(sql)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            self._backend.list_databases()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_table_list(self) -> list[str]:
        """Get list of tables from database."""
        try:
            return self._backend.list_tables()
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    def close(self):
        """Close database connection."""
        if self._backend:
            self._backend.close()
        logger.info(
            f"✓ {self._backend.backend_type if self._backend else 'Analytics'} connection closed"
        )


# Backward compatibility alias
AthenaConnection = AnalyticsConnection


class DatabaseRouter:
    """Routes queries to appropriate database based on patterns and rules."""

    def __init__(self, config: DatabaseConfig):
        """
        Initialize query router.

        Args:
            config: Database configuration

        """
        self.config = config
        self._routing_cache = {}

        # Compile regex patterns for efficiency
        self._athena_patterns = []
        if config.athena_table_patterns:
            for pattern in config.athena_table_patterns:
                regex = pattern.replace("*", ".*")
                self._athena_patterns.append(re.compile(regex, re.IGNORECASE))

    def route_query(self, sql: str) -> str:
        """
        Determine which database to use for a query.

        Returns:
            'athena' or 'postgresql'

        """
        # Check cache first
        sql_hash = hash(sql[:200])  # Hash first 200 chars
        if sql_hash in self._routing_cache:
            return self._routing_cache[sql_hash]

        # Determine routing
        database = self._determine_database(sql)

        # Cache result
        self._routing_cache[sql_hash] = database

        # Limit cache size
        if len(self._routing_cache) > 1000:
            self._routing_cache.clear()

        return database

    def _determine_database(self, sql: str) -> str:
        """Determine target database from SQL."""
        if not self.config.use_athena_for_business_data or not self.config.athena:
            return "postgresql"

        # Extract table names from SQL
        tables = self._extract_table_names(sql)

        # Check if any table matches Athena patterns
        for table in tables:
            for pattern in self._athena_patterns:
                if pattern.match(table):
                    logger.debug(
                        f"Table '{table}' matched Athena pattern, routing to Athena"
                    )
                    return "athena"

        return "postgresql"

    def _extract_table_names(self, sql: str) -> list[str]:
        """Extract table names from SQL query."""
        # Simple regex to find table names
        # This is a simplified approach - a full SQL parser would be better
        sql_upper = sql.upper()

        tables = []

        # Find tables after FROM
        from_pattern = r"FROM\s+([^\s,]+)"
        from_matches = re.findall(from_pattern, sql_upper)
        tables.extend(from_matches)

        # Find tables after JOIN
        join_pattern = r"JOIN\s+([^\s,]+)"
        join_matches = re.findall(join_pattern, sql_upper)
        tables.extend(join_matches)

        # Clean and return unique tables
        cleaned_tables = []
        for table in tables:
            # Remove schema prefix if present
            if "." in table:
                table = table.split(".")[-1]
            # Remove quotes
            table = table.strip("\"'`")
            cleaned_tables.append(table.lower())

        return list(set(cleaned_tables))


class DatabaseManager:
    """
    Manages multiple database connections and routes queries.

    This replaces the scattered database logic in VannaRAGEngine.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager.

        Args:
            config: Database configuration

        """
        self.config = config
        self._connections: dict[str, DatabaseConnection] = {}
        self._router = DatabaseRouter(config)
        self._lock = threading.Lock()

        # Initialize connections
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize all configured database connections."""
        # PostgreSQL is required
        if self.config.postgres:
            try:
                self._connections["postgresql"] = PostgreSQLConnection(
                    self.config.postgres, use_pool=True
                )
                logger.info("PostgreSQL connection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL: {e}")
                raise

        # Analytics DB (Athena/Snowflake/etc.) is optional
        if self.config.athena and self.config.use_athena_for_business_data:
            try:
                self._connections["athena"] = AnalyticsConnection(self.config.athena)
                logger.info("✓ Analytics connection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize analytics connection: {e}")
                # Don't fail completely if analytics DB is unavailable

    def execute_sql(
        self, sql: str, database: str = "auto", params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """
        Execute SQL query on appropriate database.

        Args:
            sql: SQL query to execute
            database: Target database ('postgresql', 'athena', 'auto')
            params: Query parameters (only supported by PostgreSQL)

        Returns:
            Query results as DataFrame

        """
        with self._lock:
            # Determine target database
            if database == "auto":
                database = self._router.route_query(sql)

            # Validate database choice
            if database not in self._connections:
                available = list(self._connections.keys())
                raise ValueError(
                    f"Database '{database}' not available. Available: {available}"
                )

            # Execute query
            logger.debug(f"Executing query on {database}")
            try:
                return self._connections[database].execute(sql, params)
            except Exception as e:
                logger.error(f"Query execution failed on {database}: {e}")
                raise

    def test_connections(self) -> dict[str, bool]:
        """Test all configured connections."""
        results = {}
        for name, conn in self._connections.items():
            results[name] = conn.test_connection()
        return results

    def get_available_databases(self) -> list[str]:
        """Get list of available databases."""
        return list(self._connections.keys())

    def get_tables(self, database: str | None = None) -> dict[str, list[str]]:
        """
        Get tables from specified database or all databases.

        Args:
            database: Specific database or None for all

        Returns:
            Dictionary mapping database name to table list

        """
        results = {}

        if database:
            if database in self._connections:
                results[database] = self._connections[database].get_table_list()
        else:
            for name, conn in self._connections.items():
                try:
                    results[name] = conn.get_table_list()
                except Exception as e:
                    logger.error(f"Failed to get tables from {name}: {e}")
                    results[name] = []

        return results

    def get_connection(self, database: str) -> DatabaseConnection:
        """
        Get specific database connection.

        Args:
            database: Database name

        Returns:
            Database connection object

        """
        if database not in self._connections:
            raise ValueError(f"Unknown database: {database}")
        return self._connections[database]

    def close_all(self):
        """Close all database connections."""
        for name, conn in self._connections.items():
            try:
                conn.close()
                logger.info(f"Closed {name} connection")
            except Exception as e:
                logger.error(f"Error closing {name} connection: {e}")

    def get_routing_decision(self, sql: str) -> tuple[str, str]:
        """
        Get routing decision with explanation.

        Args:
            sql: SQL query

        Returns:
            Tuple of (database, reason)

        """
        database = self._router.route_query(sql)

        if database == "athena":
            tables = self._router._extract_table_names(sql)
            matching_tables = []
            for table in tables:
                for pattern in self._router._athena_patterns:
                    if pattern.match(table):
                        matching_tables.append(table)
            reason = f"Tables {matching_tables} match Athena patterns"
        else:
            reason = "No Athena table patterns matched, using PostgreSQL"

        return database, reason

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()

    def __repr__(self) -> str:
        """Return string representation."""
        connections = list(self._connections.keys())
        return f"DatabaseManager(connections={connections})"
