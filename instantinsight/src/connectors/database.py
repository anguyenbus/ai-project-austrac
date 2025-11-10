"""
Database Connection Utilities.

Centralized database connection logic to eliminate duplication across the codebase.
Provides standardized PostgreSQL connection string building and SQLAlchemy engine creation.
"""

from typing import Any

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class DatabaseConnectionManager:
    """Centralized database connection management."""

    @staticmethod
    def build_postgres_connection_string(config: dict[str, Any]) -> str:
        """
        Build PostgreSQL connection string from config dictionary.

        Args:
            config: Dictionary containing PostgreSQL connection parameters

        Returns:
            str: PostgreSQL connection string in format postgresql://user:password@host:port/database

        Example:
            config = {
                "host": "localhost",
                "port": "5432",
                "user": "postgres",
                "password": "postgres",
                "database": "instantinsight"
            }
            connection_string = DatabaseConnectionManager.build_postgres_connection_string(config)

        """
        host = config.get("host", "localhost")
        port = config.get("port", "5432")
        user = config.get("user", "postgres")
        password = config.get("password", "postgres")
        database = config.get("database", "instantinsight")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    @staticmethod
    def create_postgres_engine(
        config: dict[str, Any],
        pool_pre_ping: bool = True,
        pool_recycle: int = 3600,
        echo: bool = False,
        **engine_kwargs,
    ) -> Engine:
        """
        Create SQLAlchemy engine for PostgreSQL with standardized configuration.

        Args:
            config: Dictionary containing PostgreSQL connection parameters
            pool_pre_ping: Enable connection health checks
            pool_recycle: Connection pool recycle time in seconds
            echo: Enable SQL logging
            **engine_kwargs: Additional SQLAlchemy engine parameters

        Returns:
            Engine: Configured SQLAlchemy engine

        """
        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            config
        )

        engine_params = {
            "pool_pre_ping": pool_pre_ping,
            "pool_recycle": pool_recycle,
            "echo": echo,
            **engine_kwargs,
        }

        engine = create_engine(connection_string, **engine_params)
        logger.info(
            f"✓ PostgreSQL SQLAlchemy engine created for {config.get('database', 'unknown')}"
        )

        return engine

    @staticmethod
    def test_postgres_connection(config: dict[str, Any]) -> bool:
        """
        Test PostgreSQL connection using provided configuration.

        Args:
            config: Dictionary containing PostgreSQL connection parameters

        Returns:
            bool: True if connection successful, False otherwise

        """
        try:
            engine = DatabaseConnectionManager.create_postgres_engine(config)
            with engine.connect() as conn:
                from sqlalchemy import text

                result = conn.execute(text("SELECT 1"))
                success = result.scalar() == 1
                if success:
                    logger.info("✓ PostgreSQL connection test successful")
                return success
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection test failed: {e}")
            return False
