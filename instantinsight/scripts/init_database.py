#!/usr/bin/env python3
"""
Database Initialization Script for instantinsight.

This script ensures that the PostgreSQL database and RAG system are properly
initialized before starting the Panel application.
"""

import sys
import time
from pathlib import Path

from loguru import logger

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.config.database_config import POSTGRES_CONFIG, RAG_CONFIG
    from src.connectors.database import DatabaseConnectionManager
    from src.rag.backends import PgVectorBackend, PgVectorBackendConfig
    from src.rag.pgvector_rag import PgvectorRAG
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Logging is configured automatically by loguru


def wait_for_database(max_attempts=30, delay=2):
    """
    Wait for PostgreSQL to be ready.

    Args:
        max_attempts: Maximum number of connection attempts
        delay: Delay in seconds between attempts

    Returns:
        True if database is ready, False otherwise

    """
    logger.info("Waiting for PostgreSQL to be ready...")

    for attempt in range(max_attempts):
        try:
            # Test connection
            connection_string = (
                DatabaseConnectionManager.build_postgres_connection_string(
                    POSTGRES_CONFIG
                )
            )
            import psycopg

            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("✓ PostgreSQL is ready")
                    return True
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.info(
                    f"Attempt {attempt + 1}/{max_attempts}: PostgreSQL not ready yet, waiting {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"PostgreSQL failed to become ready after {max_attempts} attempts: {e}"
                )
                return False

    return False


def check_rag_tables():
    """
    Check if RAG tables exist and are properly set up.

    Returns:
        True if RAG tables exist and are properly configured, False otherwise

    """
    logger.info("Checking RAG table setup...")

    try:
        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            POSTGRES_CONFIG
        )
        logger.info(f"Using connection string: {connection_string}")

        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(connection_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # First, check what database we're connected to
                cur.execute("SELECT current_database(), current_user, current_schema()")
                db_info = cur.fetchone()
                logger.info(
                    f"Connected to database: {db_info['current_database']}, user: {db_info['current_user']}, schema: {db_info['current_schema']}"
                )

                # List all tables to see what's available
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
                )
                all_tables = cur.fetchall()
                logger.info(
                    f"Available tables: {[t['table_name'] for t in all_tables]}"
                )

                # Check for required tables
                required_tables = [
                    "rag_documents",
                    "rag_sql_examples",
                    "rag_schema_info",
                ]

                for table in required_tables:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s)",
                        [table],
                    )
                    exists = cur.fetchone()["exists"]

                    if exists:
                        logger.info(f"✓ Table {table} exists")
                    else:
                        logger.error(f"❌ Table {table} does not exist")
                        return False

                # Check pgvector extension
                cur.execute(
                    "SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector')"
                )
                vector_exists = cur.fetchone()["exists"]

                if vector_exists:
                    logger.info("✓ pgvector extension is installed")
                else:
                    logger.error("❌ pgvector extension is not installed")
                    return False

                return True

    except Exception as e:
        logger.error(f"Failed to check RAG tables: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def initialize_rag_system():
    """
    Initialize the RAG system with proper error handling.

    Returns:
        True if initialization successful, False otherwise

    """
    logger.info("Initializing RAG system...")

    try:
        # Build connection string
        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            POSTGRES_CONFIG
        )

        backend_pool = RAG_CONFIG.get("pgvector_pool", {})
        backend_settings = RAG_CONFIG.get("pgvector_backend", {})
        sqlalchemy_url = RAG_CONFIG.get("sqlalchemy_database_url", connection_string)

        backend = PgVectorBackend(
            PgVectorBackendConfig(
                db_url=backend_settings.get("db_url", sqlalchemy_url),
                table_name=backend_settings.get(
                    "table_name", "instantinsight_pgvector_internal"
                ),
                schema=backend_settings.get("schema", "ai"),
                pool_size=backend_pool.get("pool_size", 5),
                max_overflow=backend_pool.get("max_overflow", 10),
            )
        )

        # Create RAG instance (PgvectorRAG only accepts specific parameters now)
        rag = PgvectorRAG(
            connection_string=connection_string,
            backend=backend,
            pool_config=backend_pool,
            backend_config={
                **backend_settings,
                "db_url": backend_settings.get("db_url", sqlalchemy_url),
            },
        )

        # Test connection
        if not rag.connect_to_database():
            logger.error("❌ Failed to connect to database")
            return False

        # Note: is_knowledge_base_ready() and refresh_knowledge_base() methods
        # no longer exist in PgvectorRAG. Skipping these checks.

        # Get statistics
        stats = rag.get_statistics()
        if "error" not in stats:
            # Access the correct structure: stats['totals']['total_documents']
            total_docs = stats.get("totals", {}).get("total_documents", 0)
            logger.info(f"RAG system ready: {total_docs} documents loaded")
        else:
            logger.warning(f"RAG statistics error: {stats['error']}")

        rag.close()
        return True

    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        return False


def main():
    """Execute main database initialization workflow."""
    logger.info("🚀 Starting database initialization...")

    # Step 1: Wait for database
    if not wait_for_database():
        logger.error("❌ Database initialization failed: PostgreSQL not ready")
        sys.exit(1)

    # Step 2: Check RAG tables
    if not check_rag_tables():
        logger.error(
            "❌ Database initialization failed: RAG tables not properly set up"
        )
        logger.error(
            "   Make sure docker-compose.yml includes the pgvector-setup.sql initialization script"
        )
        sys.exit(1)

    # Step 3: Initialize RAG system
    if not initialize_rag_system():
        logger.error(
            "❌ Database initialization failed: RAG system initialization failed"
        )
        sys.exit(1)

    logger.info("✅ Database initialization completed successfully!")
    logger.info("   - PostgreSQL is ready")
    logger.info("   - RAG tables are set up")
    logger.info("   - RAG system is initialized")
    logger.info("   - System is ready for Panel application")


if __name__ == "__main__":
    main()
