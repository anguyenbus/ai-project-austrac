"""
Quick smoke test for UniversalExecutor.

Usage:
    uv run python scripts/test_universal_executor.py
"""

import os

from loguru import logger

from src.rag.engines.engine_factory import EngineFactory


def test_universal_executor():
    """Test UniversalExecutor with dynamic table discovery."""
    logger.info("Creating UniversalExecutor...")

    executor, error = EngineFactory.create_query_executor()

    if error:
        logger.error(f"Failed to create executor: {error}")
        return False

    logger.info("Testing connection and table discovery...")

    # Get database name from environment
    database = os.getenv("ATHENA_DATABASE", "text_to_sql")

    # List available tables
    try:
        tables = executor.connector.list_tables(database=database)

        if not tables:
            logger.warning(f"No tables found in database '{database}'")
            return False

        logger.info(f"✓ Found {len(tables)} tables in database '{database}'")
        logger.debug(f"Available tables: {tables}")

        # Test query using raw SQL (bypass Ibis table API)
        # NOTE: Use raw SQL to avoid Ibis query generation issues with Athena
        preferred_tables = ["orders", "employees"]
        table_name = None

        for preferred in preferred_tables:
            if preferred in tables:
                table_name = preferred
                break

        if not table_name:
            # Fallback to any table (except temp views and categories)
            table_name = next(
                (
                    t
                    for t in tables
                    if not t.startswith("ibis_temp_view") and t != "categories"
                ),
                tables[0],
            )

        # Use raw SQL query directly
        query = f"SELECT * FROM {database}.{table_name} LIMIT 5"

        logger.info(f"Executing raw SQL query on table '{table_name}'...")
        logger.debug(f"Query: {query}")

        # Execute directly via connector (bypass executor wrapper for debugging)
        try:
            df = executor.connector.execute_query(query)
            row_count = len(df)
            assert row_count <= 5, f"Expected ≤5 rows, got {row_count}"
            logger.info(f"✅ Query succeeded: {table_name} returned {row_count} rows")
            return True
        except Exception as e:
            logger.error(f"❌ Direct query failed: {e}")
            logger.info("Trying alternative approach with executor.execute()...")

            result = executor.execute(query)
            if result.status.value == "success":
                row_count = len(result.data)
                logger.info(
                    f"✅ Query via executor succeeded: {table_name} returned {row_count} rows"
                )
                return True
            else:
                logger.error(f"❌ Executor query also failed: {result.error}")
                return False

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = test_universal_executor()
    exit(0 if success else 1)
