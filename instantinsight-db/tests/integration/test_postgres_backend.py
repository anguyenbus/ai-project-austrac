"""
Test PostgreSQL backend with Ibis.

This test suite validates the Ibis backend works correctly with PostgreSQL,
demonstrating the universal nature of the implementation.
"""

import os

import pytest
from loguru import logger

from src.connectors.analytics_backend import AnalyticsConnector
from src.utils.schema_introspector import SchemaIntrospector


@pytest.fixture
def postgres_url():
    """Get PostgreSQL connection URL."""
    # Try ANALYTICS_DB_URL first
    url = os.getenv("ANALYTICS_DB_URL")
    if url and "postgres" in url:
        return url

    # Try constructing from individual env vars
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")

    return f"postgres://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture
def postgres_backend(postgres_url):
    """Create PostgreSQL backend instance."""
    try:
        backend = AnalyticsConnector(postgres_url)
        yield backend
        backend.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


class TestPostgreSQLBackendConnectivity:
    """Test PostgreSQL backend connectivity and basic operations."""

    def test_backend_type_detection(self, postgres_backend):
        """Verify backend type is detected as postgres."""
        assert postgres_backend.backend_type == "postgres"
        logger.info(f"✓ Backend type: {postgres_backend.backend_type}")

    def test_list_databases(self, postgres_backend):
        """Test listing databases in PostgreSQL."""
        databases = postgres_backend.list_databases()

        assert isinstance(databases, list)
        assert len(databases) > 0

        logger.info(f"✓ Found {len(databases)} databases: {databases}")

    def test_list_tables(self, postgres_backend):
        """Test listing tables in PostgreSQL database."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]
        tables = postgres_backend.list_tables(database=database)

        assert isinstance(tables, list)
        logger.info(f"✓ Found {len(tables)} tables in {database}")

    def test_get_table_schema(self, postgres_backend):
        """Test getting table schema from PostgreSQL."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]
        tables = postgres_backend.list_tables(database=database)

        if not tables:
            pytest.skip("No tables found in database")

        # Get schema for first table
        table_name = tables[0]
        schema = postgres_backend.get_table_schema(table_name, database=database)

        assert "columns" in schema
        assert "column_count" in schema
        assert len(schema["columns"]) > 0

        logger.info(f"✓ Schema for {table_name}: {schema['column_count']} columns")

    def test_execute_simple_query(self, postgres_backend):
        """Test executing a simple query."""
        result = postgres_backend.execute_query("SELECT 1 as test_value")

        assert len(result) == 1
        assert "test_value" in result.columns
        assert result["test_value"][0] == 1

        logger.info("✓ Simple query execution successful")

    def test_execute_select_query(self, postgres_backend):
        """Test executing SELECT query on real table."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]
        tables = postgres_backend.list_tables(database=database)

        if not tables:
            pytest.skip("No tables found")

        table_name = tables[0]
        query = f'SELECT * FROM "{table_name}" LIMIT 5'

        result = postgres_backend.execute_query(query)

        assert len(result) <= 5
        logger.info(f"✓ SELECT query on {table_name} returned {len(result)} rows")

    def test_read_only_enforcement_postgres(self, postgres_backend):
        """Test read-only enforcement with PostgreSQL."""
        dangerous_queries = [
            "DROP TABLE test_table",
            "CREATE TABLE test_table (id INT)",
            "DELETE FROM test_table",
            "INSERT INTO test_table VALUES (1)",
        ]

        for query in dangerous_queries:
            with pytest.raises((ValueError, RuntimeError)):
                postgres_backend.execute_query(query)

        logger.info("✓ Read-only enforcement working on PostgreSQL")


class TestSchemaIntrospectorWithPostgreSQL:
    """Test SchemaIntrospector with PostgreSQL backend."""

    def test_extract_database_schema(self, postgres_backend):
        """Test schema extraction from PostgreSQL."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]

        introspector = SchemaIntrospector(postgres_backend)
        schema = introspector.extract_database_schema(database)

        assert schema["database_name"] == database
        assert schema["source"] == "postgres"
        assert "tables" in schema
        assert schema["table_count"] >= 0

        logger.info(
            f"✓ Extracted schema for {database}: {schema['table_count']} tables"
        )

    def test_get_available_databases(self, postgres_backend):
        """Test getting available databases."""
        introspector = SchemaIntrospector(postgres_backend)
        databases = introspector.get_available_databases()

        assert isinstance(databases, list)
        assert len(databases) > 0

        logger.info(f"✓ Available databases: {databases}")

    def test_schema_statistics(self, postgres_backend):
        """Test schema statistics extraction."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]

        introspector = SchemaIntrospector(postgres_backend)
        introspector.extract_database_schema(database)
        stats = introspector.get_schema_statistics(database)

        assert stats["database_name"] == database
        assert stats["backend_type"] == "postgres"
        assert "table_count" in stats
        assert "total_columns" in stats
        assert "column_types" in stats

        logger.info(f"✓ Statistics: {stats}")

    def test_schema_caching(self, postgres_backend):
        """Test schema caching functionality."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]

        introspector = SchemaIntrospector(postgres_backend)

        # First extraction
        import time

        start = time.time()
        schema1 = introspector.extract_database_schema(database)
        duration1 = time.time() - start

        # Second extraction (cached)
        start = time.time()
        schema2 = introspector.extract_database_schema(database)
        duration2 = time.time() - start

        assert schema1 == schema2
        assert duration2 < duration1, "Cached extraction should be faster"

        logger.info(f"✓ Caching: First={duration1:.2f}s, Cached={duration2:.2f}s")


class TestUniversalCompatibility:
    """Test that same code works for both Athena and PostgreSQL."""

    def test_same_interface_works(self, postgres_backend):
        """Verify the same interface works regardless of backend."""
        # These operations should work identically for any backend
        databases = postgres_backend.list_databases()
        assert isinstance(databases, list)

        if databases:
            tables = postgres_backend.list_tables(database=databases[0])
            assert isinstance(tables, list)

        result = postgres_backend.execute_query("SELECT 1 as test")
        assert len(result) == 1

        logger.info("✓ Universal interface works correctly")

    def test_schema_introspector_universal(self, postgres_backend):
        """Verify SchemaIntrospector works universally."""
        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]

        # Same code that works for Athena should work for PostgreSQL
        introspector = SchemaIntrospector(postgres_backend)
        schema = introspector.extract_database_schema(database)

        # Standard schema structure regardless of backend
        assert "database_name" in schema
        assert "tables" in schema
        assert "table_count" in schema
        assert "source" in schema

        logger.info("✓ SchemaIntrospector works universally")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_database(self, postgres_backend):
        """Test handling of invalid database name."""
        introspector = SchemaIntrospector(postgres_backend)
        schema = introspector.extract_database_schema("nonexistent_database_xyz")

        # Should return empty schema rather than raising error
        assert schema["database_name"] == "nonexistent_database_xyz"
        assert schema["table_count"] == 0

        logger.info("✓ Invalid database handled gracefully")

    def test_connection_close(self, postgres_url):
        """Test proper connection cleanup."""
        backend = AnalyticsConnector(postgres_url)
        backend.close()

        # Should be able to close without error
        logger.info("✓ Connection closed successfully")

    def test_context_manager(self, postgres_url):
        """Test using backend as context manager."""
        with AnalyticsConnector(postgres_url) as backend:
            databases = backend.list_databases()
            assert len(databases) > 0

        # Backend should be closed after context
        logger.info("✓ Context manager working correctly")


@pytest.mark.slow
class TestPerformance:
    """Performance tests for PostgreSQL backend."""

    def test_schema_extraction_performance(self, postgres_backend):
        """Benchmark schema extraction performance."""
        import time

        databases = postgres_backend.list_databases()
        if not databases:
            pytest.skip("No databases found")

        database = databases[0]
        introspector = SchemaIntrospector(postgres_backend)

        start = time.time()
        schema = introspector.extract_database_schema(database)
        duration = time.time() - start

        table_count = schema["table_count"]
        avg_per_table = duration / table_count if table_count > 0 else 0

        logger.info(
            f"✓ Schema extraction: {table_count} tables in {duration:.2f}s "
            f"({avg_per_table:.2f}s per table)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
