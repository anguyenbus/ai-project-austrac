"""
Comprehensive testing with Athena backend.

This test suite validates the Ibis backend works correctly with AWS Athena.
"""

import os

import pytest
from loguru import logger

from src.connectors.analytics_backend import AnalyticsConnector
from src.utils.glue_enricher import GlueMetadataEnricher
from src.utils.schema_introspector import SchemaIntrospector


@pytest.fixture
def athena_url():
    """Get Athena connection URL."""
    url = os.getenv("ANALYTICS_DB_URL")
    if not url or "athena://" not in url:
        pytest.skip("ANALYTICS_DB_URL not configured for Athena")
    return url


@pytest.fixture
def aws_config():
    """AWS configuration for Glue enrichment."""
    return {
        "profile": os.getenv("AWS_PROFILE"),
        "region": os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
    }


@pytest.fixture
def athena_backend(athena_url):
    """Create Athena backend instance."""
    backend = AnalyticsConnector(athena_url)
    yield backend
    backend.close()


@pytest.fixture
def glue_enricher(aws_config):
    """Create Glue metadata enricher."""
    if not aws_config["profile"]:
        pytest.skip("AWS_PROFILE not configured")
    return GlueMetadataEnricher(
        aws_profile=aws_config["profile"], region=aws_config["region"]
    )


class TestAthenaBackendConnectivity:
    """Test Athena backend connectivity and basic operations."""

    def test_backend_type_detection(self, athena_backend):
        """Verify backend type is detected as Athena."""
        assert athena_backend.backend_type == "athena"
        logger.info(f"✓ Backend type: {athena_backend.backend_type}")

    def test_list_databases(self, athena_backend):
        """Test listing databases in Athena."""
        databases = athena_backend.list_databases()

        assert isinstance(databases, list)
        assert len(databases) > 0

        logger.info(f"✓ Found {len(databases)} databases: {databases}")

    def test_list_tables(self, athena_backend):
        """Test listing tables in Athena database."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")
        tables = athena_backend.list_tables(database=database)

        assert isinstance(tables, list)
        logger.info(f"✓ Found {len(tables)} tables in {database}")

    def test_get_table_schema(self, athena_backend):
        """Test getting table schema from Athena."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")
        tables = athena_backend.list_tables(database=database)

        if not tables:
            pytest.skip("No tables found in database")

        # Get schema for first table
        table_name = tables[0]
        schema = athena_backend.get_table_schema(table_name, database=database)

        assert "columns" in schema
        assert "column_count" in schema
        assert len(schema["columns"]) > 0

        logger.info(f"✓ Schema for {table_name}: {schema['column_count']} columns")

    def test_execute_simple_query(self, athena_backend):
        """Test executing a simple query."""
        result = athena_backend.execute_query("SELECT 1 as test_value")

        assert len(result) == 1
        assert "test_value" in result.columns
        assert result["test_value"][0] == 1

        logger.info("✓ Simple query execution successful")

    def test_execute_select_query(self, athena_backend):
        """Test executing SELECT query on real table."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")
        tables = athena_backend.list_tables(database=database)

        if not tables:
            pytest.skip("No tables found in database")

        table_name = tables[0]
        query = f"SELECT * FROM {database}.{table_name} LIMIT 5"

        result = athena_backend.execute_query(query)

        assert len(result) <= 5
        logger.info(f"✓ SELECT query on {table_name} returned {len(result)} rows")

    def test_read_only_enforcement_athena(self, athena_backend):
        """Test read-only enforcement with Athena."""
        dangerous_queries = [
            "DROP TABLE test_table",
            "CREATE TABLE test_table (id INT)",
            "DELETE FROM test_table",
            "INSERT INTO test_table VALUES (1)",
        ]

        for query in dangerous_queries:
            with pytest.raises((ValueError, RuntimeError)):
                athena_backend.execute_query(query)

        logger.info("✓ Read-only enforcement working on Athena")


class TestSchemaIntrospectorWithAthena:
    """Test SchemaIntrospector with Athena backend."""

    def test_extract_database_schema_without_glue(self, athena_backend):
        """Test schema extraction without Glue enrichment."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")

        introspector = SchemaIntrospector(athena_backend)
        schema = introspector.extract_database_schema(database)

        assert schema["database_name"] == database
        assert schema["source"] == "athena"
        assert "tables" in schema
        assert schema["table_count"] > 0

        logger.info(
            f"✓ Extracted schema for {database}: {schema['table_count']} tables"
        )

    def test_extract_database_schema_with_glue(self, athena_backend, glue_enricher):
        """Test schema extraction with Glue enrichment."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")

        introspector = SchemaIntrospector(athena_backend, glue_enricher)
        schema = introspector.extract_database_schema(database)

        assert schema["database_name"] == database
        assert "tables" in schema

        # Check for Glue-enriched metadata
        for table_name, table_info in schema["tables"].items():
            # Glue enrichment should add these fields
            if "s3_location" in table_info:
                logger.info(f"✓ Glue enrichment found for {table_name}")
                assert "partition_keys" in table_info
                break

        logger.info("✓ Schema extraction with Glue enrichment completed")

    def test_get_available_databases(self, athena_backend):
        """Test getting available databases."""
        introspector = SchemaIntrospector(athena_backend)
        databases = introspector.get_available_databases()

        assert isinstance(databases, list)
        assert len(databases) > 0

        logger.info(f"✓ Available databases: {databases}")

    def test_schema_statistics(self, athena_backend):
        """Test schema statistics extraction."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")

        introspector = SchemaIntrospector(athena_backend)
        introspector.extract_database_schema(database)
        stats = introspector.get_schema_statistics(database)

        assert stats["database_name"] == database
        assert stats["backend_type"] == "athena"
        assert stats["table_count"] > 0
        assert "total_columns" in stats
        assert "column_types" in stats

        logger.info(f"✓ Statistics: {stats}")

    def test_schema_caching(self, athena_backend):
        """Test schema caching functionality."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")

        introspector = SchemaIntrospector(athena_backend)

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


class TestGlueEnrichment:
    """Test Glue metadata enrichment specifically."""

    def test_glue_enricher_initialization(self, aws_config):
        """Test Glue enricher can be initialized."""
        if not aws_config["profile"]:
            pytest.skip("AWS_PROFILE not configured")

        enricher = GlueMetadataEnricher(
            aws_profile=aws_config["profile"], region=aws_config["region"]
        )

        assert enricher is not None
        logger.info("✓ Glue enricher initialized")

    def test_glue_enrichment_adds_metadata(self, athena_backend, glue_enricher):
        """Test that Glue enrichment adds metadata to schema."""
        database = os.getenv("ATHENA_DATABASE", "text_to_sql")

        # Extract schema without enrichment
        introspector_plain = SchemaIntrospector(athena_backend)
        schema_plain = introspector_plain.extract_database_schema(database)

        # Extract schema with enrichment
        introspector_enriched = SchemaIntrospector(athena_backend, glue_enricher)
        schema_enriched = introspector_enriched.extract_database_schema(database)

        # Compare: enriched should have additional fields
        for table_name in schema_enriched["tables"]:
            plain_keys = set(schema_plain["tables"][table_name].keys())
            enriched_keys = set(schema_enriched["tables"][table_name].keys())

            # Enriched should have more or equal keys
            assert len(enriched_keys) >= len(plain_keys)

            # Check for Glue-specific keys
            if "s3_location" in enriched_keys:
                logger.info(f"✓ Table {table_name} has Glue metadata")
                assert "s3_location" in schema_enriched["tables"][table_name]
                break


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_database(self, athena_backend):
        """Test handling of invalid database name."""
        introspector = SchemaIntrospector(athena_backend)
        schema = introspector.extract_database_schema("nonexistent_database_xyz")

        # Should return empty schema rather than raising error
        assert schema["database_name"] == "nonexistent_database_xyz"
        assert schema["table_count"] == 0

        logger.info("✓ Invalid database handled gracefully")

    def test_connection_close(self, athena_url):
        """Test proper connection cleanup."""
        backend = AnalyticsConnector(athena_url)
        backend.close()

        # Should be able to close without error
        logger.info("✓ Connection closed successfully")

    def test_context_manager(self, athena_url):
        """Test using backend as context manager."""
        with AnalyticsConnector(athena_url) as backend:
            databases = backend.list_databases()
            assert len(databases) > 0

        # Backend should be closed after context
        logger.info("✓ Context manager working correctly")


@pytest.mark.slow
class TestPerformance:
    """Performance tests for Athena backend."""

    def test_schema_extraction_performance(self, athena_backend):
        """Benchmark schema extraction performance."""
        import time

        database = os.getenv("ATHENA_DATABASE", "text_to_sql")
        introspector = SchemaIntrospector(athena_backend)

        start = time.time()
        schema = introspector.extract_database_schema(database)
        duration = time.time() - start

        table_count = schema["table_count"]
        avg_per_table = duration / table_count if table_count > 0 else 0

        logger.info(
            f"✓ Schema extraction: {table_count} tables in {duration:.2f}s "
            f"({avg_per_table:.2f}s per table)"
        )

        # Should not take more than 5s per table
        assert avg_per_table < 5.0

    def test_query_execution_performance(self, athena_backend):
        """Benchmark query execution performance."""
        import time

        database = os.getenv("ATHENA_DATABASE", "text_to_sql")
        tables = athena_backend.list_tables(database=database)

        if not tables:
            pytest.skip("No tables found")

        table_name = tables[0]
        query = f"SELECT * FROM {database}.{table_name} LIMIT 100"

        start = time.time()
        result = athena_backend.execute_query(query)
        duration = time.time() - start

        logger.info(f"✓ Query execution: {len(result)} rows in {duration:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
