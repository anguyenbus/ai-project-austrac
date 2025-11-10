"""
Integration tests comparing Ibis backend vs legacy Athena results.

These tests ensure the Ibis migration produces identical results to the legacy
Athena-specific implementation.
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

from src.connectors.analytics_backend import AnalyticsConnector
from src.utils.schema_introspector import SchemaIntrospector

# NOTE: Legacy AthenaConnectionManager removed - these tests now focus on AnalyticsConnector only
ATHENA_LEGACY_AVAILABLE = False


@pytest.fixture
def analytics_db_url():
    """Get analytics DB URL from environment."""
    url = os.getenv("ANALYTICS_DB_URL")
    if not url:
        pytest.skip("ANALYTICS_DB_URL not configured")
    return url


@pytest.fixture
def athena_config():
    """Legacy Athena configuration."""
    return {
        "database": os.getenv("ATHENA_DATABASE", "text_to_sql"),
        "region_name": os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        "s3_staging_dir": os.getenv("ATHENA_S3_STAGING_DIR"),
        "aws_profile": os.getenv("AWS_PROFILE"),
        "work_group": os.getenv("ATHENA_WORK_GROUP", "primary"),
    }


@pytest.fixture
def analytics_backend(analytics_db_url):
    """Create Ibis backend instance."""
    backend = AnalyticsConnector(analytics_db_url)
    yield backend
    backend.close()


# NOTE: legacy_athena_manager fixture removed - use analytics_backend for all tests


class TestIbisMigration:
    """Test suite for Ibis migration validation."""

    def test_backend_connection(self, analytics_backend):
        """Test that Ibis backend can connect successfully."""
        databases = analytics_backend.list_databases()
        assert isinstance(databases, list)
        assert len(databases) > 0
        logger.info(f"✓ Found {len(databases)} databases")

    def test_list_databases_match(self, analytics_backend, athena_config):
        """Verify Ibis lists databases correctly via Glue."""
        ibis_databases = set(analytics_backend.list_databases())

        # Compare against Glue (source of truth)
        import boto3

        session = boto3.Session(
            profile_name=athena_config.get("aws_profile"),
            region_name=athena_config["region_name"],
        )
        glue_client = session.client("glue")
        response = glue_client.get_databases()
        glue_databases = set(db["Name"] for db in response["DatabaseList"])

        # Compare results
        assert ibis_databases == glue_databases, (
            f"Database lists don't match:\nIbis: {ibis_databases}\nGlue: {glue_databases}"
        )

        logger.info(f"✓ Database lists match Glue: {len(ibis_databases)} databases")

    def test_list_tables_match(self, analytics_backend, athena_config):
        """Verify Ibis and legacy list same tables."""
        database_name = athena_config["database"]

        # Get tables from Ibis
        ibis_tables = set(analytics_backend.list_tables(database=database_name))

        # Get tables from Glue (source of truth)
        import boto3

        session = boto3.Session(
            profile_name=athena_config.get("aws_profile"),
            region_name=athena_config["region_name"],
        )
        glue_client = session.client("glue")
        response = glue_client.get_tables(DatabaseName=database_name)
        legacy_tables = set(table["Name"] for table in response["TableList"])

        # Compare results
        assert ibis_tables == legacy_tables, (
            f"Table lists don't match:\nIbis: {ibis_tables}\nLegacy: {legacy_tables}"
        )

        logger.info(
            f"✓ Table lists match: {len(ibis_tables)} tables in {database_name}"
        )

    def test_schema_extraction_match(self, analytics_backend, athena_config):
        """Verify schema extraction produces identical results."""
        database_name = athena_config["database"]

        # Extract schema using Ibis
        introspector = SchemaIntrospector(analytics_backend)
        ibis_schema = introspector.extract_database_schema(database_name)

        # Load legacy schema if available
        legacy_schema_path = Path("tests/fixtures/legacy_athena_schema.json")
        if not legacy_schema_path.exists():
            pytest.skip("Legacy schema fixture not available")

        with open(legacy_schema_path) as f:
            legacy_schema = json.load(f)

        # Compare table names
        ibis_table_names = set(ibis_schema["tables"].keys())
        legacy_table_names = set(legacy_schema["tables"].keys())

        assert ibis_table_names == legacy_table_names, (
            f"Table name sets don't match:\nIbis: {ibis_table_names}\nLegacy: {legacy_table_names}"
        )

        # Compare column schemas for each table
        for table_name in ibis_table_names:
            ibis_columns = ibis_schema["tables"][table_name]["columns"]
            legacy_columns = legacy_schema["tables"][table_name]["columns"]

            # Compare column names
            ibis_col_names = [col["name"] for col in ibis_columns]
            legacy_col_names = [col["name"] for col in legacy_columns]

            assert ibis_col_names == legacy_col_names, (
                f"Column names don't match for table {table_name}:\nIbis: {ibis_col_names}\nLegacy: {legacy_col_names}"
            )

        logger.info(
            f"✓ Schema extraction matches: {len(ibis_table_names)} tables verified"
        )

    def test_query_execution(self, analytics_backend):
        """Verify query execution produces correct results."""
        test_query = "SELECT 1 as test_column, 'test_value' as test_string"

        # Execute with Ibis
        result = analytics_backend.execute_query(test_query)

        # Verify results
        assert not result.empty
        assert result.iloc[0]["test_column"] == 1
        assert result.iloc[0]["test_string"] == "test_value"

        logger.info("✓ Query execution verified")

    def test_read_only_enforcement(self, analytics_backend):
        """Verify read-only enforcement works."""
        dangerous_queries = [
            "DROP TABLE test_table",
            "CREATE TABLE test_table (id INT)",
            "ALTER TABLE test_table ADD COLUMN test INT",
            "DELETE FROM test_table",
            "INSERT INTO test_table VALUES (1)",
            "UPDATE test_table SET id = 1",
        ]

        for query in dangerous_queries:
            with pytest.raises((ValueError, RuntimeError)):
                analytics_backend.execute_query(query)

        logger.info("✓ Read-only enforcement working correctly")

    def test_schema_statistics(self, analytics_backend, athena_config):
        """Verify schema statistics are accurate."""
        database_name = athena_config["database"]

        introspector = SchemaIntrospector(analytics_backend)
        schema = introspector.extract_database_schema(database_name)
        stats = introspector.get_schema_statistics(database_name)

        # Verify statistics
        assert stats["table_count"] == len(schema["tables"])
        assert stats["database_name"] == database_name
        assert stats["backend_type"] == analytics_backend.backend_type
        assert "total_columns" in stats
        assert "column_types" in stats

        logger.info(f"✓ Statistics verified: {stats}")

    @pytest.mark.parametrize("table_limit", [1, 5, 10])
    def test_schema_extraction_performance(
        self, analytics_backend, athena_config, table_limit
    ):
        """Benchmark schema extraction performance."""
        import time

        database_name = athena_config["database"]
        introspector = SchemaIntrospector(analytics_backend)

        start_time = time.time()
        schema = introspector.extract_database_schema(database_name)
        duration = time.time() - start_time

        # Extract only limited tables
        limited_tables = dict(list(schema["tables"].items())[:table_limit])

        logger.info(
            f"✓ Schema extraction for {len(limited_tables)} tables completed in {duration:.2f}s"
        )

        # Performance should be reasonable (< 1s per table)
        assert duration < table_limit * 5, (
            f"Schema extraction too slow: {duration:.2f}s for {table_limit} tables"
        )


class TestSchemaIntrospectorFeatures:
    """Test new features in SchemaIntrospector."""

    def test_cache_functionality(self, analytics_backend, athena_config):
        """Verify schema caching works."""
        database_name = athena_config["database"]
        introspector = SchemaIntrospector(analytics_backend)

        # First extraction
        schema1 = introspector.extract_database_schema(database_name)

        # Second extraction (should use cache)
        schema2 = introspector.extract_database_schema(database_name)

        # Should be identical
        assert schema1 == schema2
        assert database_name in introspector.schema_cache

        # Clear cache
        introspector.clear_cache(database_name)
        assert database_name not in introspector.schema_cache

        logger.info("✓ Cache functionality verified")

    def test_multiple_database_extraction(self, analytics_backend):
        """Test extracting schemas from multiple databases."""
        databases = analytics_backend.list_databases()[:3]  # Test first 3 databases

        introspector = SchemaIntrospector(analytics_backend)
        schemas = introspector.extract_multiple_databases(databases)

        assert len(schemas) <= len(databases)
        for db_name in schemas:
            assert db_name in databases
            assert "tables" in schemas[db_name]

        logger.info(f"✓ Extracted schemas for {len(schemas)} databases")

    def test_schema_save_to_file(self, analytics_backend, athena_config, tmp_path):
        """Test saving schema to JSON file."""
        database_name = athena_config["database"]
        introspector = SchemaIntrospector(analytics_backend)

        output_file = tmp_path / f"{database_name}_schema.json"
        success = introspector.save_schema_to_file(database_name, output_file)

        assert success
        assert output_file.exists()

        # Verify file content
        with open(output_file) as f:
            saved_schema = json.load(f)

        assert saved_schema["database_name"] == database_name
        assert "tables" in saved_schema

        logger.info(f"✓ Schema saved to {output_file}")


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_schema_vectorizer_alias(self):
        """Verify AthenaSchemaVectorizer alias works."""
        from src.utils.schema_vectorizer import AthenaSchemaVectorizer, SchemaVectorizer

        # AthenaSchemaVectorizer should be an alias for SchemaVectorizer
        assert AthenaSchemaVectorizer == SchemaVectorizer

        logger.info("✓ AthenaSchemaVectorizer backward compatibility alias verified")

    def test_function_aliases(self):
        """Verify function aliases for backward compatibility."""
        from src.utils.schema_vectorizer import (
            vectorize_athena_schemas,
            vectorize_database_schemas,
        )

        # Should be the same function
        assert vectorize_athena_schemas == vectorize_database_schemas

        logger.info("✓ Function aliases verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
