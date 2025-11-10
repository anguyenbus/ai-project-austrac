"""Tests for SchemaIntrospector universal schema extraction."""

from unittest import mock
from unittest.mock import Mock, patch

import pytest

from src.connectors.analytics_backend import AnalyticsConnector
from src.utils.schema_introspector import SchemaIntrospector, create_schema_introspector


class TestSchemaIntrospector:
    """Test cases for SchemaIntrospector class."""

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_init_success(self, mock_analytics_backend):
        """Test successful initialization."""
        mock_backend = Mock(spec=AnalyticsConnector)
        mock_backend.backend_type = "postgres"
        mock_analytics_backend.return_value = mock_backend

        # Temporarily disable beartype checking for this test
        with patch("src.utils.schema_introspector.beartype"):
            from src.utils.schema_introspector import SchemaIntrospector

            introspector = SchemaIntrospector(mock_backend)

            assert introspector.backend == mock_backend
            assert introspector.glue_enricher is None
            assert introspector.schema_cache == {}

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_init_with_glue_enricher(self, mock_analytics_backend):
        """Test initialization with Glue enricher."""
        from src.utils.glue_enricher import GlueMetadataEnricher

        mock_backend = Mock(spec=AnalyticsConnector)
        mock_backend.backend_type = "athena"
        mock_analytics_backend.return_value = mock_backend
        mock_enricher = Mock(spec=GlueMetadataEnricher)

        # Temporarily disable beartype checking for this test
        with patch("src.utils.schema_introspector.beartype"):
            from src.utils.schema_introspector import SchemaIntrospector

            introspector = SchemaIntrospector(mock_backend, mock_enricher)

            assert introspector.backend == mock_backend
            assert introspector.glue_enricher == mock_enricher
            assert introspector.schema_cache == {}

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_get_available_databases_success(self, mock_analytics_backend):
        """Test successful database listing."""
        mock_backend = Mock()
        mock_backend.list_databases.return_value = ["db1", "db2", "db3"]
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)
        databases = introspector.get_available_databases()

        assert databases == ["db1", "db2", "db3"]
        mock_backend.list_databases.assert_called_once()

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_get_available_databases_failure(self, mock_analytics_backend):
        """Test database listing failure."""
        mock_backend = Mock()
        mock_backend.list_databases.side_effect = Exception("List failed")
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)
        databases = introspector.get_available_databases()

        assert databases == []

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_extract_database_schema_cached(self, mock_analytics_backend):
        """Test schema extraction with caching."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)

        # Set up cache
        cached_schema = {
            "database_name": "testdb",
            "tables": {"users": {"table_name": "users"}},
            "table_count": 1,
        }
        introspector.schema_cache["testdb"] = cached_schema

        # Extract should return cached version
        result = introspector.extract_database_schema("testdb")

        assert result == cached_schema
        mock_backend.list_tables.assert_not_called()

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_extract_database_schema_success(self, mock_analytics_backend):
        """Test successful schema extraction."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_backend.list_tables.return_value = ["users", "orders"]
        mock_backend.get_table_schema.side_effect = [
            {
                "columns": [
                    {"name": "id", "type": "int64"},
                    {"name": "name", "type": "string"},
                ],
                "column_count": 2,
            },
            {
                "columns": [
                    {"name": "id", "type": "int64"},
                    {"name": "user_id", "type": "int64"},
                    {"name": "amount", "type": "float64"},
                ],
                "column_count": 3,
            },
        ]
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)
        result = introspector.extract_database_schema("testdb")

        expected = {
            "database_name": "testdb",
            "tables": {
                "users": {
                    "table_name": "users",
                    "database_name": "testdb",
                    "columns": [
                        {"name": "id", "type": "int64"},
                        {"name": "name", "type": "string"},
                    ],
                    "column_count": 2,
                },
                "orders": {
                    "table_name": "orders",
                    "database_name": "testdb",
                    "columns": [
                        {"name": "id", "type": "int64"},
                        {"name": "user_id", "type": "int64"},
                        {"name": "amount", "type": "float64"},
                    ],
                    "column_count": 3,
                },
            },
            "table_count": 2,
            "extracted_at": mock.ANY,  # We don't care about exact timestamp
            "source": "postgres",
        }

        # Check structure without comparing timestamp
        assert result["database_name"] == expected["database_name"]
        assert result["tables"] == expected["tables"]
        assert result["table_count"] == expected["table_count"]
        assert result["source"] == expected["source"]
        assert "extracted_at" in result

        # Verify backend calls
        mock_backend.list_tables.assert_called_once_with(database="testdb")
        assert mock_backend.get_table_schema.call_count == 2

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_extract_database_schema_no_tables(self, mock_analytics_backend):
        """Test schema extraction with no tables."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_backend.list_tables.return_value = []
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)
        result = introspector.extract_database_schema("emptydb")

        expected = {
            "database_name": "emptydb",
            "tables": {},
            "table_count": 0,
            "extracted_at": mock.ANY,
            "source": "postgres",
        }

        assert result["database_name"] == expected["database_name"]
        assert result["tables"] == expected["tables"]
        assert result["table_count"] == expected["table_count"]
        assert result["source"] == expected["source"]

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    @patch("src.utils.schema_introspector.GlueMetadataEnricher")
    def test_extract_database_schema_with_glue_enrichment(
        self, mock_glue_enricher, mock_analytics_backend
    ):
        """Test schema extraction with Glue enrichment."""
        mock_backend = Mock()
        mock_backend.backend_type = "athena"
        mock_backend.list_tables.return_value = ["users"]
        mock_backend.get_table_schema.return_value = {
            "columns": [{"name": "id", "type": "int64"}],
            "column_count": 1,
        }
        mock_analytics_backend.return_value = mock_backend

        mock_enricher = Mock()
        enriched_schema = {
            "database_name": "testdb",
            "tables": {
                "users": {
                    "table_name": "users",
                    "database_name": "testdb",
                    "columns": [{"name": "id", "type": "int64"}],
                    "column_count": 1,
                    "s3_location": "s3://bucket/path/",
                }
            },
            "table_count": 1,
        }
        mock_enricher.enrich.return_value = enriched_schema
        mock_glue_enricher.return_value = mock_enricher

        introspector = SchemaIntrospector(mock_backend, mock_enricher)
        result = introspector.extract_database_schema("testdb")

        mock_enricher.enrich.assert_called_once()
        assert result == enriched_schema

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_extract_multiple_databases(self, mock_analytics_backend):
        """Test extracting multiple databases."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_backend.list_tables.return_value = ["users"]
        mock_backend.get_table_schema.return_value = {
            "columns": [{"name": "id", "type": "int64"}],
            "column_count": 1,
        }
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)
        result = introspector.extract_multiple_databases(["db1", "db2"])

        assert len(result) == 2
        assert "db1" in result
        assert "db2" in result
        assert mock_backend.list_tables.call_count == 2

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_get_schema_statistics(self, mock_analytics_backend):
        """Test schema statistics calculation."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)

        # Set up cached schema
        cached_schema = {
            "database_name": "testdb",
            "tables": {
                "users": {
                    "columns": [
                        {"name": "id", "type": "int64"},
                        {"name": "name", "type": "string"},
                    ],
                    "column_count": 2,
                },
                "orders": {
                    "columns": [
                        {"name": "id", "type": "int64"},
                        {"name": "amount", "type": "float64"},
                    ],
                    "column_count": 2,
                },
            },
            "table_count": 2,
            "source": "postgres",
            "extracted_at": "2025-01-25T10:00:00",
        }
        introspector.schema_cache["testdb"] = cached_schema

        stats = introspector.get_schema_statistics("testdb")

        expected = {
            "database_name": "testdb",
            "table_count": 2,
            "total_columns": 4,
            "column_types": {"int64": 3, "string": 1},
            "backend_type": "postgres",
            "extracted_at": "2025-01-25T10:00:00",
        }

        assert stats == expected

    def test_clear_cache_specific_database(self):
        """Test clearing cache for specific database."""
        mock_backend = Mock()
        introspector = SchemaIntrospector(mock_backend)

        # Set up cache
        introspector.schema_cache = {
            "db1": {"data": "test1"},
            "db2": {"data": "test2"},
        }

        introspector.clear_cache("db1")

        assert "db1" not in introspector.schema_cache
        assert "db2" in introspector.schema_cache

    def test_clear_cache_all(self):
        """Test clearing all cache."""
        mock_backend = Mock()
        introspector = SchemaIntrospector(mock_backend)

        # Set up cache
        introspector.schema_cache = {
            "db1": {"data": "test1"},
            "db2": {"data": "test2"},
        }

        introspector.clear_cache()

        assert len(introspector.schema_cache) == 0

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_save_schema_to_file_success(
        self, mock_json_dump, mock_open, mock_analytics_backend
    ):
        """Test successful schema saving to file."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"
        mock_backend.list_tables.return_value = ["users"]
        mock_backend.get_table_schema.return_value = {
            "columns": [{"name": "id", "type": "int64"}],
            "column_count": 1,
        }
        mock_analytics_backend.return_value = mock_backend

        introspector = SchemaIntrospector(mock_backend)

        from pathlib import Path

        result = introspector.save_schema_to_file("testdb", Path("/tmp/schema.json"))

        assert result is True
        mock_json_dump.assert_called_once()

    def test_save_schema_to_file_no_schema(self):
        """Test saving when no schema is available."""
        mock_backend = Mock()
        mock_backend.list_tables.return_value = []
        mock_backend.get_table_schema.return_value = {
            "columns": [],
            "column_count": 0,
        }

        introspector = SchemaIntrospector(mock_backend)

        from pathlib import Path

        result = introspector.save_schema_to_file("emptydb", Path("/tmp/schema.json"))

        assert result is False

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_context_manager(self, mock_analytics_backend):
        """Test context manager functionality."""
        mock_backend = Mock()
        mock_analytics_backend.return_value = mock_backend

        with SchemaIntrospector(mock_backend) as introspector:
            assert introspector.backend == mock_backend

        mock_backend.close.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        mock_backend = Mock()
        mock_backend.backend_type = "postgres"

        introspector = SchemaIntrospector(mock_backend)
        assert repr(introspector) == "SchemaIntrospector(backend='postgres')"


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    @patch("src.utils.schema_introspector.GlueMetadataEnricher")
    def test_create_schema_introspector_with_glue(
        self, mock_glue_enricher, mock_analytics_backend
    ):
        """Test creating introspector with Glue enrichment."""
        mock_backend = Mock()
        mock_analytics_backend.return_value = mock_backend
        mock_enricher = Mock()
        mock_glue_enricher.return_value = mock_enricher

        result = create_schema_introspector(
            connection_string="athena://awsdatacatalog?region=ap-southeast-2&database=testdb",
            enable_glue_enrichment=True,
            aws_profile="default",
            aws_region="ap-southeast-2",
        )

        mock_analytics_backend.assert_called_once_with(
            "athena://awsdatacatalog?region=ap-southeast-2&database=testdb"
        )
        mock_glue_enricher.assert_called_once_with("default", "ap-southeast-2")
        assert result.backend == mock_backend
        assert result.glue_enricher == mock_enricher

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_create_schema_introspector_without_glue(self, mock_analytics_backend):
        """Test creating introspector without Glue enrichment."""
        mock_backend = Mock()
        mock_analytics_backend.return_value = mock_backend

        result = create_schema_introspector(
            connection_string="postgres://user:pass@host:5432/db",
            enable_glue_enrichment=False,
        )

        mock_analytics_backend.assert_called_once_with(
            "postgres://user:pass@host:5432/db"
        )
        assert result.backend == mock_backend
        assert result.glue_enricher is None

    @patch("src.utils.schema_introspector.AnalyticsConnector")
    def test_create_schema_introspector_athena_no_glue_params(
        self, mock_analytics_backend
    ):
        """Test creating introspector for Athena without Glue parameters."""
        mock_backend = Mock()
        mock_analytics_backend.return_value = mock_backend

        result = create_schema_introspector(
            connection_string="athena://awsdatacatalog?region=ap-southeast-2&database=testdb",
            enable_glue_enrichment=True,
            aws_profile=None,  # Missing parameter
            aws_region="ap-southeast-2",
        )

        # Should not create Glue enricher due to missing profile
        assert result.glue_enricher is None


class TestIntegration:
    """Integration tests (require actual database connections)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration")
        if hasattr(pytest, "config")
        else True,
        reason="Integration tests require --run-integration flag",
    )
    def test_real_postgres_schema_extraction(self):
        """Test schema extraction with real PostgreSQL."""
        # This test requires a running PostgreSQL instance
        # Run with: pytest --run-integration
        try:
            backend = AnalyticsConnector(
                "postgres://postgres:postgres@localhost:5432/testdb"
            )
            introspector = SchemaIntrospector(backend)

            databases = introspector.get_available_databases()
            assert isinstance(databases, list)

            if databases:
                schema = introspector.extract_database_schema(databases[0])
                assert "tables" in schema
                assert "table_count" in schema

            backend.close()
        except Exception:
            pytest.skip("PostgreSQL not available for integration testing")


if __name__ == "__main__":
    pytest.main([__file__])
