"""Tests for GlueMetadataEnricher Athena metadata enrichment."""

from unittest.mock import Mock, patch

import pytest

from src.utils.glue_enricher import GlueMetadataEnricher, create_glue_enricher_if_needed


class TestGlueMetadataEnricher:
    """Test cases for GlueMetadataEnricher class."""

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_init_success(self, mock_boto3_session):
        """Test successful initialization."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")

        mock_boto3_session.assert_called_once_with(
            profile_name="test-profile", region_name="ap-southeast-2"
        )
        mock_session.client.assert_called_once_with("glue")
        assert enricher.glue_client == mock_glue_client
        assert enricher.aws_profile == "test-profile"
        assert enricher.aws_region == "ap-southeast-2"

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_init_failure(self, mock_boto3_session):
        """Test initialization failure."""
        mock_session = Mock()
        mock_session.side_effect = Exception("AWS credentials not found")
        mock_boto3_session.return_value = mock_session

        with pytest.raises(RuntimeError, match="Glue enricher initialization failed"):
            GlueMetadataEnricher("invalid-profile", "ap-southeast-2")

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_enrich_success(self, mock_boto3_session):
        """Test successful schema enrichment."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        # Mock Glue table response
        mock_glue_client.get_table.side_effect = [
            {
                "Table": {
                    "Name": "users",
                    "Description": "User accounts table",
                    "TableType": "EXTERNAL_TABLE",
                    "CreateTime": "2025-01-01T00:00:00Z",
                    "LastAnalyzedTime": "2025-01-25T10:00:00Z",
                    "Retention": 0,
                    "StorageDescriptor": {
                        "Location": "s3://bucket/path/users/",
                        "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                        "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                        "Compressed": False,
                        "SerdeInfo": {
                            "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
                        },
                        "Columns": [
                            {"Name": "id", "Type": "bigint", "Comment": "User ID"},
                            {"Name": "name", "Type": "string", "Comment": "User name"},
                        ],
                    },
                    "PartitionKeys": [
                        {
                            "Name": "created_date",
                            "Type": "date",
                            "Comment": "Partition by creation date",
                        }
                    ],
                    "Parameters": {
                        "classification": "parquet",
                        "compressionType": "snappy",
                    },
                }
            }
        ]

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")

        # Input schema data
        schema_data = {
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
                }
            },
            "table_count": 1,
            "source": "athena",
        }

        result = enricher.enrich(schema_data)

        # Verify enrichment
        enriched_users = result["tables"]["users"]
        assert enriched_users["s3_location"] == "s3://bucket/path/users/"
        assert enriched_users["partition_keys"] == [
            {
                "Name": "created_date",
                "Type": "date",
                "Comment": "Partition by creation date",
            }
        ]
        assert enriched_users["serde_info"] == {
            "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
        }
        assert enriched_users["description"] == "User accounts table"
        assert enriched_users["table_type"] == "EXTERNAL_TABLE"
        assert enriched_users["creation_time"] == "2025-01-01T00:00:00Z"
        assert enriched_users["last_analyzed_time"] == "2025-01-25T10:00:00Z"
        assert enriched_users["retention"] == 0
        assert (
            enriched_users["input_format"]
            == "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
        )
        assert (
            enriched_users["output_format"]
            == "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
        )
        assert enriched_users["compressed"] is False
        assert enriched_users["parameters"] == {
            "classification": "parquet",
            "compressionType": "snappy",
        }

        # Verify column enrichment
        columns = enriched_users["columns"]
        id_col = next(col for col in columns if col["name"] == "id")
        name_col = next(col for col in columns if col["name"] == "name")

        assert id_col["glue_comment"] == "User ID"
        assert id_col["glue_type"] == "bigint"
        assert name_col["glue_comment"] == "User name"
        assert name_col["glue_type"] == "string"

        # Verify enrichment metadata
        assert "glue_enrichment" in result
        assert result["glue_enrichment"]["enabled"] is True
        assert result["glue_enrichment"]["enriched_tables"] == 1
        assert result["glue_enrichment"]["total_tables"] == 1

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_enrich_no_database_name(self, mock_boto3_session):
        """Test enrichment with no database name."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")

        schema_data = {"tables": {}}  # No database_name

        result = enricher.enrich(schema_data)

        # Should return unchanged schema
        assert result == schema_data

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_enrich_table_not_found(self, mock_boto3_session):
        """Test enrichment when table not found in Glue."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        # Mock Glue table not found
        mock_glue_client.get_table.side_effect = Exception("Table not found")

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")

        schema_data = {
            "database_name": "testdb",
            "tables": {
                "users": {
                    "table_name": "users",
                    "columns": [{"name": "id", "type": "int64"}],
                    "column_count": 1,
                }
            },
            "table_count": 1,
        }

        result = enricher.enrich(schema_data)

        # Should skip enrichment for missing table but continue with others
        assert result["tables"]["users"]["table_name"] == "users"
        assert "glue_enrichment" in result
        assert result["glue_enrichment"]["enriched_tables"] == 0

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_get_table_location_success(self, mock_boto3_session):
        """Test successful table location retrieval."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.return_value = {
            "Table": {"StorageDescriptor": {"Location": "s3://bucket/path/table/"}}
        }

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        location = enricher.get_table_location("testdb", "users")

        assert location == "s3://bucket/path/table/"
        mock_glue_client.get_table.assert_called_once_with(
            DatabaseName="testdb", Name="users"
        )

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_get_table_location_failure(self, mock_boto3_session):
        """Test table location retrieval failure."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.side_effect = Exception("Access denied")

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        location = enricher.get_table_location("testdb", "users")

        assert location == ""

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_get_partition_keys_success(self, mock_boto3_session):
        """Test successful partition keys retrieval."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.return_value = {
            "Table": {
                "PartitionKeys": [
                    {"Name": "created_date", "Type": "date"},
                    {"Name": "region", "Type": "string"},
                ]
            }
        }

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        partition_keys = enricher.get_partition_keys("testdb", "users")

        expected = [
            {"Name": "created_date", "Type": "date"},
            {"Name": "region", "Type": "string"},
        ]
        assert partition_keys == expected

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_get_partition_keys_failure(self, mock_boto3_session):
        """Test partition keys retrieval failure."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.side_effect = Exception("Access denied")

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        partition_keys = enricher.get_partition_keys("testdb", "users")

        assert partition_keys == []

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_test_glue_access_success(self, mock_boto3_session):
        """Test successful Glue access test."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_database.return_value = {"Name": "testdb"}

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        result = enricher.test_glue_access("testdb")

        assert result is True
        mock_glue_client.get_database.assert_called_once_with(Name="testdb")

    @patch("src.utils.glue_enricher.boto3.Session")
    def test_test_glue_access_failure(self, mock_boto3_session):
        """Test Glue access test failure."""
        mock_session = Mock()
        mock_glue_client = Mock()
        mock_session.return_value = mock_session
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_database.side_effect = Exception("Access denied")

        enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
        result = enricher.test_glue_access("testdb")

        assert result is False

    def test_repr(self):
        """Test string representation."""
        with patch("src.utils.glue_enricher.boto3.Session"):
            enricher = GlueMetadataEnricher("test-profile", "ap-southeast-2")
            assert (
                repr(enricher)
                == "GlueMetadataEnricher(profile='test-profile', region='ap-southeast-2')"
            )


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @patch("src.utils.glue_enricher.GlueMetadataEnricher")
    def test_create_glue_enricher_if_needed_athena(self, mock_glue_enricher):
        """Test creating Glue enricher for Athena backend."""
        mock_enricher = Mock()
        mock_glue_enricher.return_value = mock_enricher

        result = create_glue_enricher_if_needed(
            connection_string="athena://awsdatacatalog?region=ap-southeast-2&database=testdb",
            aws_profile="default",
            aws_region="ap-southeast-2",
        )

        mock_glue_enricher.assert_called_once_with("default", "ap-southeast-2")
        assert result == mock_enricher

    @patch("src.utils.glue_enricher.GlueMetadataEnricher")
    def test_create_glue_enricher_if_needed_postgres(self, mock_glue_enricher):
        """Test not creating Glue enricher for PostgreSQL backend."""
        result = create_glue_enricher_if_needed(
            connection_string="postgres://user:pass@host:5432/db",
            aws_profile="default",
            aws_region="ap-southeast-2",
        )

        mock_glue_enricher.assert_not_called()
        assert result is None

    @patch("src.utils.glue_enricher.GlueMetadataEnricher")
    def test_create_glue_enricher_if_needed_missing_params(self, mock_glue_enricher):
        """Test not creating Glue enricher when parameters missing."""
        result = create_glue_enricher_if_needed(
            connection_string="athena://awsdatacatalog?region=ap-southeast-2&database=testdb",
            aws_profile=None,  # Missing
            aws_region="ap-southeast-2",
        )

        mock_glue_enricher.assert_not_called()
        assert result is None

    @patch("src.utils.glue_enricher.GlueMetadataEnricher")
    def test_create_glue_enricher_if_needed_creation_failure(self, mock_glue_enricher):
        """Test handling Glue enricher creation failure."""
        mock_glue_enricher.side_effect = Exception("AWS credentials invalid")

        result = create_glue_enricher_if_needed(
            connection_string="athena://awsdatacatalog?region=ap-southeast-2&database=testdb",
            aws_profile="invalid",
            aws_region="ap-southeast-2",
        )

        mock_glue_enricher.assert_called_once_with("invalid", "ap-southeast-2")
        assert result is None


class TestIntegration:
    """Integration tests (require AWS credentials)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration")
        if hasattr(pytest, "config")
        else True,
        reason="Integration tests require --run-integration flag and AWS credentials",
    )
    def test_real_glue_access(self):
        """Test with real AWS Glue access."""
        # This test requires AWS credentials
        # Run with: pytest --run-integration
        try:
            enricher = GlueMetadataEnricher("default", "ap-southeast-2")

            # Test access to a common database (may not exist)
            result = enricher.test_glue_access("nonexistent_db")

            # Should either succeed or fail gracefully
            assert isinstance(result, bool)

        except Exception as e:
            pytest.skip(f"AWS credentials not available: {e}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration")
        if hasattr(pytest, "config")
        else True,
        reason="Integration tests require --run-integration flag and AWS credentials",
    )
    def test_real_table_enrichment(self):
        """Test enrichment with real Glue table."""
        # This test requires AWS credentials and existing Glue tables
        # Run with: pytest --run-integration
        try:
            enricher = GlueMetadataEnricher("default", "ap-southeast-2")

            # Try to enrich a schema (will likely fail for non-existent tables)
            schema_data = {
                "database_name": "testdb",
                "tables": {
                    "test_table": {
                        "table_name": "test_table",
                        "columns": [{"name": "id", "type": "int64"}],
                        "column_count": 1,
                    }
                },
                "table_count": 1,
            }

            result = enricher.enrich(schema_data)

            # Should have enrichment metadata even if table not found
            assert "glue_enrichment" in result
            assert result["glue_enrichment"]["enabled"] is True

        except Exception as e:
            pytest.skip(f"AWS credentials not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
