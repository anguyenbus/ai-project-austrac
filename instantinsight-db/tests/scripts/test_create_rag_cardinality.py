#!/usr/bin/env python3
"""Test suite for create_rag_cardinality.py script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.create_rag_cardinality import ConfigBasedProcessor


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "columns": [
            {"table": "test_table", "column": "test_column"},
            {"table": "another_table", "column": "special/column"},
        ]
    }


@pytest.fixture
def mock_athena_client():
    """Mock Athena client."""
    mock_client = MagicMock()

    # Mock successful query execution
    mock_client.start_query_execution.return_value = {
        "QueryExecutionId": "test-query-id-123"
    }

    # Mock query status - succeeded
    mock_client.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }

    # Mock query results
    mock_client.get_query_results.return_value = {
        "ResultSet": {
            "Rows": [
                {
                    "Data": [{"VarCharValue": "value"}, {"VarCharValue": "frequency"}]
                },  # Header
                {"Data": [{"VarCharValue": "TestValue1"}, {"VarCharValue": "100"}]},
                {"Data": [{"VarCharValue": "TestValue2"}, {"VarCharValue": "50"}]},
            ]
        }
    }

    return mock_client


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Configure cursor context manager
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=None)

    # Configure connection cursor method
    mock_conn.cursor.return_value = mock_cursor

    # Mock fetchone for _register_column
    mock_cursor.fetchone.return_value = {"id": 1}

    # Mock fetchall for _generate_embeddings
    mock_cursor.fetchall.return_value = [
        {"id": 1, "category": "TestValue1", "category_norm": "testvalue1"},
        {"id": 2, "category": "TestValue2", "category_norm": "testvalue2"},
    ]

    return mock_conn, mock_cursor


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    mock_embed = MagicMock()
    mock_embed.embed_documents.return_value = [
        [0.1, 0.2, 0.3],  # Embedding for TestValue1
        [0.4, 0.5, 0.6],  # Embedding for TestValue2
    ]
    return mock_embed


class TestConfigBasedProcessor:
    """Test suite for ConfigBasedProcessor class."""

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_init(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        tmp_path,
    ):
        """Test processor initialization."""
        # Create temporary config file
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))

        # Initialize processor
        processor = ConfigBasedProcessor(str(config_path))

        # Assertions
        assert processor.config == sample_config
        assert processor.database_name in ["text_to_sql", "test_database"]
        mock_boto_client.assert_called_once_with("athena")
        mock_psycopg_connect.assert_called_once()
        mock_bedrock.assert_called_once()
        mock_register.assert_called_once()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_load_config_missing_columns(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        tmp_path,
    ):
        """Test configuration loading with missing columns section."""
        # Create invalid config
        invalid_config = {"not_columns": []}
        config_path = tmp_path / "invalid_config.yaml"
        config_path.write_text(yaml.dump(invalid_config))

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Configuration must contain 'columns' section"
        ):
            ConfigBasedProcessor(str(config_path))

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_register_column(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_db_connection,
        tmp_path,
    ):
        """Test column registration in database."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_conn, mock_cursor = mock_db_connection
        mock_psycopg_connect.return_value = mock_conn

        processor = ConfigBasedProcessor(str(config_path))

        # Test registration
        column_id = processor._register_column("test_table", "test_column")

        # Assertions
        assert column_id == 1
        mock_cursor.execute.assert_called_once()
        sql_call = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO rag_cardinality_columns" in sql_call
        assert "ON CONFLICT" in sql_call
        mock_conn.commit.assert_called()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    @patch("scripts.create_rag_cardinality.time.sleep")
    def test_extract_values(
        self,
        mock_sleep,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_athena_client,
        tmp_path,
    ):
        """Test extracting values from Athena."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_boto_client.return_value = mock_athena_client

        processor = ConfigBasedProcessor(str(config_path))

        # Test extraction
        values = processor._extract_values("test_table", "test_column")

        # Assertions
        assert len(values) == 2
        assert values[0] == {"value": "TestValue1", "frequency": 100}
        assert values[1] == {"value": "TestValue2", "frequency": 50}
        mock_athena_client.start_query_execution.assert_called_once()
        mock_athena_client.get_query_execution.assert_called()
        mock_athena_client.get_query_results.assert_called()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    @patch("scripts.create_rag_cardinality.time.sleep")
    def test_extract_values_with_special_column(
        self,
        mock_sleep,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_athena_client,
        tmp_path,
    ):
        """Test extracting values with special characters in column name."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_boto_client.return_value = mock_athena_client

        processor = ConfigBasedProcessor(str(config_path))

        # Test extraction with special column
        values = processor._extract_values("test_table", "city/town")

        # Verify column is properly quoted
        query_call = mock_athena_client.start_query_execution.call_args[1][
            "QueryString"
        ]
        assert '"city/town"' in query_call
        assert len(values) == 2

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    @patch("scripts.create_rag_cardinality.time.sleep")
    def test_extract_values_query_failed(
        self,
        mock_sleep,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        tmp_path,
    ):
        """Test handling of failed Athena query."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))

        mock_client = MagicMock()
        mock_client.start_query_execution.return_value = {
            "QueryExecutionId": "failed-query-id"
        }
        mock_client.get_query_execution.return_value = {
            "QueryExecution": {
                "Status": {"State": "FAILED", "StateChangeReason": "Query error"}
            }
        }
        mock_boto_client.return_value = mock_client

        processor = ConfigBasedProcessor(str(config_path))

        # Test extraction
        values = processor._extract_values("test_table", "test_column")

        # Should return empty list on failure
        assert values == []

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_store_values(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_db_connection,
        tmp_path,
    ):
        """Test storing values in database."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_conn, mock_cursor = mock_db_connection
        mock_psycopg_connect.return_value = mock_conn

        processor = ConfigBasedProcessor(str(config_path))

        # Test data
        values = [
            {"value": "TestValue1", "frequency": 100},
            {"value": "TestValue2", "frequency": 50},
        ]

        # Store values
        processor._store_values(1, "test_table", "test_column", values)

        # Assertions
        assert mock_cursor.execute.call_count == 2
        for call_args in mock_cursor.execute.call_args_list:
            sql = call_args[0][0]
            assert "INSERT INTO rag_cardinality" in sql
            assert "ON CONFLICT" in sql
        mock_conn.commit.assert_called()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    @patch("scripts.create_rag_cardinality.time.sleep")
    def test_generate_embeddings(
        self,
        mock_sleep,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_db_connection,
        mock_embeddings,
        tmp_path,
    ):
        """Test embedding generation."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_conn, mock_cursor = mock_db_connection
        mock_psycopg_connect.return_value = mock_conn
        mock_bedrock.return_value = mock_embeddings

        processor = ConfigBasedProcessor(str(config_path))

        # Generate embeddings
        processor._generate_embeddings([])

        # Assertions
        mock_cursor.execute.assert_called()

        # Check SELECT query
        select_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "SELECT" in call[0][0]
        ]
        assert len(select_calls) == 1
        assert "embedding_status = 'pending'" in select_calls[0][0][0]

        # Check UPDATE queries
        update_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "UPDATE" in call[0][0]
        ]
        assert len(update_calls) == 2  # One for each value

        mock_embeddings.embed_documents.assert_called_once_with(
            ["TestValue1", "TestValue2"]
        )
        mock_conn.commit.assert_called()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_generate_embeddings_no_pending(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        tmp_path,
    ):
        """Test embedding generation when no pending values."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_cursor.fetchall.return_value = []  # No pending values
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg_connect.return_value = mock_conn

        processor = ConfigBasedProcessor(str(config_path))

        # Generate embeddings
        processor._generate_embeddings([])

        # Should not call embed_documents if no pending values
        mock_bedrock.return_value.embed_documents.assert_not_called()

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_process_all(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_athena_client,
        mock_db_connection,
        tmp_path,
    ):
        """Test full processing of all columns."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_conn, mock_cursor = mock_db_connection
        mock_psycopg_connect.return_value = mock_conn
        mock_boto_client.return_value = mock_athena_client

        processor = ConfigBasedProcessor(str(config_path))

        # Mock methods
        with (
            patch.object(processor, "_register_column", return_value=1) as mock_reg,
            patch.object(
                processor,
                "_extract_values",
                return_value=[{"value": "test", "frequency": 10}],
            ) as mock_extract,
            patch.object(processor, "_store_values") as mock_store,
            patch.object(processor, "_generate_embeddings") as mock_gen,
        ):
            processor.process_all()

            # Should process both columns
            assert mock_reg.call_count == 2
            assert mock_extract.call_count == 2
            assert mock_store.call_count == 2
            assert mock_gen.call_count == 2

            # Check calls
            mock_reg.assert_any_call("test_table", "test_column")
            mock_reg.assert_any_call("another_table", "special/column")

    @patch("scripts.create_rag_cardinality.psycopg.connect")
    @patch("scripts.create_rag_cardinality.boto3.client")
    @patch("scripts.create_rag_cardinality.BedrockEmbeddings")
    @patch("scripts.create_rag_cardinality.register_vector")
    def test_close(
        self,
        mock_register,
        mock_bedrock,
        mock_boto_client,
        mock_psycopg_connect,
        sample_config,
        mock_db_connection,
        tmp_path,
    ):
        """Test closing database connection."""
        # Setup
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(sample_config))
        mock_conn, _ = mock_db_connection
        mock_psycopg_connect.return_value = mock_conn

        processor = ConfigBasedProcessor(str(config_path))

        # Close connection
        processor.close()

        # Assert connection closed
        mock_conn.close.assert_called_once()


class TestMainFunction:
    """Test suite for main function."""

    @patch("scripts.create_rag_cardinality.ConfigBasedProcessor")
    @patch("scripts.create_rag_cardinality.os.path.exists")
    def test_main_all_mode(self, mock_exists, mock_processor_class, tmp_path):
        """Test main function in 'all' mode."""
        # Setup
        mock_exists.return_value = True
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump({"columns": []}))

        # Test with 'all' mode
        test_args = ["script", "--config", str(config_path), "--mode", "all"]
        with patch("sys.argv", test_args):
            from scripts.create_rag_cardinality import main

            # Should not raise exception
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

            mock_processor.process_all.assert_called_once()
            mock_processor.close.assert_called_once()

    @patch("scripts.create_rag_cardinality.ConfigBasedProcessor")
    @patch("scripts.create_rag_cardinality.os.path.exists")
    def test_main_embed_mode(self, mock_exists, mock_processor_class, tmp_path):
        """Test main function in 'embed' mode."""
        # Setup
        mock_exists.return_value = True
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump({"columns": []}))

        # Test with 'embed' mode
        test_args = ["script", "--config", str(config_path), "--mode", "embed"]
        with patch("sys.argv", test_args):
            from scripts.create_rag_cardinality import main

            # Should not raise exception
            try:
                main()
            except SystemExit as e:
                assert e.code == 0 or e.code is None

            mock_processor._generate_embeddings.assert_called_once_with([])
            mock_processor.close.assert_called_once()

    @patch("scripts.create_rag_cardinality.os.path.exists")
    def test_main_config_not_found(self, mock_exists):
        """Test main function with non-existent config file."""
        # Setup
        mock_exists.return_value = False

        # Test with non-existent config
        test_args = ["script", "--config", "/non/existent/config.yaml"]
        with patch("sys.argv", test_args):
            from scripts.create_rag_cardinality import main

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch("scripts.create_rag_cardinality.ConfigBasedProcessor")
    @patch("scripts.create_rag_cardinality.os.path.exists")
    def test_main_processor_exception(
        self, mock_exists, mock_processor_class, tmp_path
    ):
        """Test main function handling processor exceptions."""
        # Setup
        mock_exists.return_value = True
        mock_processor = MagicMock()
        mock_processor.process_all.side_effect = Exception("Test error")
        mock_processor_class.return_value = mock_processor

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump({"columns": []}))

        # Test with exception
        test_args = ["script", "--config", str(config_path)]
        with patch("sys.argv", test_args):
            from scripts.create_rag_cardinality import main

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            mock_processor.close.assert_called_once()
