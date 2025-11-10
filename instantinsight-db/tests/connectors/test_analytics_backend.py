"""Tests for AnalyticsConnector universal database abstraction."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.connectors.analytics_backend import AnalyticsConnector, create_postgres_backend


class TestAnalyticsConnector:
    """Test cases for AnalyticsConnector class."""

    def test_backend_type_detection(self):
        """Test automatic backend type detection."""
        # Test Athena
        assert AnalyticsConnector._detect_backend_type("athena://catalog") == "athena"
        assert (
            AnalyticsConnector._detect_backend_type(
                "athena://awsdatacatalog?region=ap-southeast-2"
            )
            == "athena"
        )

        # Test PostgreSQL
        assert (
            AnalyticsConnector._detect_backend_type("postgres://host/db") == "postgres"
        )
        assert (
            AnalyticsConnector._detect_backend_type(
                "postgresql://user:pass@host:5432/db"
            )
            == "postgres"
        )

        # Test Snowflake
        assert (
            AnalyticsConnector._detect_backend_type("snowflake://account/db")
            == "snowflake"
        )

        # Test BigQuery
        assert (
            AnalyticsConnector._detect_backend_type("bigquery://project/dataset")
            == "bigquery"
        )

        # Test edge cases
        assert AnalyticsConnector._detect_backend_type("unknown://test") == "unknown"
        assert AnalyticsConnector._detect_backend_type("invalid_string") == "unknown"

    def test_read_only_validation(self):
        """Test dangerous SQL operation detection."""
        # Create a backend instance to test the method
        with patch("ibis.connect") as mock_connect:
            mock_connect.return_value = Mock()
            backend = AnalyticsConnector("postgres://test")

            # Safe queries
            assert backend._is_read_only("SELECT * FROM table")
            assert backend._is_read_only("SELECT COUNT(*) FROM table WHERE id > 10")
            assert backend._is_read_only(
                "SELECT col1, col2 FROM table JOIN other_table ON table.id = other_table.id"
            )
            assert backend._is_read_only(
                "WITH cte AS (SELECT * FROM table) SELECT * FROM cte"
            )

            # Dangerous queries
            assert not backend._is_read_only("DROP TABLE table")
            assert not backend._is_read_only("CREATE TABLE new_table (id INT)")
            assert not backend._is_read_only("ALTER TABLE table ADD COLUMN new_col INT")
            assert not backend._is_read_only("TRUNCATE TABLE table")
            assert not backend._is_read_only("DELETE FROM table WHERE id = 1")
            assert not backend._is_read_only("INSERT INTO table VALUES (1)")
            assert not backend._is_read_only("UPDATE table SET col = 1")
            assert not backend._is_read_only("GRANT SELECT ON table TO user")
            assert not backend._is_read_only("REVOKE SELECT ON table FROM user")

            # Case insensitive
            assert not backend._is_read_only("drop table table")
            assert not backend._is_read_only("Create Table new_table (id INT)")

    @patch("ibis.connect")
    def test_init_success(self, mock_connect):
        """Test successful initialization."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")

        assert backend.connection_string == "postgres://user:pass@host:5432/db"
        assert backend.backend_type == "postgres"
        assert backend._conn == mock_conn
        mock_connect.assert_called_once_with("postgres://user:pass@host:5432/db")

    @patch("ibis.connect")
    def test_init_connection_failure(self, mock_connect):
        """Test initialization with connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(ConnectionError, match="Connection failed"):
            AnalyticsConnector("postgres://invalid")

    def test_init_invalid_connection_string(self):
        """Test initialization with invalid connection string."""
        # NOTE: icontract raises ViolationError instead of ValueError
        with pytest.raises(Exception) as exc_info:
            AnalyticsConnector("invalid_string_without_protocol")
        assert "Connection string must contain '://'" in str(exc_info.value)

    @patch("ibis.connect")
    def test_list_databases_success(self, mock_connect):
        """Test successful database listing."""
        mock_conn = Mock()
        mock_conn.list_databases.return_value = ["db1", "db2", "db3"]
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        databases = backend.list_databases()

        assert databases == ["db1", "db2", "db3"]
        mock_conn.list_databases.assert_called_once()

    @patch("ibis.connect")
    def test_list_databases_failure(self, mock_connect):
        """Test database listing failure."""
        mock_conn = Mock()
        mock_conn.list_databases.side_effect = Exception("List failed")
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")

        with pytest.raises(RuntimeError, match="Database listing failed"):
            backend.list_databases()

    @patch("ibis.connect")
    def test_list_tables_success(self, mock_connect):
        """Test successful table listing."""
        mock_conn = Mock()
        mock_conn.list_tables.return_value = ["table1", "table2"]
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        tables = backend.list_tables("test_db")

        assert tables == ["table1", "table2"]
        mock_conn.list_tables.assert_called_once_with(database="test_db")

    @patch("ibis.connect")
    def test_list_tables_no_database(self, mock_connect):
        """Test table listing without database filter."""
        mock_conn = Mock()
        mock_conn.list_tables.return_value = ["table1", "table2"]
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        tables = backend.list_tables()

        assert tables == ["table1", "table2"]
        mock_conn.list_tables.assert_called_once_with(database=None)

    @patch("ibis.connect")
    def test_get_table_schema_success(self, mock_connect):
        """Test successful table schema extraction."""
        mock_conn = Mock()
        mock_table = Mock()

        # Create a proper mock schema that supports len() and items()
        class MockSchema:
            def __init__(self, items):
                self._items = items

            def items(self):
                return self._items

            def __len__(self):
                return len(self._items)

        mock_schema = MockSchema(
            [("id", "int64"), ("name", "string"), ("created_at", "timestamp")]
        )

        mock_table.schema.return_value = mock_schema
        mock_conn.table.return_value = mock_table
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        schema = backend.get_table_schema("users", "test_db")

        expected = {
            "columns": [
                {"name": "id", "type": "int64"},
                {"name": "name", "type": "string"},
                {"name": "created_at", "type": "timestamp"},
            ],
            "column_count": 3,
        }
        assert schema == expected
        mock_conn.table.assert_called_once_with("users", database="test_db")

    @patch("ibis.connect")
    def test_get_table_schema_failure(self, mock_connect):
        """Test table schema extraction failure."""
        mock_conn = Mock()
        mock_conn.table.side_effect = Exception("Table not found")
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        schema = backend.get_table_schema("nonexistent", "test_db")

        expected = {"columns": [], "column_count": 0}
        assert schema == expected

    @patch("ibis.connect")
    def test_execute_query_success(self, mock_connect):
        """Test successful query execution."""
        mock_conn = Mock()
        mock_sql_expr = Mock()
        mock_df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mock_sql_expr.to_pandas.return_value = mock_df
        mock_conn.sql.return_value = mock_sql_expr
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
        result = backend.execute_query("SELECT * FROM users LIMIT 3")

        assert len(result) == 3
        assert list(result.columns) == ["id", "name"]
        mock_conn.sql.assert_called_once_with("SELECT * FROM users LIMIT 3")
        mock_sql_expr.to_pandas.assert_called_once()

    @patch("ibis.connect")
    def test_execute_query_dangerous_sql(self, mock_connect):
        """Test query execution with dangerous SQL."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")

        # NOTE: icontract raises ViolationError for contract violations
        with pytest.raises(Exception) as exc_info:
            backend.execute_query("DROP TABLE users")
        assert "Only SELECT queries allowed" in str(exc_info.value)

    @patch("ibis.connect")
    def test_execute_query_failure(self, mock_connect):
        """Test query execution failure."""
        mock_conn = Mock()
        mock_conn.sql.side_effect = Exception("Query failed")
        mock_connect.return_value = mock_conn

        backend = AnalyticsConnector("postgres://user:pass@host:5432/db")

        with pytest.raises(RuntimeError, match="Query failed"):
            backend.execute_query("SELECT * FROM users")

    @patch("ibis.connect")
    def test_context_manager(self, mock_connect):
        """Test context manager functionality."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        with AnalyticsConnector("postgres://user:pass@host:5432/db") as backend:
            assert backend.backend_type == "postgres"
            assert backend._conn == mock_conn

        mock_conn.close.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        # _detect_backend_type is a static method that returns a string, not an AnalyticsConnector instance
        backend_type = AnalyticsConnector._detect_backend_type("postgres://test")
        assert backend_type == "postgres"

        # Test repr on actual backend instance
        with patch("ibis.connect") as mock_connect:
            mock_connect.return_value = Mock()
            backend = AnalyticsConnector("postgres://user:pass@host:5432/db")
            assert repr(backend) == "AnalyticsConnector(backend='postgres')"


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @patch("ibis.connect")
    def test_create_postgres_backend(self, mock_connect):
        """Test PostgreSQL backend creation utility."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        result = create_postgres_backend(
            host="localhost",
            database="testdb",
            user="testuser",
            password="testpass",
            port=5432,
        )

        expected_connection_string = (
            "postgres://testuser:testpass@localhost:5432/testdb"
        )
        mock_connect.assert_called_once_with(expected_connection_string)
        assert isinstance(result, AnalyticsConnector)
        assert result.backend_type == "postgres"
        assert result.connection_string == expected_connection_string


class TestIntegration:
    """Integration tests (require actual database connections)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration")
        if hasattr(pytest, "config")
        else True,
        reason="Integration tests require --run-integration flag",
    )
    def test_real_postgres_connection(self):
        """Test with real PostgreSQL connection if available."""
        # This test requires a running PostgreSQL instance
        # Run with: pytest --run-integration
        try:
            backend = AnalyticsConnector(
                "postgres://postgres:postgres@localhost:5432/testdb"
            )
            databases = backend.list_databases()
            assert isinstance(databases, list)
            backend.close()
        except Exception:
            pytest.skip("PostgreSQL not available for integration testing")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration")
        if hasattr(pytest, "config")
        else True,
        reason="Integration tests require --run-integration flag",
    )
    def test_real_athena_connection(self):
        """Test with real Athena connection if credentials available."""
        # This test requires AWS credentials
        # Run with: pytest --run-integration
        try:
            backend = AnalyticsConnector(
                "athena://awsdatacatalog?region=ap-southeast-2&database=testdb"
            )
            databases = backend.list_databases()
            assert isinstance(databases, list)
            backend.close()
        except Exception:
            pytest.skip("Athena credentials not available for integration testing")


if __name__ == "__main__":
    pytest.main([__file__])
