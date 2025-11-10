"""Pytest configuration and shared fixtures for instantinsight tests."""

import shutil

# Add src to path for imports
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))


@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Create a temporary directory for test data."""
    test_dir = project_root_path / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after all tests
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def mock_postgres_config():
    """Mock PostgreSQL configuration for testing."""
    return {
        "host": "localhost",
        "port": "5432",
        "user": "test_user",
        "password": "test_password",
        "database": "test_your_db_name",
    }


@pytest.fixture
def sample_table_info():
    """Sample table information for testing schema extraction."""
    return {
        "customers": {
            "columns": [
                ("customer_id", "character varying", "NO", None, 5),
                ("company_name", "character varying", "NO", None, 40),
                ("contact_name", "character varying", "YES", None, 30),
                ("country", "character varying", "YES", None, 15),
            ],
            "primary_keys": ["customer_id"],
            "foreign_keys": [],
        },
        "orders": {
            "columns": [
                ("order_id", "smallint", "NO", None, None),
                ("customer_id", "character varying", "YES", None, 5),
                ("employee_id", "smallint", "YES", None, None),
                ("order_date", "date", "YES", None, None),
            ],
            "primary_keys": ["order_id"],
            "foreign_keys": [
                ("customer_id", "customers", "customer_id"),
                ("employee_id", "employees", "employee_id"),
            ],
        },
    }


@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture(scope="function")
def isolated_test_env(monkeypatch, tmp_path):
    """Create an isolated test environment with temporary directories."""
    # Set up temporary directories
    test_vectorstore = tmp_path / "vectorstore"
    test_vectorstore.mkdir()

    # Mock environment variables
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    # Neo4j support removed
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "false")

    return {"vectorstore_path": test_vectorstore, "temp_path": tmp_path}


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that require databases"
    )
    config.addinivalue_line(
        "markers", "postgres: Tests that require PostgreSQL database"
    )
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and requirements."""
    for item in items:
        # Mark tests that use external services
        if "neo4j" in item.name.lower() or "Neo4j" in str(item.function):
            item.add_marker(pytest.mark.neo4j)

        if "postgres" in item.name.lower() or "PostgreSQL" in str(item.function):
            item.add_marker(pytest.mark.postgres)

        # Mark integration tests
        if "integration" in item.name.lower() or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default for most tests)
        if not any(
            marker.name in ["integration", "postgres"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
