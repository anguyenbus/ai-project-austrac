"""Unit tests for PgVector backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag.backends.pgvector_backend import PgVectorBackend, PgVectorBackendConfig


class FakeCursor:
    """Fake cursor for testing."""

    def __init__(self, row_factory):
        """
        Initialize fake cursor.

        Args:
            row_factory: Row factory function

        """
        self.row_factory = row_factory
        self.closed = False

    def close(self):
        """Close the cursor."""
        self.closed = True


class FakeDriverConnection:
    """Fake driver connection for testing."""

    def __init__(self):
        """Initialize fake driver connection."""
        self.cursors: list[FakeCursor] = []

    def cursor(self, row_factory=None):
        """
        Create a new cursor.

        Args:
            row_factory: Row factory function

        Returns:
            FakeCursor instance

        """
        cursor = FakeCursor(row_factory)
        self.cursors.append(cursor)
        return cursor


class FakeRawConnection:
    """Fake raw connection for testing."""

    def __init__(self):
        """Initialize fake raw connection."""
        self.driver_connection = FakeDriverConnection()
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def commit(self):
        """Commit transaction."""
        self.commit_count += 1

    def rollback(self):
        """Rollback transaction."""
        self.rollback_count += 1

    def close(self):
        """Close connection."""
        self.closed = True


class FakeConnectionContext:
    """Fake connection context for testing."""

    def __init__(self, sql_log: list[str]):
        """
        Initialize fake connection context.

        Args:
            sql_log: List to log SQL statements

        """
        self.sql_log = sql_log
        self.commit_count = 0

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager."""
        return False

    def execute(self, statement):
        """
        Execute SQL statement.

        Args:
            statement: SQL statement to execute

        """
        self.sql_log.append(statement)

    def commit(self):
        """Commit transaction."""
        self.commit_count += 1


@dataclass
class FakeEngine:
    """Fake engine for testing."""

    raw_connection_obj: FakeRawConnection

    def __post_init__(self):
        """Post-initialization setup."""
        self.sql_log: list[str] = []
        self.connect_calls = 0
        self.raw_connection_calls = 0

    def connect(self):
        """
        Create a connection.

        Returns:
            FakeConnectionContext instance

        """
        self.connect_calls += 1
        return FakeConnectionContext(self.sql_log)

    def raw_connection(self):
        """
        Get raw connection.

        Returns:
            FakeRawConnection instance

        """
        self.raw_connection_calls += 1
        return self.raw_connection_obj

    def dispose(self):
        """Dispose engine."""
        self.disposed = True


def test_ensure_initialized_creates_extension(monkeypatch):
    """Test that ensure_initialized creates vector extension."""
    raw_conn = FakeRawConnection()
    engine = FakeEngine(raw_connection_obj=raw_conn)

    def fake_create_engine(url, **kwargs):
        assert url == "postgresql+psycopg://test"
        assert kwargs["pool_size"] == 3
        assert kwargs["max_overflow"] == 7
        assert kwargs["pool_pre_ping"] is True
        return engine

    monkeypatch.setattr(
        "src.rag.backends.pgvector_backend.create_engine", fake_create_engine
    )
    monkeypatch.setattr("src.rag.backends.pgvector_backend.text", lambda sql: sql)

    backend = PgVectorBackend(
        PgVectorBackendConfig(
            db_url="postgresql+psycopg://test",
            pool_size=3,
            max_overflow=7,
        )
    )

    assert backend.ensure_initialized() is True
    assert backend.is_initialized is True
    assert engine.connect_calls == 1
    assert engine.sql_log == ["CREATE EXTENSION IF NOT EXISTS vector"]


def test_cursor_commits_and_registers_vector(monkeypatch):
    """Test that cursor commits and registers vector."""
    raw_conn = FakeRawConnection()
    engine = FakeEngine(raw_connection_obj=raw_conn)

    backend = PgVectorBackend(PgVectorBackendConfig(db_url="postgresql+psycopg://test"))
    backend._engine = engine
    backend._initialized = True

    registered = []

    monkeypatch.setattr(
        "src.rag.backends.pgvector_backend.register_vector",
        lambda conn: registered.append(conn),
    )
    monkeypatch.setattr("src.rag.backends.pgvector_backend.dict_row", "DICT")

    with backend.cursor(row_factory="DICT") as cur:
        assert cur.row_factory == "DICT"

    assert registered == [raw_conn.driver_connection]
    assert raw_conn.commit_count == 1
    assert raw_conn.rollback_count == 0
    assert raw_conn.closed is True
    assert raw_conn.driver_connection.cursors[0].closed is True


def test_cursor_rollback_on_error(monkeypatch):
    """Test that cursor rolls back on error."""
    raw_conn = FakeRawConnection()
    engine = FakeEngine(raw_connection_obj=raw_conn)

    backend = PgVectorBackend(PgVectorBackendConfig(db_url="postgresql+psycopg://test"))
    backend._engine = engine
    backend._initialized = True

    monkeypatch.setattr(
        "src.rag.backends.pgvector_backend.register_vector",
        lambda conn: None,
    )
    monkeypatch.setattr("src.rag.backends.pgvector_backend.dict_row", None)

    with pytest.raises(RuntimeError):
        with backend.cursor():
            raise RuntimeError("boom")

    assert raw_conn.commit_count == 0
    assert raw_conn.rollback_count == 1
    assert raw_conn.closed is True


def test_dispose_clears_engine():
    """Test that dispose clears the engine."""
    raw_conn = FakeRawConnection()
    engine = FakeEngine(raw_connection_obj=raw_conn)

    backend = PgVectorBackend(
        PgVectorBackendConfig(
            db_url="postgresql+psycopg://test",
        )
    )
    backend._engine = engine
    backend._initialized = True

    assert backend.is_initialized is True
    backend.dispose()
    assert backend.is_initialized is False
    assert backend._engine is None
    assert engine.disposed is True
