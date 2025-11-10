"""Unit tests for PgvectorRAG using the backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag.pgvector_rag import PgvectorRAG


@dataclass
class DummyBackend:
    """Dummy backend for testing."""

    ensure_result: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        self.ensure_calls = 0
        self.bound_embedder = None

    def ensure_initialized(self) -> bool:
        """
        Ensure backend is initialized.

        Returns:
            True if backend is initialized

        """
        self.ensure_calls += 1
        return self.ensure_result

    def cursor(self, row_factory=None):
        """
        Create cursor.

        Args:
            row_factory: Row factory function

        Raises:
            AssertionError: Always raises as cursor should not be used

        """
        raise AssertionError("cursor should not be used in this test")

    @property
    def is_initialized(self) -> bool:
        """
        Check if backend is initialized.

        Returns:
            True if backend is initialized

        """
        return self.ensure_calls > 0 and self.ensure_result

    def close_all(self):
        """Close all connections."""
        self.closed = True


def test_connect_to_database_creates_backend(monkeypatch):
    """Test that connect_to_database creates backend."""
    created_configs = {}

    def fake_backend_class(config):
        created_configs["config"] = config
        return DummyBackend()

    monkeypatch.setattr("src.rag.pgvector_rag.PgVectorBackend", fake_backend_class)

    rag = PgvectorRAG(
        connection_string="postgresql://user:pass@localhost/db",
        pool_config={"pool_size": 9, "max_overflow": 2},
        backend_config={
            "db_url": "postgresql+psycopg://user:pass@localhost/db",
            "schema": "custom",
        },
    )

    assert rag.connect_to_database() is True
    backend = rag.backend
    assert isinstance(backend, DummyBackend)
    # NOTE: ensure_initialized is called twice in connect_to_database (lines 100 and 104)
    assert backend.ensure_calls == 2
    cfg = created_configs["config"]
    assert cfg.pool_size == 9
    assert cfg.max_overflow == 2
    assert cfg.schema == "custom"
    assert cfg.db_url == "postgresql+psycopg://user:pass@localhost/db"


def test_connect_to_database_returns_false_when_backend_fails(monkeypatch):
    """Test that connect_to_database returns False when backend fails."""
    monkeypatch.setattr(
        "src.rag.pgvector_rag.PgVectorBackend",
        lambda config: DummyBackend(ensure_result=False),
    )

    rag = PgvectorRAG(connection_string="postgresql://some")
    assert rag.connect_to_database() is False


def test_initialize_embeddings_creates_bedrock_embeddings(monkeypatch):
    """Test that initialize_embeddings creates Bedrock embeddings."""

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_query(self, text):
            return [0.0] * 1024

    dummy_backend = DummyBackend()

    rag = PgvectorRAG(
        connection_string="postgresql://some",
        backend=dummy_backend,
    )

    monkeypatch.setattr("src.rag.pgvector_rag.BedrockEmbeddings", FakeEmbeddings)

    assert rag.initialize_embeddings(aws_region="us-west-2") is True
    # NOTE: New backend no longer has bind_embedder method
    assert rag.embeddings is not None
    assert rag.embeddings.kwargs["region_name"] == "us-west-2"
    assert rag.embeddings.kwargs["model_id"] == "amazon.titan-embed-text-v2:0"


def test_cursor_requires_backend():
    """Test that cursor requires backend."""
    rag = PgvectorRAG(connection_string="postgresql://some")

    with pytest.raises(RuntimeError):
        with rag.cursor():
            pass
