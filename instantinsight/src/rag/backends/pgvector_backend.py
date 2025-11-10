"""PgVector-backed session and connection management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pgvector.psycopg import register_vector
from psycopg.rows import RowFactory, dict_row
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.rag.vector_db.config import HNSW, Distance, SearchType


@dataclass
class PgVectorBackendConfig:
    """Configuration options for PgVectorBackend."""

    db_url: str
    table_name: str = "text2sql2vis_pgvector_internal"
    schema: str = "ai"
    pool_size: int = 5
    max_overflow: int = 10
    search_type: SearchType = SearchType.vector
    vector_index: HNSW = field(default_factory=HNSW)
    distance: Distance = Distance.cosine
    prefix_match: bool = False
    vector_score_weight: float = 0.5


class PgVectorBackend:
    """Manage pooled access to PostgreSQL with pgvector extension."""

    __slots__ = ("_config", "_engine", "_initialized")

    def __init__(self, config: PgVectorBackendConfig):
        """Initialize PgVector backend with configuration."""
        self._config = config
        self._engine: Engine | None = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    def ensure_initialized(self) -> bool:
        """Initialise the SQLAlchemy engine and ensure pgvector extension."""
        if self._initialized:
            return True

        try:
            engine = self._ensure_engine()
            with engine.connect() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.commit()

            self._initialized = True
            logger.info("✓ PgVector backend initialised with pooled engine")
            return True

        except SQLAlchemyError as error:
            logger.error(f"Failed to initialise PgVector backend: {error}")
            return False

    def _ensure_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self._config.db_url,
                pool_size=self._config.pool_size,
                max_overflow=self._config.max_overflow,
                pool_pre_ping=True,
                future=True,
            )
        return self._engine

    @contextmanager
    def cursor(self, *, row_factory: RowFactory | None = dict_row) -> Iterator[Any]:
        """
        Yield a psycopg cursor from the pooled engine.

        Args:
            row_factory: Optional row factory for cursor results

        Yields:
            Psycopg cursor with pgvector support registered

        """
        engine = self._ensure_engine()
        connection = engine.raw_connection()

        # NOTE: We use psycopg3 exclusively (postgresql+psycopg://)
        # SQLAlchemy wraps the connection, so we need to access the actual driver connection
        driver_connection = connection.driver_connection

        register_vector(driver_connection)
        if row_factory:
            cursor = driver_connection.cursor(row_factory=row_factory)
        else:
            cursor = driver_connection.cursor()

        try:
            yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    @contextmanager
    def session(self) -> Iterator[Any]:
        """
        Yield a raw database connection for session-like operations.

        NOTE This method exists for backward compatibility. Since RAG operations
        use direct SQL via cursor(), this returns a raw connection rather than
        a SQLAlchemy ORM session.

        Raises:
            RuntimeError: Session-based operations are not supported in this implementation

        """
        raise RuntimeError(
            "Session-based operations are not supported. "
            "Use cursor() for direct SQL queries instead."
        )

    def dispose(self) -> None:
        """Dispose engine resources and reset state."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

        self._initialized = False

    def close_all(self) -> None:
        """Compatibility alias for dispose."""
        self.dispose()
