"""
Structured result types for SQL generation pipeline.

Eliminates sentinel strings and provides clear type safety.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SQLGenerationStatus(Enum):
    """Status of SQL generation operation."""

    SUCCESS = "success"
    CLARIFICATION_NEEDED = "clarification_needed"
    ERROR = "error"


@dataclass
class SQLGenerationResult:
    """Structured result from SQL generation."""

    status: SQLGenerationStatus
    sql: str | None = None
    clarification_message: str | None = None
    schema_context: str | None = None
    selected_tables: list[str] | None = None
    error: str | None = None
    validation_warnings: list[str] | None = None
    normalized_query: Any | None = None

    @property
    def success(self) -> bool:
        """Return True if SQL generation was successful."""
        return self.status == SQLGenerationStatus.SUCCESS

    @property
    def needs_clarification(self) -> bool:
        """Return True if clarification is needed from user."""
        return self.status == SQLGenerationStatus.CLARIFICATION_NEEDED


class ExecutionStatus(Enum):
    """Status of query execution."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Structured result from query execution."""

    status: ExecutionStatus
    data: Any = None
    error: str | None = None
    rows_affected: int | None = None

    @property
    def success(self) -> bool:
        """Return True if query execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


class ValidationStatus(Enum):
    """Status of schema validation."""

    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Structured result from schema validation."""

    status: ValidationStatus
    valid_tables: list[str] | None = None
    invalid_tables: list[str] | None = None
    invalid_columns: list[str] | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        """Return True if validation passed successfully."""
        return self.status == ValidationStatus.VALID
