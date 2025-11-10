"""
Pipeline stage definitions and results.

Simple types for pipeline execution tracking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Stage(Enum):
    """Pipeline stages for SQL generation."""

    QUERY_VALIDATION = "query_validation"
    CACHE_LOOKUP = "cache_lookup"
    SQL_GENERATION = "sql_generation"
    QUERY_EXECUTION = "query_execution"
    CACHE_STORAGE = "cache_storage"
    VISUALIZATION = "visualization"
    QUERY_EXECUTION_RETRY_1 = "query_execution_retry_1"
    QUERY_EXECUTION_RETRY_2 = "query_execution_retry_2"
    QUERY_EXECUTION_RETRY_3 = "query_execution_retry_3"


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: Stage
    success: bool
    data: Any = None
    error: str | None = None
    duration: float = 0.0


@dataclass
class PipelineResult:
    """Final result from pipeline execution."""

    query: str
    sql: str | None = None
    success: bool = False
    error: str | None = None
    stages: dict[Stage, StageResult] = None
    total_duration: float = 0.0
    retrieved_context: str | None = None
    visualization: dict[str, Any] | None = None
    normalized_query: Any | None = None
    prior_turns: list[dict[str, str]] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.stages is None:
            self.stages = {}
        if self.prior_turns is None:
            self.prior_turns = []
