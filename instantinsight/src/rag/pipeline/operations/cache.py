"""
Cache operations for pipeline.

Simple cache lookup and storage functions.
"""

import time
from typing import Any

from src.utils.langfuse_client import observe

from ..stages import Stage, StageResult


@observe(name="cache_lookup")
def run_cache_lookup(
    semantic_cache, query: str, prior_turns: list[dict] | None = None
) -> StageResult:
    """Run cache lookup with conversation context."""
    start_time = time.time()

    try:
        if not semantic_cache:
            return StageResult(
                stage=Stage.CACHE_LOOKUP,
                success=False,
                error="No semantic cache available",
                duration=time.time() - start_time,
            )

        if hasattr(semantic_cache, "get_cached_result_sync"):
            # Pass prior_turns to cache lookup
            cached_result = semantic_cache.get_cached_result_sync(
                query, prior_turns=prior_turns
            )
            if cached_result:
                return _build_cache_result(cached_result, start_time)

        return StageResult(
            stage=Stage.CACHE_LOOKUP,
            success=False,
            error="No cache hit found",
            duration=time.time() - start_time,
        )

    except Exception as e:
        return StageResult(
            stage=Stage.CACHE_LOOKUP,
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )


def _build_cache_result(cached_result, start_time: float) -> StageResult:
    """Build cache result safely for both dataclass and dict types."""
    # Safe extraction for both dataclass and dict types
    sql = getattr(cached_result, "sql", None)
    if sql is None and hasattr(cached_result, "get"):
        sql = cached_result.get("sql", str(cached_result))

    confidence = getattr(cached_result, "confidence", None)
    if confidence is None and hasattr(cached_result, "get"):
        confidence = cached_result.get("confidence", 1.0)

    execution_result = getattr(cached_result, "data", None)
    if execution_result is None and hasattr(cached_result, "get"):
        execution_result = cached_result.get("execution_result")

    visualization = getattr(cached_result, "visualization", None)

    return StageResult(
        stage=Stage.CACHE_LOOKUP,
        success=True,
        data={
            "sql": sql,
            "confidence": confidence or 1.0,
            "execution_result": execution_result,
            "visualization": visualization,
        },
        duration=time.time() - start_time,
    )


@observe(name="cache_storage")
def run_cache_storage(
    semantic_cache,
    query: str,
    sql: str,
    execution_data: Any,
    visualization: dict[str, Any] | None = None,
    prior_turns: list[dict] | None = None,
) -> StageResult:
    """Run cache storage with conversation context."""
    start_time = time.time()

    try:
        if not semantic_cache:
            return StageResult(
                stage=Stage.CACHE_STORAGE,
                success=False,
                error="No semantic cache available",
                duration=time.time() - start_time,
            )

        # Skip caching for historical context queries
        if "Historical context" in query or "Current SQL query:" in query:
            return StageResult(
                stage=Stage.CACHE_STORAGE,
                success=True,
                data="Skipped - Historical context query",
                duration=time.time() - start_time,
            )

        if hasattr(semantic_cache, "store_result_sync"):
            success = semantic_cache.store_result_sync(
                query=query,
                sql=sql,
                result_data=execution_data,
                visualization=visualization,
                prior_turns=prior_turns,  # NEW
            )

            return StageResult(
                stage=Stage.CACHE_STORAGE,
                success=success,
                data="Results cached successfully"
                if success
                else "Cache storage failed",
                error=None if success else "Cache storage returned False",
                duration=time.time() - start_time,
            )

        return StageResult(
            stage=Stage.CACHE_STORAGE,
            success=False,
            error="No cache storage method available",
            duration=time.time() - start_time,
        )

    except Exception as e:
        return StageResult(
            stage=Stage.CACHE_STORAGE,
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )
