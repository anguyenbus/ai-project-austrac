"""
Query validation and execution operations.

Simple query processing functions.
"""

import time

from loguru import logger

from src.utils.langfuse_client import observe

from ..stages import Stage, StageResult


@observe(name="query_validation")
def run_query_validation(guardrail_pipeline, query: str) -> StageResult:
    """Run query validation stage using guardrail pipeline."""
    start_time = time.time()

    try:
        if not guardrail_pipeline:
            return StageResult(
                stage=Stage.QUERY_VALIDATION,
                success=True,
                data=None,
                duration=time.time() - start_time,
            )

        is_valid, clarification_message, sanitized_query = (
            guardrail_pipeline.process_query(query)
        )

        if is_valid:
            logger.info("Query validation passed")
            return StageResult(
                stage=Stage.QUERY_VALIDATION,
                success=True,
                data=sanitized_query,
                duration=time.time() - start_time,
            )
        else:
            logger.warning("Query validation failed")
            return StageResult(
                stage=Stage.QUERY_VALIDATION,
                success=False,
                data=clarification_message,
                error="Query validation failed",
                duration=time.time() - start_time,
            )

    except Exception as e:
        logger.error(f"Query validation error: {e}")
        return StageResult(
            stage=Stage.QUERY_VALIDATION,
            success=True,  # Allow to proceed on validation errors
            error=str(e),
            duration=time.time() - start_time,
        )


@observe(name="execute_query")
def run_query_execution(rag_instance, sql: str) -> StageResult:
    """Execute the SQL query using UniversalExecutor via RAG instance."""
    start_time = time.time()

    try:
        if not rag_instance:
            return StageResult(
                stage=Stage.QUERY_EXECUTION,
                success=False,
                error="No RAG instance available for query execution",
                duration=time.time() - start_time,
            )

        # Execute using RAG instance (routes through UniversalExecutor)
        if not hasattr(rag_instance, "execute_sql"):
            return StageResult(
                stage=Stage.QUERY_EXECUTION,
                success=False,
                error="RAG instance does not support execute_sql method",
                duration=time.time() - start_time,
            )

        result_data = rag_instance.execute_sql(sql)

        if result_data is None:
            return StageResult(
                stage=Stage.QUERY_EXECUTION,
                success=False,
                error="Query execution failed - None returned",
                duration=time.time() - start_time,
            )

        return StageResult(
            stage=Stage.QUERY_EXECUTION,
            success=True,
            data=result_data,
            duration=time.time() - start_time,
        )

    except Exception as e:
        return StageResult(
            stage=Stage.QUERY_EXECUTION,
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )
