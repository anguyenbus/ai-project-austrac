"""
SQL generation and refinement operations.

Simple SQL processing functions.
"""

import time

from loguru import logger

from src.utils.langfuse_client import observe

from ..stages import Stage, StageResult


@observe(name="sql_generation")
def run_sql_generation(
    rag_instance,
    query: str,
    return_context: bool = False,
    prior_turns: list[dict[str, str]] | None = None,
) -> StageResult:
    """Run SQL generation stage."""
    start_time = time.time()

    try:
        if not rag_instance:
            return StageResult(
                stage=Stage.SQL_GENERATION,
                success=False,
                error="No RAG instance available",
                duration=time.time() - start_time,
            )

        # Generate SQL using RAGEngine
        if return_context:
            result = rag_instance.generate_sql(
                query, return_context=True, prior_turns=prior_turns
            )
            result_data = _extract_context_result(result)
            sql = result_data.get("sql", "")
        else:
            sql = rag_instance.generate_sql(query, prior_turns=prior_turns)
            result_data = sql

        # Check if this is a clarification response
        is_clarification = (
            sql and isinstance(sql, str) and sql.startswith("CANNOT FIND TABLES:")
        )

        if is_clarification:
            return StageResult(
                stage=Stage.SQL_GENERATION,
                success=False,
                data=result_data,
                error="Query too vague - clarification needed",
                duration=time.time() - start_time,
            )

        return StageResult(
            stage=Stage.SQL_GENERATION,
            success=bool(sql) and not is_clarification,
            data=result_data,
            error=None if sql else "Failed to generate SQL",
            duration=time.time() - start_time,
        )

    except Exception as e:
        return StageResult(
            stage=Stage.SQL_GENERATION,
            success=False,
            error=str(e),
            duration=time.time() - start_time,
        )


def _extract_context_result(result):
    """Extract SQL and context from generation result."""
    if isinstance(result, dict):
        sql = result.get("sql", "")
        schema_context = result.get("schema_context", "")
        normalized = result.get("normalized_query")
        return {
            "sql": sql,
            "retrieved_context": schema_context,
            "normalized_query": normalized,
        }
    else:
        return {"sql": result, "retrieved_context": "", "normalized_query": None}


def attempt_sql_refinement(
    sql_fixer,
    rag_instance,
    current_sql: str,
    execution_error: str,
    _query: str | None = None,
) -> tuple:
    """
    Attempt SQL refinement using SQLFixer.

    Returns:
        (success: bool, refined_sql: str, retry_results: list)

    """
    if not sql_fixer or not execution_error or not rag_instance:
        return False, current_sql, []

    logger.info("🔧 Attempting SQL recovery refinement...")

    schema_context = None
    if hasattr(rag_instance, "get_schema_context"):
        try:
            schema_context = rag_instance.get_schema_context()
        except Exception:
            schema_context = None

    # Try refinement up to 3 times
    refined_sql = current_sql
    last_error = execution_error
    retry_results = []

    for attempt in range(1, 4):  # 1, 2, 3
        logger.info(f"🔄 Refinement attempt {attempt}/3")

        # Fix SQL using latest error
        fix_result = sql_fixer.refine_sql(
            sql=refined_sql,
            error=last_error,
            schema_context=schema_context,
        )

        if not fix_result.success:
            logger.warning(
                f"Refinement attempt {attempt} failed: {fix_result.error_message}"
            )
            break

        logger.info(f"✅ SQL refined (attempt {attempt}), testing execution...")

        from .query import run_query_execution

        retry_execution_result = run_query_execution(
            rag_instance, fix_result.corrected_sql
        )

        retry_stage = getattr(
            Stage, f"QUERY_EXECUTION_RETRY_{attempt}", Stage.QUERY_EXECUTION
        )
        retry_result = StageResult(
            stage=retry_stage,
            success=retry_execution_result.success,
            data=retry_execution_result.data,
            error=retry_execution_result.error,
            duration=retry_execution_result.duration or 0.0,
        )
        retry_results.append((attempt, retry_result))

        if retry_result.success:
            logger.info(f"✅ Query execution succeeded after {attempt} refinement(s)")
            return True, fix_result.corrected_sql, retry_results

        refined_sql = fix_result.corrected_sql
        last_error = retry_result.error or execution_error

        if last_error and execution_error and last_error == execution_error:
            logger.warning("Same error repeated, stopping refinement attempts")
            break

    logger.error("All 3 refinement attempts failed")
    return False, refined_sql, retry_results
