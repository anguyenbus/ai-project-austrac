"""
Pipeline coordinator with sub-20-line functions.

Simple orchestration following project rules.
"""

import time

from loguru import logger

from src.utils.langfuse_client import observe

from .handlers import PipelineHandlers
from .stages import PipelineResult


class Pipeline:
    """Pipeline coordinator with small functions."""

    def __init__(self, rag_instance=None):
        """Initialize coordinator with handlers."""
        self.rag_engine = rag_instance
        self.handlers = PipelineHandlers()

    @observe(name="pipeline_process")
    def process(
        self,
        query: str,
        prior_turns: list[dict[str, str]] | None = None,
        return_context: bool = True,
        generate_visualization: bool = True,
    ) -> PipelineResult:
        """Process query with conversation history."""
        start_time = time.time()
        result = PipelineResult(query=query)

        # Store prior_turns in result for access by stages
        result.prior_turns = prior_turns or []

        try:
            # Execute pipeline stages sequentially
            result = self._run_validation_stage(result)
            if not result.success and result.error:
                return self._finalize_result(result, start_time)

            result = self._run_cache_lookup_stage(result)
            if result.success:  # Cache hit succeeded
                return self._finalize_result(result, start_time)

            result = self._run_sql_generation_stage(result, return_context)
            if not result.success:
                return self._finalize_result(result, start_time)

            result = self._run_execution_stage(result, generate_visualization)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.error = str(e)
            result.success = False

        return self._finalize_result(result, start_time)

    def _run_validation_stage(self, result: PipelineResult) -> PipelineResult:
        """Run query validation stage."""
        validation_result = self.handlers.validate_query(result.query)
        result.stages[validation_result.stage] = validation_result

        if not validation_result.success:
            if validation_result.data:
                result.sql = validation_result.data
                result.error = validation_result.error
            else:
                result.error = validation_result.error or "Query validation failed"

        return result

    def _run_cache_lookup_stage(self, result: PipelineResult) -> PipelineResult:
        """Run cache lookup with conversation context."""
        from .operations.cache import run_cache_lookup

        # Pass prior_turns to cache lookup
        cache_result = run_cache_lookup(
            self.handlers.semantic_cache, result.query, prior_turns=result.prior_turns
        )
        result.stages[cache_result.stage] = cache_result

        if cache_result.success and cache_result.data:
            return self.handlers.handle_cache_hit(result, cache_result, self.rag_engine)

        return result

    def _run_sql_generation_stage(
        self, result: PipelineResult, return_context: bool
    ) -> PipelineResult:
        """Run SQL generation stage."""
        sql_result = self.handlers.generate_sql(
            self.rag_engine,
            result.query,
            return_context,
            prior_turns=result.prior_turns,
        )
        result.stages[sql_result.stage] = sql_result

        if sql_result.success:
            result = self.handlers.extract_sql_and_context(result, sql_result)
            result.success = True  # Mark as successful when SQL is generated
        else:
            result.error = sql_result.error
            result.success = False

        return result

    def _run_execution_stage(
        self, result: PipelineResult, generate_visualization: bool
    ) -> PipelineResult:
        """Run query execution stage with potential refinement."""
        execution_result = self.handlers.execute_query(self.rag_engine, result.sql)
        result.stages[execution_result.stage] = execution_result

        if execution_result.success:
            result = self.handlers.handle_successful_execution(
                result, execution_result, generate_visualization
            )
        else:
            result = self.handlers.handle_execution_failure(
                result, execution_result, generate_visualization, self.rag_engine
            )

        return result

    def _finalize_result(
        self, result: PipelineResult, start_time: float
    ) -> PipelineResult:
        """Finalize pipeline result with timing and cost tracking."""
        result.total_duration = time.time() - start_time
        self.handlers.add_cost_tracking()
        return result
