"""
Pipeline stage handlers with sub-20-line functions.

All handler functions stay under 20 lines each.
"""

from typing import Any

from langfuse import Langfuse
from loguru import logger

from src.utils.cost_accumulator import cost_accumulator

from .operations.cache import run_cache_lookup, run_cache_storage
from .operations.query import run_query_execution, run_query_validation
from .operations.sql import attempt_sql_refinement, run_sql_generation
from .stages import PipelineResult, Stage


class PipelineHandlers:
    """Handler methods for pipeline stages."""

    def __init__(self):
        """Initialize handlers with components."""
        self._init_components()

    def _init_components(self):
        """Initialize pipeline components."""
        try:
            from src.agents.intent_clarification_pipeline import GuardrailPipeline
            from src.agents.strand_agents.output.visualizer import VisualizationAgent
            from src.agents.strand_agents.sql.corrector import SQLFixer
            from src.cache.semantic_cache import SemanticCache

            self.sql_fixer = SQLFixer()
            self.visualization_agent = VisualizationAgent()
            self.guardrail_pipeline = GuardrailPipeline()
            self.semantic_cache = SemanticCache(similarity_threshold=0.92)

        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}")
            self.sql_fixer = None
            self.visualization_agent = None
            self.guardrail_pipeline = None
            self.semantic_cache = None

    def validate_query(self, query: str):
        """Validate query using guardrail pipeline."""
        return run_query_validation(self.guardrail_pipeline, query)

    def lookup_cache(self, query: str):
        """Look up query in semantic cache."""
        return run_cache_lookup(self.semantic_cache, query)

    def generate_sql(
        self,
        rag_instance,
        query: str,
        return_context: bool,
        prior_turns: list[dict[str, str]] | None = None,
    ):
        """Generate SQL using RAG instance."""
        return run_sql_generation(rag_instance, query, return_context, prior_turns)

    def execute_query(self, rag_instance, sql: str):
        """Execute SQL query."""
        return run_query_execution(rag_instance, sql)

    def extract_sql_and_context(self, result: PipelineResult, sql_result):
        """Extract SQL and context from generation result."""
        if sql_result.data and isinstance(sql_result.data, dict):
            result.sql = sql_result.data.get("sql")
            result.retrieved_context = sql_result.data.get("retrieved_context", "")
            result.normalized_query = sql_result.data.get("normalized_query")
        else:
            result.sql = sql_result.data
            result.normalized_query = None
        return result

    def handle_cache_hit(self, result: PipelineResult, cache_result, rag_instance):
        """Handle cache hit scenario."""
        logger.info("Cache hit - executing cached SQL")
        cached_data = cache_result.data
        result.sql = cached_data.get("sql")

        execution_result = self.execute_query(rag_instance, result.sql)
        result.stages[Stage.QUERY_EXECUTION] = execution_result

        if execution_result.success:
            result.success = True
            result.error = None
            if cached_data.get("visualization"):
                result.visualization = cached_data["visualization"]
        else:
            result.error = execution_result.error

        return result

    def handle_successful_execution(
        self, result: PipelineResult, execution_result, generate_visualization: bool
    ):
        """Handle successful query execution."""
        result.success = True
        result.error = None  # Clear any previous errors

        if generate_visualization and execution_result.data is not None:
            viz_result = self._generate_visualization(
                result.query, execution_result.data, result.normalized_query
            )
            result.stages[Stage.VISUALIZATION] = viz_result
            if viz_result.success:
                result.visualization = viz_result.data

        cache_result = run_cache_storage(
            self.semantic_cache,
            result.query,
            result.sql,
            execution_result.data,
            result.visualization,
            prior_turns=result.prior_turns,
        )
        result.stages[Stage.CACHE_STORAGE] = cache_result

        return result

    def handle_execution_failure(
        self,
        result: PipelineResult,
        execution_result,
        generate_visualization: bool,
        rag_instance,
    ):
        """Handle query execution failure with refinement."""
        logger.error(f"Query execution failed: {execution_result.error}")

        success, refined_sql, retry_results = attempt_sql_refinement(
            self.sql_fixer,
            rag_instance,
            result.sql,
            execution_result.error,
            result.query,
        )

        self._store_retry_results(result, retry_results)

        if success:
            result.sql = refined_sql
            result.success = True
            result.error = None  # Clear error after successful refinement

            successful_retry = retry_results[-1][1]
            if generate_visualization and successful_retry.data is not None:
                viz_result = self._generate_visualization(
                    result.query, successful_retry.data, result.normalized_query
                )
                result.stages[Stage.VISUALIZATION] = viz_result
                if viz_result.success:
                    result.visualization = viz_result.data

            cache_result = run_cache_storage(
                self.semantic_cache,
                result.query,
                refined_sql,
                successful_retry.data,
                result.visualization,
                prior_turns=result.prior_turns,
            )
            result.stages[Stage.CACHE_STORAGE] = cache_result
        else:
            result.error = (
                f"Query execution failed after refinement: {execution_result.error}"
            )

        return result

    def _store_retry_results(self, result: PipelineResult, retry_results):
        """Store retry results in appropriate stage slots."""
        for attempt, retry_result in retry_results:
            if attempt == 1:
                result.stages[Stage.QUERY_EXECUTION_RETRY_1] = retry_result
            elif attempt == 2:
                result.stages[Stage.QUERY_EXECUTION_RETRY_2] = retry_result
            elif attempt == 3:
                result.stages[Stage.QUERY_EXECUTION_RETRY_3] = retry_result

    def _generate_visualization(self, query: str, data, normalized_query: Any | None):
        """Generate visualization for query results."""
        import time

        start_time = time.time()

        try:
            if not self.visualization_agent:
                from .stages import StageResult

                return StageResult(
                    stage=Stage.VISUALIZATION,
                    success=False,
                    error="No visualization agent",
                    duration=time.time() - start_time,
                )

            import pandas as pd

            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

            viz_result = self.visualization_agent.process(
                df, query, normalized_query=normalized_query
            )
            success = viz_result.get("success", False)

            from .stages import StageResult

            return StageResult(
                stage=Stage.VISUALIZATION,
                success=success,
                data=viz_result if success else None,
                error=viz_result.get("error") if not success else None,
                duration=time.time() - start_time,
            )

        except Exception as e:
            from .stages import StageResult

            return StageResult(
                stage=Stage.VISUALIZATION,
                success=False,
                error=str(e),
                duration=time.time() - start_time,
            )

    def add_cost_tracking(self):
        """Add cost tracking to Langfuse."""
        langfuse_client = None
        try:
            total_cost = cost_accumulator.get_total_cost()
            trace_id = cost_accumulator.trace_id

            if trace_id and total_cost > 0:
                langfuse_client = Langfuse()
                langfuse_client.score(
                    trace_id=trace_id,
                    name="total_pipeline_cost_usd",
                    value=total_cost,
                    comment=f"Total: ${total_cost:.6f}",
                )

            cost_accumulator.reset()
            if langfuse_client:
                langfuse_client.flush()

        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")
