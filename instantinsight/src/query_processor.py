"""
QueryProcessor - Production-ready query orchestration with Lambda compatibility.

This module provides a unified interface for natural language to SQL processing
with semantic caching, result export, and both synchronous and asynchronous execution.

Key Features:
    - Dual Interface: Both sync (local) and async (Lambda) methods
    - Semantic Caching: Automatic caching with Redis backend
    - Result Export: Local filesystem or S3 export
    - Pipeline Integration: Orchestrates RAG pipeline execution
    - Type Safety: Full type hints with runtime validation
    - Observability: Langfuse instrumentation built-in

Architecture:
    QueryProcessor orchestrates but does not implement:
    - SQL generation (delegates to RAGEngine → Pipeline)
    - Cache storage (delegates to SemanticCache)
    - Query execution (delegates to QueryExecutor)
    - AWS operations (uses boto3 defaults)

Usage Examples:
    Synchronous (Local Development):
        >>> processor = QueryProcessor()
        >>> result = processor.process_query("Show top 10 products by revenue")
        >>> print(result["sql"])
        >>> print(result["data"])

    Asynchronous (Lambda):
        >>> processor = QueryProcessor()
        >>> result = await processor.process_query_async(
        ...     "Show top 10 products",
        ...     export_results=True
        ... )
        >>> print(result["export_path"])  # S3 URI

    Custom Configuration:
        >>> config = {
        ...     "enable_cache": True,
        ...     "export_enabled": True,
        ...     "s3_bucket": "my-results-bucket",
        ...     "query_timeout_seconds": 120.0,
        ... }
        >>> processor = QueryProcessor(config=config)

Lambda Integration:
    See reference/lambda/query_handler.py for Lambda handler example.
"""

import io
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from loguru import logger
from typing_extensions import TypedDict

from src.rag.engines.rag_engine import RAGEngine
from src.rag.pipeline.base import Pipeline
from src.utils.langfuse_client import observe

# ============================================================================
# Type Definitions
# ============================================================================


class QueryResult(TypedDict, total=False):
    """
    Complete query processing result with performance metrics.

    This structure provides comprehensive information about query processing,
    including results, cache statistics, export information, and errors.

    Fields:
        Core Results:
            query: Original user query
            sql: Generated SQL statement
            data: Query results as DataFrame
            success: Overall success flag

        Performance Metrics:
            total_duration_ms: End-to-end processing time
            sql_generation_ms: Time to generate SQL
            execution_ms: Query execution time

        Cache Information:
            cache_hit: Whether cache was used
            cache_confidence: Similarity score (0.0-1.0)
            cache_retrieval_ms: Cache lookup time

        Export Information:
            export_path: Local file path or S3 URI
            export_type: "local" or "s3"
            export_size_bytes: Size of exported file

        Visualization:
            visualization: Plotly chart schema

        Error Handling:
            error: Error message if failed
            error_type: Error classification

        Pipeline Metadata:
            pipeline_stages: Stages executed
            row_count: Number of rows returned
    """

    # Core Results
    query: str
    sql: str | None
    data: pd.DataFrame | None
    success: bool

    # Performance Metrics
    total_duration_ms: int
    sql_generation_ms: int | None
    execution_ms: int | None

    # Cache Information
    cache_hit: bool
    cache_confidence: float | None
    cache_retrieval_ms: int | None

    # Export Information
    export_path: str | None
    export_type: str | None
    export_size_bytes: int | None

    # Visualization
    visualization: dict[str, Any] | None

    # Error Handling
    error: str | None
    error_type: str | None

    # Pipeline Metadata
    pipeline_stages: list[str] | None
    row_count: int | None


class QueryProcessorConfig(TypedDict, total=False):
    """
    Configuration for QueryProcessor initialization.

    Provides fine-grained control over pipeline behavior, caching,
    export strategies, and resource limits.

    Fields:
        Pipeline Settings:
            use_pipeline: Enable pipeline architecture (default: True)
            enable_cache: Enable semantic caching (default: True)

        Export Settings:
            export_enabled: Enable automatic export (default: True)
            export_format: Export format (default: "csv")
            local_output_dir: Local export directory
            s3_bucket: S3 bucket for Lambda exports
            s3_prefix: S3 key prefix (default: "query-results/")

        Environment Detection:
            is_lambda: Running in Lambda environment

        Resource Limits:
            query_timeout_seconds: Query timeout (default: 60.0)
            max_result_rows: Max rows to export (default: 10000)
    """

    # Pipeline Settings
    use_pipeline: bool
    enable_cache: bool

    # Export Settings
    export_enabled: bool
    export_format: str
    local_output_dir: str
    s3_bucket: str | None
    s3_prefix: str

    # Environment Detection
    is_lambda: bool

    # Resource Limits
    query_timeout_seconds: float
    max_result_rows: int


# ============================================================================
# Custom Exceptions
# ============================================================================


class QueryProcessorError(Exception):
    """Base exception for QueryProcessor operations."""


class ValidationError(QueryProcessorError):
    """Query validation failed."""


class GenerationError(QueryProcessorError):
    """SQL generation failed."""


class ExecutionError(QueryProcessorError):
    """Query execution failed."""


class ExportError(QueryProcessorError):
    """Result export failed."""


# ============================================================================
# Constants
# ============================================================================

# Default configuration for local development
DEFAULT_LOCAL_CONFIG: Final[QueryProcessorConfig] = {
    "use_pipeline": True,
    "enable_cache": True,
    "export_enabled": True,
    "export_format": "csv",
    "local_output_dir": "temp_results",
    "s3_bucket": None,
    "s3_prefix": "query-results/",
    "is_lambda": False,
    "query_timeout_seconds": 60.0,
    "max_result_rows": 10000,
}


# ============================================================================
# Main QueryProcessor Class
# ============================================================================


class QueryProcessor:
    """
    Production-ready query processor with sync/async interfaces.

    Responsibilities:
        1. Orchestrate pipeline execution with semantic caching
        2. Execute SQL queries via RAGEngine
        3. Export results to local filesystem or S3
        4. Provide both sync and async interfaces
        5. Track performance metrics and cache statistics

    Does NOT Handle:
        - SQL generation (delegates to RAGEngine → Pipeline)
        - Cache storage logic (delegates to SemanticCache)
        - Direct database connections (delegates to QueryExecutor)
        - AWS credential management (uses boto3 defaults)
    """

    __slots__ = ("rag_engine", "pipeline", "config", "_environment")

    def __init__(
        self,
        rag_engine: RAGEngine | None = None,
        pipeline: Pipeline | None = None,
        config: QueryProcessorConfig | None = None,
    ):
        """
        Initialize QueryProcessor with dependency injection.

        Args:
            rag_engine: Configured RAGEngine instance (created if None)
            pipeline: Configured Pipeline instance (created if None)
            config: Custom configuration (uses defaults if None)

        Raises:
            RuntimeError: If engine or pipeline creation fails

        """
        # NOTE: Detect environment before initializing components
        self._environment = self._detect_environment()

        # Initialize configuration
        self.config = self._merge_config(config)

        # Initialize components (create if not provided)
        if rag_engine is None or pipeline is None:
            engine, error = RAGEngine.create_instance()
            if not engine:
                raise RuntimeError(f"Failed to create RAGEngine: {error}")
            self.rag_engine = engine
            self.pipeline = Pipeline(rag_instance=engine)
        else:
            self.rag_engine = rag_engine
            self.pipeline = pipeline

        logger.info(
            f"✅ QueryProcessor initialized (cache={self.config['enable_cache']}, "
            f"export={self.config['export_enabled']}, env={self._environment['export_strategy']})"
        )

    def _detect_environment(self) -> dict[str, Any]:
        """
        Detect execution environment and configure accordingly.

        Returns:
            Environment configuration with:
                - is_lambda: bool
                - export_strategy: "local" | "s3"
                - output_location: Path or S3 bucket

        """
        is_lambda = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

        if is_lambda:
            return {
                "is_lambda": True,
                "export_strategy": "s3",
                "s3_bucket": os.getenv("RESULTS_BUCKET", "instantinsight-query-results"),
                "s3_prefix": os.getenv("RESULTS_PREFIX", "query-results/"),
            }
        else:
            return {
                "is_lambda": False,
                "export_strategy": "local",
                "output_dir": Path(os.getenv("RESULTS_DIR", "temp_results")),
            }

    def _merge_config(
        self, custom_config: QueryProcessorConfig | None
    ) -> QueryProcessorConfig:
        """Merge custom config with defaults and environment detection."""
        config = DEFAULT_LOCAL_CONFIG.copy()

        # Override with environment-detected settings
        config["is_lambda"] = self._environment["is_lambda"]
        if self._environment["export_strategy"] == "s3":
            config["s3_bucket"] = self._environment.get("s3_bucket")
            config["s3_prefix"] = self._environment.get("s3_prefix", "query-results/")
        else:
            config["local_output_dir"] = str(
                self._environment.get("output_dir", "temp_results")
            )

        # Override with custom config if provided
        if custom_config:
            config.update(custom_config)

        return config

    @observe(name="query_processor_process_query")
    def process_query(
        self,
        query: str,
        prior_turns: list[dict[str, str]] | None = None,
        export_results: bool = True,
        generate_visualization: bool = True,
    ) -> QueryResult:
        """
        Process query synchronously for local development.

        Args:
            query: Natural language query
            prior_turns: Optional list of prior conversation turns
            export_results: Whether to export results
            generate_visualization: Whether to generate visualization

        Returns:
            QueryResult with sql, data, cache_info, export_path, visualization

        Raises:
            ValidationError: If query validation fails
            GenerationError: If SQL generation fails
            ExecutionError: If query execution fails

        """
        start_time = time.time()
        result: QueryResult = {
            "query": query,
            "success": False,
            "cache_hit": False,
        }

        try:
            # Validate input
            self._validate_query(query)

            # Process through pipeline
            pipeline_start = time.time()
            pipeline_result = self.pipeline.process(
                query=query,
                prior_turns=prior_turns,
                return_context=True,
                generate_visualization=generate_visualization,
            )
            pipeline_duration = int((time.time() - pipeline_start) * 1000)

            # Extract pipeline results
            result["sql"] = pipeline_result.sql
            result["success"] = pipeline_result.success
            result["sql_generation_ms"] = pipeline_duration

            # Check cache hit
            from src.rag.pipeline.stages import Stage

            if Stage.CACHE_LOOKUP in pipeline_result.stages:
                cache_stage = pipeline_result.stages[Stage.CACHE_LOOKUP]
                result["cache_hit"] = cache_stage.success
                if cache_stage.success and cache_stage.data:
                    result["cache_confidence"] = getattr(
                        cache_stage.data, "confidence", 1.0
                    )

            # Handle pipeline failure
            if not pipeline_result.success:
                result["error"] = pipeline_result.error or "Pipeline processing failed"
                result["error_type"] = "generation_error"
                return result

            # Extract data and visualization
            if hasattr(pipeline_result, "visualization"):
                result["visualization"] = pipeline_result.visualization

            # NOTE: Data extraction from pipeline stages
            if Stage.QUERY_EXECUTION in pipeline_result.stages:
                exec_stage = pipeline_result.stages[Stage.QUERY_EXECUTION]
                if exec_stage.success and exec_stage.data is not None:
                    result["data"] = exec_stage.data
                    result["row_count"] = len(exec_stage.data)
                    result["execution_ms"] = int(exec_stage.duration * 1000)

            # Export results if requested and data available
            if (
                export_results
                and self.config["export_enabled"]
                and result.get("data") is not None
            ):
                try:
                    if self._environment["export_strategy"] == "s3":
                        # S3 export requires async, skip for sync method
                        logger.warning(
                            "S3 export not available in sync mode, use process_query_async"
                        )
                    else:
                        export_path = self._export_to_local(
                            result["data"],
                            query,
                            Path(self.config["local_output_dir"]),
                        )
                        result["export_path"] = export_path
                        result["export_type"] = "local"
                        # Get file size
                        result["export_size_bytes"] = Path(export_path).stat().st_size
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    result["error"] = f"Export failed: {e}"

            # Extract pipeline stages
            result["pipeline_stages"] = [
                stage.value for stage in pipeline_result.stages.keys()
            ]

        except ValidationError as e:
            logger.warning(f"Query validation failed: {e}")
            result["error"] = str(e)
            result["error_type"] = "validation_error"

        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}")
            result["error"] = str(e)
            result["error_type"] = "unexpected_error"

        finally:
            # Always set total duration
            result["total_duration_ms"] = int((time.time() - start_time) * 1000)

        return result

    @observe(name="query_processor_process_query_async")
    async def process_query_async(
        self,
        query: str,
        prior_turns: list[dict[str, str]] | None = None,
        export_results: bool = True,
        generate_visualization: bool = True,
        user_id: str | None = None,
    ) -> QueryResult:
        """
        Process query asynchronously for Lambda execution.

        Args:
            query: Natural language query
            prior_turns: Optional list of prior conversation turns
            export_results: Whether to export results
            generate_visualization: Whether to generate visualization
            user_id: Optional user ID for cache isolation

        Returns:
            QueryResult with complete processing information

        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            ValidationError: If query validation fails

        """
        import asyncio

        timeout = self.config["query_timeout_seconds"]

        try:
            result = await asyncio.wait_for(
                self._process_query_internal_async(
                    query, prior_turns, export_results, generate_visualization, user_id
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Query processing timed out after {timeout}s")
            return QueryResult(
                query=query,
                success=False,
                error=f"Query processing timed out after {timeout}s",
                error_type="timeout_error",
                cache_hit=False,
                total_duration_ms=int(timeout * 1000),
            )

    async def _process_query_internal_async(
        self,
        query: str,
        prior_turns: list[dict[str, str]] | None,
        export_results: bool,
        generate_visualization: bool,
        user_id: str | None,
    ) -> QueryResult:
        """Process query internally with async execution."""
        import asyncio

        start_time = time.time()
        result: QueryResult = {
            "query": query,
            "success": False,
            "cache_hit": False,
        }

        try:
            # Validate input
            self._validate_query(query)

            # Process through pipeline (run in executor since pipeline is sync)
            loop = asyncio.get_event_loop()
            pipeline_start = time.time()
            pipeline_result = await loop.run_in_executor(
                None,
                lambda: self.pipeline.process(
                    query=query,
                    prior_turns=prior_turns,
                    return_context=True,
                    generate_visualization=generate_visualization,
                ),
            )
            pipeline_duration = int((time.time() - pipeline_start) * 1000)

            # Extract results (same as sync version)
            result["sql"] = pipeline_result.sql
            result["success"] = pipeline_result.success
            result["sql_generation_ms"] = pipeline_duration

            # Check cache hit
            from src.rag.pipeline.stages import Stage

            if Stage.CACHE_LOOKUP in pipeline_result.stages:
                cache_stage = pipeline_result.stages[Stage.CACHE_LOOKUP]
                result["cache_hit"] = cache_stage.success

            if not pipeline_result.success:
                result["error"] = pipeline_result.error or "Pipeline processing failed"
                result["error_type"] = "generation_error"
                return result

            # Extract visualization
            if hasattr(pipeline_result, "visualization"):
                result["visualization"] = pipeline_result.visualization

            # Extract data
            if Stage.QUERY_EXECUTION in pipeline_result.stages:
                exec_stage = pipeline_result.stages[Stage.QUERY_EXECUTION]
                if exec_stage.success and exec_stage.data is not None:
                    result["data"] = exec_stage.data
                    result["row_count"] = len(exec_stage.data)
                    result["execution_ms"] = int(exec_stage.duration * 1000)

            # Export to S3 if requested (async)
            if (
                export_results
                and self.config["export_enabled"]
                and result.get("data") is not None
            ):
                try:
                    if self._environment["export_strategy"] == "s3":
                        export_path = await self._export_to_s3_async(
                            result["data"],
                            query,
                            self.config["s3_bucket"],
                            self.config["s3_prefix"],
                        )
                        result["export_path"] = export_path
                        result["export_type"] = "s3"
                    else:
                        export_path = self._export_to_local(
                            result["data"],
                            query,
                            Path(self.config["local_output_dir"]),
                        )
                        result["export_path"] = export_path
                        result["export_type"] = "local"
                        result["export_size_bytes"] = Path(export_path).stat().st_size
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    result["error"] = f"Export failed: {e}"

            result["pipeline_stages"] = [
                stage.value for stage in pipeline_result.stages.keys()
            ]

        except ValidationError as e:
            logger.warning(f"Query validation failed: {e}")
            result["error"] = str(e)
            result["error_type"] = "validation_error"

        except Exception as e:
            logger.error(f"Unexpected error in async processing: {e}")
            result["error"] = str(e)
            result["error_type"] = "unexpected_error"

        finally:
            result["total_duration_ms"] = int((time.time() - start_time) * 1000)

        return result

    def _validate_query(self, query: str) -> None:
        """
        Validate query before processing.

        Args:
            query: User query to validate

        Raises:
            ValidationError: If query fails validation

        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        if len(query) > 5000:
            raise ValidationError("Query exceeds maximum length of 5000 characters")

        # NOTE: Check for potentially dangerous patterns (basic protection)
        # SQL is AI-generated, but log suspicious patterns for monitoring
        dangerous_patterns = ["DROP TABLE", "DELETE FROM", "TRUNCATE"]
        upper_query = query.upper()
        for pattern in dangerous_patterns:
            if pattern in upper_query:
                logger.warning(f"Potentially dangerous pattern detected: {pattern}")

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use in filenames.

        Args:
            text: Text to sanitize
            max_length: Maximum filename length

        Returns:
            Sanitized filename string

        """
        # Remove special characters and replace spaces
        sanitized = re.sub(r"[^\w\s-]", "", text)
        sanitized = re.sub(r"[-\s]+", "_", sanitized)
        # Truncate to max length
        return sanitized[:max_length].strip("_")

    def _export_to_local(self, data: pd.DataFrame, query: str, output_dir: Path) -> str:
        """
        Export results to local CSV file.

        Args:
            data: Query results DataFrame
            query: Original query for filename
            output_dir: Output directory path

        Returns:
            Path to exported file

        Raises:
            ExportError: If export fails

        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            safe_query = self._sanitize_filename(query)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_query}_{timestamp}.csv"
            filepath = output_dir / filename

            # Export to CSV
            data.to_csv(filepath, index=False)

            logger.info(f"📤 Exported {len(data)} rows to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Local export failed: {e}")
            raise ExportError(f"Failed to export to local file: {e}") from e

    async def _export_to_s3_async(
        self, data: pd.DataFrame, query: str, bucket: str, prefix: str
    ) -> str:
        """
        Export results to S3 using boto3 (async).

        Args:
            data: Query results DataFrame
            query: Original query for filename
            bucket: S3 bucket name
            prefix: S3 key prefix

        Returns:
            S3 URI of exported file

        Raises:
            ExportError: If S3 export fails

        """
        try:
            import boto3

            # Generate S3 key
            safe_query = self._sanitize_filename(query)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"{prefix}{safe_query}_{timestamp}.csv"

            # Convert DataFrame to CSV in memory
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)

            # Upload to S3 (run in executor to avoid blocking)
            import asyncio

            loop = asyncio.get_event_loop()

            def upload():
                s3_client = boto3.client("s3")
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=csv_buffer.getvalue().encode("utf-8"),
                    ContentType="text/csv",
                )

            await loop.run_in_executor(None, upload)

            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"📤 Exported {len(data)} rows to {s3_uri}")

            return s3_uri

        except Exception as e:
            logger.error(f"S3 export failed: {e}")
            raise ExportError(f"Failed to export to S3: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with processor statistics

        """
        return {
            "config": self.config,
            "environment": self._environment,
            "rag_engine_initialized": self.rag_engine is not None,
            "pipeline_initialized": self.pipeline is not None,
        }

    def close(self) -> None:
        """Clean up resources."""
        try:
            if self.rag_engine:
                self.rag_engine.close()
            logger.info("✅ QueryProcessor closed successfully")
        except Exception as e:
            logger.error(f"Error closing QueryProcessor: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
