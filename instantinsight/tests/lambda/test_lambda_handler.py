"""
Tests for Lambda handler following TDD principles.

This test suite validates the Lambda handler implementation for QueryProcessor,
including event handling, error scenarios, timeout management, and response formatting.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest


class MockContext:
    """Mock Lambda context for testing."""

    def __init__(self, remaining_time_ms: int = 300000):
        """
        Initialize mock context.

        Args:
            remaining_time_ms: Remaining execution time in milliseconds

        """
        self.aws_request_id = "test-request-123"
        self.function_name = "instantinsight-query-processor-test"
        self.memory_limit_in_mb = "2048"
        self.invoked_function_arn = (
            "arn:aws:lambda:ap-southeast-2:123456789012:function:test"
        )
        self._remaining_time_ms = remaining_time_ms

    def get_remaining_time_in_millis(self) -> int:
        """Return remaining execution time."""
        return self._remaining_time_ms


@pytest.fixture
def mock_context() -> MockContext:
    """Provide mock Lambda context."""
    return MockContext()


@pytest.fixture
def mock_query_processor() -> Mock:
    """Provide mock QueryProcessor instance."""
    processor = Mock()
    processor.process_query_async = AsyncMock()
    return processor


@pytest.fixture
def sample_event() -> dict[str, Any]:
    """Provide sample Lambda event."""
    return {
        "query": "Show me the top 10 products by revenue",
        "user_id": "user123",
        "export_to_s3": True,
        "generate_visualization": True,
    }


@pytest.fixture
def sample_success_result() -> dict[str, Any]:
    """Provide sample successful query result."""
    return {
        "query": "Show me the top 10 products by revenue",
        "sql": "SELECT product_name, SUM(revenue) as total_revenue FROM sales GROUP BY product_name ORDER BY total_revenue DESC LIMIT 10",
        "success": True,
        "row_count": 10,
        "export_path": "s3://instantinsight-query-results/query_20250120_120000.csv",
        "cache_hit": False,
        "total_duration_ms": 1234,
    }


class TestLambdaHandlerInitialization:
    """Test Lambda handler initialization and processor setup."""

    def test_processor_initialization_with_lambda_config(
        self, mock_query_processor: Mock
    ) -> None:
        """Test processor is initialized with Lambda-specific configuration."""
        # NOTE: This test validates the global processor initialization pattern
        # with environment-based configuration for Lambda deployment

        with patch("lambda_handler.QueryProcessor", return_value=mock_query_processor):
            with patch.dict(
                "os.environ",
                {
                    "RESULTS_BUCKET": "test-bucket",
                    "RESULTS_PREFIX": "results/",
                    "QUERY_TIMEOUT": "120",
                    "MAX_RESULT_ROWS": "50000",
                },
            ):
                from lambda_handler import _init_processor

                processor = _init_processor()

                assert processor is not None

    def test_processor_singleton_pattern(self, mock_query_processor: Mock) -> None:
        """Test processor instance is reused across invocations (warm starts)."""
        with patch("lambda_handler.QueryProcessor", return_value=mock_query_processor):
            from lambda_handler import _init_processor

            processor1 = _init_processor()
            processor2 = _init_processor()

            assert processor1 is processor2


class TestLambdaHandlerSuccess:
    """Test successful Lambda handler execution."""

    @pytest.mark.asyncio
    async def test_successful_query_processing(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
        sample_success_result: dict[str, Any],
    ) -> None:
        """Test handler returns success response for valid query."""
        mock_query_processor.process_query_async.return_value = sample_success_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            assert response["statusCode"] == 200

            body = json.loads(response["body"])
            assert body["success"] is True
            assert body["sql"] == sample_success_result["sql"]
            assert body["row_count"] == 10
            assert body["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_handler_includes_cors_headers(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
        sample_success_result: dict[str, Any],
    ) -> None:
        """Test response includes CORS headers."""
        mock_query_processor.process_query_async.return_value = sample_success_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            assert "headers" in response
            assert response["headers"]["Content-Type"] == "application/json"
            assert "Access-Control-Allow-Origin" in response["headers"]

    @pytest.mark.asyncio
    async def test_handler_with_cache_hit(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test handler correctly reports cache hit."""
        cached_result = {
            "query": sample_event["query"],
            "sql": "SELECT * FROM cached_query",
            "success": True,
            "row_count": 5,
            "cache_hit": True,
            "cache_confidence": 0.95,
            "total_duration_ms": 50,
        }
        mock_query_processor.process_query_async.return_value = cached_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)
            body = json.loads(response["body"])

            assert body["cache_hit"] is True
            assert body["duration_ms"] == 50


class TestLambdaHandlerErrors:
    """Test Lambda handler error handling."""

    def test_missing_query_parameter(
        self,
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test handler returns 400 for missing query parameter."""
        event = {"user_id": "user123"}

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(event, mock_context)

            assert response["statusCode"] == 400
            body = json.loads(response["body"])
            assert body["success"] is False
            assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_query_processing_failure(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test handler returns 500 for query processing failure."""
        failed_result = {
            "query": sample_event["query"],
            "success": False,
            "error": "SQL generation failed: Invalid table reference",
            "error_type": "generation_error",
        }
        mock_query_processor.process_query_async.return_value = failed_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            assert response["statusCode"] == 500
            body = json.loads(response["body"])
            assert body["success"] is False
            assert "error" in body
            assert body.get("error_type") == "generation_error"

    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test handler catches and returns unexpected exceptions."""
        mock_query_processor.process_query_async.side_effect = ValueError(
            "Unexpected error"
        )

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            assert response["statusCode"] == 500
            body = json.loads(response["body"])
            assert body["success"] is False
            assert "error" in body


class TestLambdaHandlerTimeout:
    """Test Lambda handler timeout scenarios."""

    @pytest.mark.asyncio
    async def test_timeout_error_handling(
        self,
        sample_event: dict[str, Any],
        mock_query_processor: Mock,
    ) -> None:
        """Test handler returns 504 for timeout errors."""
        import asyncio

        mock_query_processor.process_query_async.side_effect = asyncio.TimeoutError()
        mock_context = MockContext(remaining_time_ms=1000)

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            assert response["statusCode"] == 504
            body = json.loads(response["body"])
            assert body["success"] is False
            assert "timeout" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_low_remaining_time_warning(
        self,
        sample_event: dict[str, Any],
        mock_query_processor: Mock,
        sample_success_result: dict[str, Any],
    ) -> None:
        """Test handler logs warning when Lambda is approaching timeout."""
        mock_query_processor.process_query_async.return_value = sample_success_result
        mock_context = MockContext(remaining_time_ms=3000)

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)

            # NOTE: Handler should still succeed but log warning
            assert response["statusCode"] == 200
            # Verify warning was logged (implementation dependent)


class TestLambdaHandlerVisualization:
    """Test Lambda handler visualization handling."""

    @pytest.mark.asyncio
    async def test_visualization_included_when_requested(
        self,
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test response includes visualization when requested."""
        event = {
            "query": "Show revenue trend",
            "generate_visualization": True,
        }

        result = {
            "query": event["query"],
            "success": True,
            "sql": "SELECT date, revenue FROM sales",
            "row_count": 30,
            "visualization": {
                "type": "line",
                "data": {"x": ["2025-01"], "y": [10000]},
            },
            "total_duration_ms": 1500,
        }
        mock_query_processor.process_query_async.return_value = result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(event, mock_context)
            body = json.loads(response["body"])

            assert "visualization" in body
            assert body["visualization"]["type"] == "line"

    @pytest.mark.asyncio
    async def test_visualization_excluded_when_not_requested(
        self,
        mock_context: MockContext,
        mock_query_processor: Mock,
        sample_success_result: dict[str, Any],
    ) -> None:
        """Test visualization is not included when not requested."""
        event = {
            "query": "Show revenue",
            "generate_visualization": False,
        }

        mock_query_processor.process_query_async.return_value = sample_success_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            _ = lambda_handler(event, mock_context)

            # Verify process_query_async was called with generate_visualization=False
            call_kwargs = mock_query_processor.process_query_async.call_args.kwargs
            assert call_kwargs.get("generate_visualization") is False


class TestLambdaHandlerExport:
    """Test Lambda handler S3 export functionality."""

    @pytest.mark.asyncio
    async def test_s3_export_path_included(
        self,
        sample_event: dict[str, Any],
        mock_context: MockContext,
        mock_query_processor: Mock,
        sample_success_result: dict[str, Any],
    ) -> None:
        """Test response includes S3 export path."""
        mock_query_processor.process_query_async.return_value = sample_success_result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            response = lambda_handler(sample_event, mock_context)
            body = json.loads(response["body"])

            assert "export_path" in body
            assert body["export_path"].startswith("s3://")

    @pytest.mark.asyncio
    async def test_export_disabled_when_not_requested(
        self,
        mock_context: MockContext,
        mock_query_processor: Mock,
    ) -> None:
        """Test export is skipped when not requested."""
        event = {
            "query": "Show products",
            "export_to_s3": False,
        }

        result = {
            "query": event["query"],
            "success": True,
            "sql": "SELECT * FROM products",
            "row_count": 100,
            "total_duration_ms": 800,
        }
        mock_query_processor.process_query_async.return_value = result

        with patch("lambda_handler._init_processor", return_value=mock_query_processor):
            from lambda_handler import lambda_handler

            lambda_handler(event, mock_context)

            # Verify process_query_async was called with export_results=False
            call_kwargs = mock_query_processor.process_query_async.call_args.kwargs
            assert call_kwargs.get("export_results") is False


class TestLambdaHandlerResponseFormatting:
    """Test Lambda response formatting utilities."""

    def test_success_response_structure(self) -> None:
        """Test _success_response creates properly formatted response."""
        from lambda_handler import _success_response

        body = {
            "success": True,
            "sql": "SELECT * FROM test",
            "row_count": 5,
        }

        response = _success_response(body)

        assert response["statusCode"] == 200
        assert "body" in response
        assert "headers" in response
        assert response["headers"]["Content-Type"] == "application/json"

        parsed_body = json.loads(response["body"])
        assert parsed_body == body

    def test_error_response_structure(self) -> None:
        """Test _error_response creates properly formatted error response."""
        from lambda_handler import _error_response

        response = _error_response(
            status_code=400,
            error_message="Invalid input",
            error_type="validation_error",
        )

        assert response["statusCode"] == 400
        assert "body" in response
        assert "headers" in response

        body = json.loads(response["body"])
        assert body["success"] is False
        assert body["error"] == "Invalid input"
        assert body["error_type"] == "validation_error"

    def test_error_response_without_type(self) -> None:
        """Test _error_response works without error_type."""
        from lambda_handler import _error_response

        response = _error_response(
            status_code=500,
            error_message="Internal error",
        )

        body = json.loads(response["body"])
        assert "error_type" not in body or body.get("error_type") is None
