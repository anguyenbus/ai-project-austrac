"""
Async-specific tests for QueryProcessor.

Tests cover asynchronous query processing, timeout handling,
and Lambda-specific functionality.
"""

import asyncio
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.query_processor import QueryProcessor, QueryProcessorConfig
from src.rag.pipeline.stages import PipelineResult, Stage, StageResult


class TestAsyncQueryProcessing:
    """Test asynchronous query processing."""

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    async def test_process_query_async_success(self, mock_create_instance):
        """Test async query processing succeeds."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2, 3]})
        pipeline_result = PipelineResult(
            query="Show data", sql="SELECT * FROM table", success=True
        )
        pipeline_result.stages = {
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.1
            )
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = await processor.process_query_async("Show data", export_results=False)

        # Assert
        assert result["success"] is True
        assert result["sql"] == "SELECT * FROM table"
        assert result["row_count"] == 3

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    async def test_process_query_async_timeout(self, mock_create_instance):
        """Test async timeout handling."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        config: QueryProcessorConfig = {
            "query_timeout_seconds": 0.05,  # Very short timeout
        }
        processor = QueryProcessor(config=config)

        # Mock slow pipeline - return value that will take too long
        def slow_pipeline(*args, **kwargs):
            import time

            time.sleep(1.0)  # Longer than timeout
            return PipelineResult(query="test", success=True)

        processor.pipeline.process = Mock(side_effect=slow_pipeline)

        # Act
        result = await processor.process_query_async("test", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "timeout_error"
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    async def test_process_query_async_validation_error(self, mock_create_instance):
        """Test async validation error handling."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act
        result = await processor.process_query_async("", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "validation_error"


class TestAsyncS3Export:
    """Test async S3 export functionality."""

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    @patch("boto3.client")
    async def test_export_to_s3_success(self, mock_boto3_client, mock_create_instance):
        """Test S3 export succeeds."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        # Mock S3 client
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Act
        s3_uri = await processor._export_to_s3_async(
            mock_data, "test query", "my-bucket", "results/"
        )

        # Assert
        assert s3_uri.startswith("s3://my-bucket/results/")
        assert s3_uri.endswith(".csv")
        mock_s3_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    @patch("boto3.client")
    async def test_export_to_s3_handles_failure(
        self, mock_boto3_client, mock_create_instance
    ):
        """Test S3 export error handling."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2]})

        # Mock S3 client to fail
        mock_boto3_client.side_effect = Exception("S3 connection failed")

        # Act & Assert
        from src.query_processor import ExportError

        with pytest.raises(ExportError, match="Failed to export to S3"):
            await processor._export_to_s3_async(mock_data, "test", "bucket", "prefix/")


class TestConcurrentQueries:
    """Test concurrent async query processing."""

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    async def test_concurrent_queries(self, mock_create_instance):
        """Test multiple concurrent queries."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Mock successful pipeline results
        pipeline_result = PipelineResult(query="test", sql="SELECT 1", success=True)
        pipeline_result.stages = {}
        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act - run 3 queries concurrently
        tasks = [
            processor.process_query_async(f"query {i}", export_results=False)
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 3
        assert all(r["success"] is True for r in results)


class TestLambdaIntegration:
    """Test Lambda-specific scenarios."""

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    @patch.dict(
        "os.environ",
        {
            "AWS_LAMBDA_FUNCTION_NAME": "query-processor",
            "RESULTS_BUCKET": "my-results",
            "RESULTS_PREFIX": "queries/",
        },
    )
    async def test_lambda_environment_detected(self, mock_create_instance):
        """Test Lambda environment is detected correctly."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        # Act
        processor = QueryProcessor()

        # Assert
        assert processor.config["is_lambda"] is True
        assert processor.config["s3_bucket"] == "my-results"
        assert processor.config["s3_prefix"] == "queries/"
        assert processor._environment["export_strategy"] == "s3"

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    @patch("boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_LAMBDA_FUNCTION_NAME": "query-processor", "RESULTS_BUCKET": "bucket"},
    )
    async def test_lambda_with_s3_export(self, mock_boto3_client, mock_create_instance):
        """Test Lambda query processing with S3 export."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"result": [1, 2, 3]})
        pipeline_result = PipelineResult(
            query="test", sql="SELECT * FROM table", success=True
        )
        pipeline_result.stages = {
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.1
            )
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Mock S3 client
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Act
        result = await processor.process_query_async("test query", export_results=True)

        # Assert
        assert result["success"] is True
        assert result["export_type"] == "s3"
        assert result["export_path"].startswith("s3://bucket/")
        mock_s3_client.put_object.assert_called_once()


class TestAsyncErrorRecovery:
    """Test async error recovery and graceful degradation."""

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    @patch("pathlib.Path.stat")
    async def test_export_failure_does_not_block_result(
        self, mock_stat, mock_create_instance
    ):
        """Test that export failures don't prevent result return."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2]})
        pipeline_result = PipelineResult(
            query="test", sql="SELECT * FROM table", success=True
        )
        pipeline_result.stages = {
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.1
            )
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Mock stat to fail (simulates export error during file size check)
        mock_stat.side_effect = Exception("Export failed")

        # Act
        result = await processor.process_query_async("test", export_results=True)

        # Assert - query succeeded even though export failed
        assert result["success"] is True
        assert result["sql"] == "SELECT * FROM table"
        assert "Export failed" in result.get("error", "")

    @pytest.mark.asyncio
    @patch("src.query_processor.RAGEngine.create_instance")
    async def test_pipeline_exception_handled_gracefully(self, mock_create_instance):
        """Test pipeline exceptions are handled gracefully."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Mock pipeline to raise exception
        processor.pipeline.process = Mock(side_effect=RuntimeError("Pipeline crashed"))

        # Act
        result = await processor.process_query_async("test", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "unexpected_error"
        assert "Pipeline crashed" in result["error"]
        assert "total_duration_ms" in result
