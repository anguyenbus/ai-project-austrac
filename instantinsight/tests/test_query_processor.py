"""
Unit tests for QueryProcessor core functionality.

Tests cover synchronous processing, validation, error handling,
and basic integration with the pipeline.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    ValidationError,
)
from src.rag.pipeline.stages import PipelineResult, Stage, StageResult


class TestQueryProcessorInitialization:
    """Test QueryProcessor initialization and configuration."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_initialization_with_defaults(self, mock_create_instance):
        """Test processor initializes with default configuration."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        # Act
        processor = QueryProcessor()

        # Assert
        assert processor.rag_engine is not None
        assert processor.pipeline is not None
        assert processor.config["enable_cache"] is True
        assert processor.config["export_enabled"] is True
        assert processor.config["is_lambda"] is False

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_initialization_with_custom_config(self, mock_create_instance):
        """Test processor accepts custom configuration."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        custom_config: QueryProcessorConfig = {
            "enable_cache": False,
            "export_enabled": False,
            "query_timeout_seconds": 120.0,
            "max_result_rows": 50000,
        }

        # Act
        processor = QueryProcessor(config=custom_config)

        # Assert
        assert processor.config["enable_cache"] is False
        assert processor.config["export_enabled"] is False
        assert processor.config["query_timeout_seconds"] == 120.0
        assert processor.config["max_result_rows"] == 50000

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_initialization_with_provided_components(self, mock_create_instance):
        """Test processor accepts pre-configured components."""
        # Arrange
        mock_engine = Mock()
        mock_pipeline = Mock()

        # Act
        processor = QueryProcessor(rag_engine=mock_engine, pipeline=mock_pipeline)

        # Assert
        assert processor.rag_engine is mock_engine
        assert processor.pipeline is mock_pipeline
        # Should not call create_instance when components provided
        mock_create_instance.assert_not_called()

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_initialization_failure_raises_error(self, mock_create_instance):
        """Test initialization raises error when engine creation fails."""
        # Arrange
        mock_create_instance.return_value = (None, "Engine creation failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to create RAGEngine"):
            QueryProcessor()


class TestEnvironmentDetection:
    """Test environment detection logic."""

    @patch("src.query_processor.RAGEngine.create_instance")
    @patch.dict("os.environ", {}, clear=True)
    def test_detect_local_environment(self, mock_create_instance):
        """Test detection of local development environment."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        # Act
        processor = QueryProcessor()

        # Assert
        assert processor._environment["is_lambda"] is False
        assert processor._environment["export_strategy"] == "local"
        assert "output_dir" in processor._environment

    @patch("src.query_processor.RAGEngine.create_instance")
    @patch.dict(
        "os.environ",
        {
            "AWS_LAMBDA_FUNCTION_NAME": "my-lambda",
            "RESULTS_BUCKET": "my-bucket",
            "RESULTS_PREFIX": "results/",
        },
    )
    def test_detect_lambda_environment(self, mock_create_instance):
        """Test detection of Lambda environment."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        # Act
        processor = QueryProcessor()

        # Assert
        assert processor._environment["is_lambda"] is True
        assert processor._environment["export_strategy"] == "s3"
        assert processor._environment["s3_bucket"] == "my-bucket"
        assert processor._environment["s3_prefix"] == "results/"


class TestQueryValidation:
    """Test query validation logic."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_validate_empty_query_raises_error(self, mock_create_instance):
        """Test validation rejects empty queries."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act & Assert
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            processor._validate_query("")

        with pytest.raises(ValidationError, match="Query cannot be empty"):
            processor._validate_query("   ")

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_validate_long_query_raises_error(self, mock_create_instance):
        """Test validation rejects excessively long queries."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()
        long_query = "x" * 5001

        # Act & Assert
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            processor._validate_query(long_query)

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_validate_dangerous_patterns_logs_warning(self, mock_create_instance):
        """Test validation logs warnings for dangerous SQL patterns."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act - should not raise, just log
        processor._validate_query("Show me data but also DROP TABLE users")
        # NOTE: This logs a warning but doesn't block (AI-generated SQL context)


class TestProcessQuerySuccess:
    """Test successful query processing flows."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_process_query_success(self, mock_create_instance):
        """Test successful query processing flow."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Mock successful pipeline result
        mock_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        pipeline_result = PipelineResult(
            query="Show top products",
            sql="SELECT * FROM products LIMIT 10",
            success=True,
        )
        pipeline_result.stages = {
            Stage.CACHE_LOOKUP: StageResult(
                stage=Stage.CACHE_LOOKUP, success=False, duration=0.01
            ),
            Stage.SQL_GENERATION: StageResult(
                stage=Stage.SQL_GENERATION, success=True, duration=0.5
            ),
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.3
            ),
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = processor.process_query("Show top products", export_results=False)

        # Assert
        assert result["success"] is True
        assert result["sql"] == "SELECT * FROM products LIMIT 10"
        assert result["row_count"] == 3
        assert result["cache_hit"] is False
        assert "total_duration_ms" in result
        assert "sql_generation_ms" in result

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_process_query_with_cache_hit(self, mock_create_instance):
        """Test query processing with cache hit."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2]})
        cache_data = Mock()
        cache_data.confidence = 0.95

        pipeline_result = PipelineResult(
            query="Show products", sql="SELECT * FROM products", success=True
        )
        pipeline_result.stages = {
            Stage.CACHE_LOOKUP: StageResult(
                stage=Stage.CACHE_LOOKUP, success=True, data=cache_data, duration=0.01
            ),
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.1
            ),
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = processor.process_query("Show products", export_results=False)

        # Assert
        assert result["success"] is True
        assert result["cache_hit"] is True
        assert result["cache_confidence"] == 0.95


class TestProcessQueryFailures:
    """Test query processing error handling."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_process_query_validation_error(self, mock_create_instance):
        """Test error handling for invalid queries."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act
        result = processor.process_query("", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "validation_error"
        assert "cannot be empty" in result["error"]

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_process_query_generation_error(self, mock_create_instance):
        """Test error handling for SQL generation failure."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Mock failed pipeline result
        pipeline_result = PipelineResult(
            query="Show products",
            sql=None,
            success=False,
            error="Failed to generate SQL",
        )
        pipeline_result.stages = {}

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = processor.process_query("Show products", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "generation_error"
        assert "Failed to generate SQL" in result["error"]

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_process_query_unexpected_error(self, mock_create_instance):
        """Test error handling for unexpected exceptions."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Mock pipeline raising exception
        processor.pipeline.process = Mock(
            side_effect=RuntimeError("Unexpected pipeline error")
        )

        # Act
        result = processor.process_query("Show products", export_results=False)

        # Assert
        assert result["success"] is False
        assert result["error_type"] == "unexpected_error"
        assert "Unexpected pipeline error" in result["error"]


class TestFilenameSanitization:
    """Test filename sanitization for exports."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_sanitize_filename_removes_special_chars(self, mock_create_instance):
        """Test sanitization removes special characters."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act
        result = processor._sanitize_filename("Show me top 10 products?")

        # Assert
        assert "?" not in result
        assert " " not in result
        assert "_" in result

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_sanitize_filename_truncates_long_names(self, mock_create_instance):
        """Test sanitization truncates long filenames."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        long_query = "x" * 100

        # Act
        result = processor._sanitize_filename(long_query, max_length=50)

        # Assert
        assert len(result) <= 50


class TestLocalExport:
    """Test local file export functionality."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_export_to_local_creates_file(self, mock_create_instance, tmp_path):
        """Test local export creates CSV file."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        output_dir = tmp_path / "results"

        # Act
        export_path = processor._export_to_local(mock_data, "test query", output_dir)

        # Assert
        assert Path(export_path).exists()
        assert export_path.endswith(".csv")
        assert "test_query" in export_path

        # Verify content
        loaded_data = pd.read_csv(export_path)
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ["col1", "col2"]

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_export_to_local_creates_directory(self, mock_create_instance, tmp_path):
        """Test export creates output directory if missing."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        mock_data = pd.DataFrame({"col1": [1, 2]})
        output_dir = tmp_path / "new_dir" / "results"

        # Act
        export_path = processor._export_to_local(mock_data, "test", output_dir)

        # Assert
        assert output_dir.exists()
        assert Path(export_path).exists()


class TestProcessorUtilities:
    """Test utility methods."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_get_stats_returns_info(self, mock_create_instance):
        """Test get_stats returns processor information."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        # Act
        stats = processor.get_stats()

        # Assert
        assert "config" in stats
        assert "environment" in stats
        assert "rag_initialized" in stats
        assert stats["rag_initialized"] is True

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_context_manager_support(self, mock_create_instance):
        """Test QueryProcessor works as context manager."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        # Act & Assert
        with QueryProcessor() as processor:
            assert processor is not None
            assert processor.rag_engine is not None

        # Verify close was called
        mock_engine.close.assert_called_once()


class TestIntegrationScenarios:
    """Integration tests with more complete flows."""

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_end_to_end_query_with_export(self, mock_create_instance, tmp_path):
        """Test complete flow with export."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)

        config: QueryProcessorConfig = {
            "export_enabled": True,
            "local_output_dir": str(tmp_path),
        }
        processor = QueryProcessor(config=config)

        mock_data = pd.DataFrame({"product": ["A", "B"], "revenue": [100, 200]})
        pipeline_result = PipelineResult(
            query="Show revenue", sql="SELECT * FROM products", success=True
        )
        pipeline_result.stages = {
            Stage.QUERY_EXECUTION: StageResult(
                stage=Stage.QUERY_EXECUTION, success=True, data=mock_data, duration=0.2
            )
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = processor.process_query(
            "Show revenue", export_results=True, generate_visualization=False
        )

        # Assert
        assert result["success"] is True
        assert result["export_path"] is not None
        assert Path(result["export_path"]).exists()
        assert result["export_type"] == "local"
        assert result["row_count"] == 2

    @patch("src.query_processor.RAGEngine.create_instance")
    def test_pipeline_stages_captured(self, mock_create_instance):
        """Test pipeline stages are captured in result."""
        # Arrange
        mock_engine = Mock()
        mock_create_instance.return_value = (mock_engine, None)
        processor = QueryProcessor()

        pipeline_result = PipelineResult(query="test", sql="SELECT 1", success=True)
        pipeline_result.stages = {
            Stage.QUERY_VALIDATION: StageResult(
                stage=Stage.QUERY_VALIDATION, success=True, duration=0.01
            ),
            Stage.CACHE_LOOKUP: StageResult(
                stage=Stage.CACHE_LOOKUP, success=False, duration=0.02
            ),
            Stage.SQL_GENERATION: StageResult(
                stage=Stage.SQL_GENERATION, success=True, duration=0.5
            ),
        }

        processor.pipeline.process = Mock(return_value=pipeline_result)

        # Act
        result = processor.process_query("test", export_results=False)

        # Assert
        assert "pipeline_stages" in result
        assert len(result["pipeline_stages"]) == 3
        assert "query_validation" in result["pipeline_stages"]
        assert "cache_lookup" in result["pipeline_stages"]
        assert "sql_generation" in result["pipeline_stages"]
