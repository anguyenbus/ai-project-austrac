"""Tests for RAGEngine with conversation context."""

from unittest.mock import Mock

import pytest

from src.rag.engines.query_executor import QueryExecutor
from src.rag.engines.rag_engine import RAGEngine
from src.rag.engines.sql_engine import SQLEngine


class TestRAGEngineContext:
    """Test RAGEngine with conversation context."""

    def test_generate_sql_passes_prior_turns(self):
        """Test that generate_sql passes prior_turns to SQL engine."""
        # Create mock SQL engine
        mock_sql_engine = Mock(spec=SQLEngine)
        mock_result = Mock()
        mock_result.success = True
        mock_result.sql = "SELECT * FROM test"
        mock_result.schema_context = "test context"
        mock_result.normalized_query = "normalized query"
        mock_sql_engine.generate_sql.return_value = mock_result

        # Create mock query executor
        mock_query_executor = Mock(spec=QueryExecutor)

        # Create RAGEngine with mocks
        engine = RAGEngine(
            sql_engine=mock_sql_engine, query_executor=mock_query_executor
        )

        # Test with prior_turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]
        result = engine.generate_sql(
            question="Current query", prior_turns=prior_turns, return_context=True
        )

        # Verify SQL engine was called with prior_turns
        mock_sql_engine.generate_sql.assert_called_once_with(
            "Current query", prior_turns=prior_turns
        )

        # Verify result format
        assert result["sql"] == "SELECT * FROM test"
        assert result["schema_context"] == "test context"
        assert result["normalized_query"] == "normalized query"

    def test_generate_sql_without_prior_turns(self):
        """Test that generate_sql works without prior_turns."""
        # Create mock SQL engine
        mock_sql_engine = Mock(spec=SQLEngine)
        mock_result = Mock()
        mock_result.success = True
        mock_result.sql = "SELECT * FROM test"
        mock_sql_engine.generate_sql.return_value = mock_result

        # Create mock query executor
        mock_query_executor = Mock(spec=QueryExecutor)

        # Create RAGEngine with mocks
        engine = RAGEngine(
            sql_engine=mock_sql_engine, query_executor=mock_query_executor
        )

        # Test without prior_turns
        result = engine.generate_sql(question="Current query")

        # Verify SQL engine was called with None prior_turns
        mock_sql_engine.generate_sql.assert_called_once_with(
            "Current query", prior_turns=None
        )

        # Verify result
        assert result == "SELECT * FROM test"

    def test_generate_sql_with_empty_prior_turns(self):
        """Test that generate_sql works with empty prior_turns."""
        # Create mock SQL engine
        mock_sql_engine = Mock(spec=SQLEngine)
        mock_result = Mock()
        mock_result.success = True
        mock_result.sql = "SELECT * FROM test"
        mock_sql_engine.generate_sql.return_value = mock_result

        # Create mock query executor
        mock_query_executor = Mock(spec=QueryExecutor)

        # Create RAGEngine with mocks
        engine = RAGEngine(
            sql_engine=mock_sql_engine, query_executor=mock_query_executor
        )

        # Test with empty prior_turns
        result = engine.generate_sql(question="Current query", prior_turns=[])

        # Verify SQL engine was called with empty list
        mock_sql_engine.generate_sql.assert_called_once_with(
            "Current query", prior_turns=[]
        )

        # Verify result
        assert result == "SELECT * FROM test"

    def test_generate_sql_handles_clarification_with_prior_turns(self):
        """Test that generate_sql handles clarification with prior_turns."""
        # Create mock SQL engine
        mock_sql_engine = Mock(spec=SQLEngine)
        mock_result = Mock()
        mock_result.success = False
        mock_result.needs_clarification = True
        mock_result.clarification_message = "Please specify table"
        mock_result.normalized_query = "normalized query"
        mock_sql_engine.generate_sql.return_value = mock_result

        # Create mock query executor
        mock_query_executor = Mock(spec=QueryExecutor)

        # Create RAGEngine with mocks
        engine = RAGEngine(
            sql_engine=mock_sql_engine, query_executor=mock_query_executor
        )

        # Test with prior_turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]
        result = engine.generate_sql(
            question="Ambiguous query", prior_turns=prior_turns, return_context=True
        )

        # Verify SQL engine was called with prior_turns
        mock_sql_engine.generate_sql.assert_called_once_with(
            "Ambiguous query", prior_turns=prior_turns
        )

        # Verify clarification result
        assert result["sql"].startswith("CANNOT FIND TABLES:")
        assert result["schema_context"] == ""
        assert result["normalized_query"] == "normalized query"

    def test_generate_sql_handles_error_with_prior_turns(self):
        """Test that generate_sql handles errors with prior_turns."""
        # Create mock SQL engine
        mock_sql_engine = Mock(spec=SQLEngine)
        mock_result = Mock()
        mock_result.success = False
        mock_result.needs_clarification = False
        mock_result.error = "SQL generation failed"
        mock_sql_engine.generate_sql.return_value = mock_result

        # Create mock query executor
        mock_query_executor = Mock(spec=QueryExecutor)

        # Create RAGEngine with mocks
        engine = RAGEngine(
            sql_engine=mock_sql_engine, query_executor=mock_query_executor
        )

        # Test with prior_turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to generate SQL"):
            engine.generate_sql(question="Failing query", prior_turns=prior_turns)

        # Verify SQL engine was called with prior_turns
        mock_sql_engine.generate_sql.assert_called_once_with(
            "Failing query", prior_turns=prior_turns
        )
