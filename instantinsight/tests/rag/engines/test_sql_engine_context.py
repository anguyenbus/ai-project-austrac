"""Tests for SQL engine with conversation context."""

from unittest.mock import Mock

import pytest

from src.rag.engines.sql_engine import SQLEngine
from src.rag.engines.types import SQLGenerationStatus


class TestSQLEngineContext:
    """Test SQL engine context threading."""

    def test_sql_engine_passes_context_to_normalizer(self):
        """Test SQL engine passes prior_turns to query normalizer."""
        # Create mock dependencies
        mock_normalizer = Mock()
        mock_normalizer.normalize.return_value = Mock(
            main_clause="test query",
            details_for_filterings=[],
            required_visuals=None,
            tables=None,
            model_dump=lambda exclude_none: {"main_clause": "test query"},
        )

        mock_table_agent = Mock()
        mock_table_agent.select_tables.return_value = Mock(
            selected_tables=["test_table"], schema_context="schema info"
        )

        mock_clarification_agent = Mock()
        mock_clarification_agent.needs_clarification.return_value = False

        mock_rag = Mock()
        mock_rag.find_relevant_examples.return_value = {
            "schema_context": [],
            "example_context": [],
            "selected_tables": ["test_table"],
            "search_results": [],
            "best_similarity": 0.8,
        }

        mock_schema_validator = Mock()
        mock_schema_validator.validate_sql_tables.return_value = Mock(
            validation_passed=True
        )

        # Create SQL engine with mocked normalizer
        engine = SQLEngine(
            rag_instance=mock_rag,
            table_agent=mock_table_agent,
            schema_validator=mock_schema_validator,
            clarification_agent=mock_clarification_agent,
            query_normalizer=mock_normalizer,
        )

        # Test with prior turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]

        result = engine.generate_sql("New query", prior_turns)

        # Verify normalizer was called with prior_turns
        mock_normalizer.normalize.assert_called_once_with("New query", prior_turns)

        # Verify result structure
        assert result.status == SQLGenerationStatus.SUCCESS

    def test_sql_engine_passes_context_to_sql_generation(self):
        """Test SQL engine passes prior_turns to SQL generation."""
        # Create mock dependencies
        mock_normalizer = Mock()
        mock_normalizer.normalize.return_value = Mock(
            main_clause="test query",
            details_for_filterings=[],
            required_visuals=None,
            tables=None,
            model_dump=lambda exclude_none: {"main_clause": "test query"},
        )

        mock_table_agent = Mock()
        mock_table_agent.select_tables.return_value = Mock(
            selected_tables=["test_table"], schema_context="schema info"
        )

        mock_clarification_agent = Mock()
        mock_clarification_agent.needs_clarification.return_value = False

        mock_rag = Mock()
        mock_rag.find_relevant_examples.return_value = {
            "schema_context": [],
            "example_context": [],
            "selected_tables": ["test_table"],
            "search_results": [],
            "best_similarity": 0.8,
        }

        mock_schema_validator = Mock()
        mock_schema_validator.validate_sql_tables.return_value = Mock(
            validation_passed=True
        )

        # Mock SQL writer to track prior_turns
        mock_sql_writer = Mock()
        mock_sql_writer.generate_sql.return_value = {"sql": "SELECT * FROM test_table"}

        # Create SQL engine
        engine = SQLEngine(
            rag_instance=mock_rag,
            table_agent=mock_table_agent,
            schema_validator=mock_schema_validator,
            clarification_agent=mock_clarification_agent,
            query_normalizer=mock_normalizer,
        )

        # Test with prior turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]

        # Mock the SQLWriterAgent import and instance
        with pytest.mock.patch(
            "src.rag.engines.sql_engine.SQLWriterAgent"
        ) as mock_writer_class:
            mock_writer_class.return_value = mock_sql_writer

            engine.generate_sql("New query", prior_turns)

            # Verify SQL writer was called with prior_turns
            mock_sql_writer.generate_sql.assert_called_once()
            call_args = mock_sql_writer.generate_sql.call_args
            assert call_args.kwargs["prior_turns"] == prior_turns

    def test_sql_engine_without_prior_turns(self):
        """Test SQL engine works without prior_turns (backward compatibility)."""
        # Create mock dependencies
        mock_normalizer = Mock()
        mock_normalizer.normalize.return_value = Mock(
            main_clause="test query",
            details_for_filterings=[],
            required_visuals=None,
            tables=None,
            model_dump=lambda exclude_none: {"main_clause": "test query"},
        )

        mock_table_agent = Mock()
        mock_table_agent.select_tables.return_value = Mock(
            selected_tables=["test_table"], schema_context="schema info"
        )

        mock_clarification_agent = Mock()
        mock_clarification_agent.needs_clarification.return_value = False

        mock_rag = Mock()
        mock_rag.find_relevant_examples.return_value = {
            "schema_context": [],
            "example_context": [],
            "selected_tables": ["test_table"],
            "search_results": [],
            "best_similarity": 0.8,
        }

        mock_schema_validator = Mock()
        mock_schema_validator.validate_sql_tables.return_value = Mock(
            validation_passed=True
        )

        # Create SQL engine
        engine = SQLEngine(
            rag_instance=mock_rag,
            table_agent=mock_table_agent,
            schema_validator=mock_schema_validator,
            clarification_agent=mock_clarification_agent,
            query_normalizer=mock_normalizer,
        )

        # Test without prior turns (old API)
        result = engine.generate_sql("New query")

        # Verify normalizer was called with None for prior_turns
        mock_normalizer.normalize.assert_called_once_with("New query", None)

        # Verify result structure
        assert result.status == SQLGenerationStatus.SUCCESS
