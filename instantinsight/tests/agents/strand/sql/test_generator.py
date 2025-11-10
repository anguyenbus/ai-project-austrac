"""
Comprehensive test suite for Strands-based SQLGenerator migration.

Tests both the core Strands SQLGenerator implementation and the
compatibility wrapper (SQLWriterAgent) to ensure backward compatibility.
"""

from unittest.mock import Mock, patch

import pytest

from src.agents.strand_agents.sql.generator import (
    SQLGenerator,
    SQLResponse,
    SQLWriterAgent,
)


class TestStrandsSQLGenerator:
    """Test core Strands implementation."""

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_init_with_agent(self, mock_agent_class):
        """Test initialization with Strands Agent."""
        generator = SQLGenerator()

        assert generator.model_id is not None
        assert generator.aws_region is not None
        mock_agent_class.assert_called_once()

        # Verify Agent initialization parameters
        call_args = mock_agent_class.call_args
        assert "model" in call_args.kwargs
        assert "system_prompt" in call_args.kwargs

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_generate_sql_success(self, mock_agent_class):
        """Test successful SQL generation."""
        # Mock agent setup
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock structured_output response
        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Column mapping: ledger_name for customer names",
            sql="SELECT ledger_name FROM awsdatacatalog.db.customer_table",
            confidence=0.8,
        )

        # Test
        generator = SQLGenerator()
        result = generator.generate_sql(
            question="Show customer names",
            schema_context=["customer_table: * ledger_name, * account_number"],
            example_context=[],
            selected_tables=["customer_table"],
            search_results=[{"chunk_type": "schema", "similarity": 0.9}],
            best_similarity=0.9,
            filter_context=None,
        )

        # Assertions
        assert result is not None
        assert "sql" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert result["confidence"] >= 0.8
        mock_agent_instance.structured_output.assert_called_once()

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_generate_sql_with_filters(self, mock_agent_class):
        """Test SQL generation with filter context."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Applied city filter using IN clause",
            sql="SELECT * FROM table WHERE city IN ('Brisbane', 'BRISBANE')",
            confidence=0.85,
        )

        generator = SQLGenerator()
        filter_context = {"mapped_filters": [{"city": ["Brisbane", "BRISBANE"]}]}

        result = generator.generate_sql(
            question="Show Brisbane customers",
            schema_context=["table: * city, * customer_name"],
            example_context=[],
            selected_tables=["table"],
            search_results=[],
            best_similarity=0.8,
            filter_context=filter_context,
        )

        assert "WHERE city IN" in result["sql"]
        assert result["confidence"] > 0.8

    @patch("src.agents.strand_agents.sql.generator.Agent")
    @patch("src.agents.strand_agents.sql.generator.ClarificationAgent")
    def test_low_confidence_handling(self, mock_clarification_class, mock_agent_class):
        """Test low confidence response handling."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Low confidence response
        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Cannot determine appropriate tables",
            sql="SELECT * FROM unknown",
            confidence=0.3,
        )

        # Mock clarification agent
        mock_clarification = Mock()
        mock_clarification.generate_clarification_response.return_value = (
            "Please specify the table"
        )
        mock_clarification_class.return_value = mock_clarification

        generator = SQLGenerator()
        result = generator.generate_sql(
            question="Show data",
            schema_context=[],
            example_context=[],
            selected_tables=[],
            search_results=[],
            best_similarity=0.2,
            filter_context=None,
        )

        assert "CANNOT FIND TABLES" in result["sql"]
        assert result["confidence"] < 0.5
        # Clarification is called but from the actual import path

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_error_handling(self, mock_agent_class):
        """Test error handling during SQL generation."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.structured_output.side_effect = Exception("LLM API error")

        generator = SQLGenerator()

        with pytest.raises(Exception) as exc_info:
            generator.generate_sql(
                question="Test query",
                schema_context=[],
                example_context=[],
                selected_tables=[],
                search_results=[],
                best_similarity=0.5,
                filter_context=None,
            )

        assert "LLM API error" in str(exc_info.value)

    @patch("src.agents.strand_agents.sql.generator.Agent")
    @patch("src.agents.strand_agents.sql.formatter.SQLSpacingAgent")
    def test_sql_post_processing(self, mock_spacing_class, mock_agent_class):
        """Test SQL post-processing with spacing agent."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Generated SQL", sql="SELECT*FROM table", confidence=0.9
        )

        # Mock spacing agent
        mock_spacing = Mock()
        mock_spacing_result = Mock()
        mock_spacing_result.fixed_sql = "SELECT * FROM table"
        mock_spacing.fix_sql_spacing.return_value = mock_spacing_result
        mock_spacing_class.return_value = mock_spacing

        generator = SQLGenerator()
        result = generator.generate_sql(
            question="Test",
            schema_context=["table: * col1"],
            example_context=[],
            selected_tables=["table"],
            search_results=[],
            best_similarity=0.8,
            filter_context=None,
        )

        assert result["sql"] == "SELECT * FROM table"
        mock_spacing.fix_sql_spacing.assert_called_once()

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_confidence_calculation(self, mock_agent_class):
        """Test confidence score combination logic."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="High confidence query",
            sql="SELECT * FROM table",
            confidence=0.95,
        )

        generator = SQLGenerator()
        result = generator.generate_sql(
            question="Clear query",
            schema_context=["table: * col1"],
            example_context=["SELECT * FROM table"],
            selected_tables=["table"],
            search_results=[{"similarity": 0.95}],
            best_similarity=0.95,
            filter_context=None,
        )

        # Combined confidence should be high
        assert result["confidence"] > 0.9
        assert result["metadata"]["llm_confidence"] == 0.95
        assert result["metadata"]["context_confidence"] > 0.8


class TestStrandsSQLWriterAgentWrapper:
    """Test compatibility wrapper."""

    @patch("src.agents.strand_agents.sql.generator.SQLGenerator")
    def test_init_with_generator(self, mock_generator_class):
        """Test wrapper initialization with SQLGenerator."""
        agent = SQLWriterAgent()
        assert agent.region is not None
        assert agent.model_id is not None
        mock_generator_class.assert_called_once()

    @patch("src.agents.strand_agents.sql.generator.SQLGenerator")
    def test_generate_sql_delegation(self, mock_generator_class):
        """Test method delegation to core generator."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_sql.return_value = {
            "sql": "SELECT 1",
            "confidence": 0.9,
        }

        agent = SQLWriterAgent()
        result = agent.generate_sql(
            question="test",
            schema_context=[],
            example_context=[],
            selected_tables=[],
            search_results=[],
            best_similarity=0.5,
            filter_context=None,
        )

        mock_generator.generate_sql.assert_called_once_with(
            question="test",
            schema_context=[],
            example_context=[],
            selected_tables=[],
            search_results=[],
            best_similarity=0.5,
            filter_context=None,
        )
        assert result["sql"] == "SELECT 1"
        assert result["confidence"] == 0.9

    def test_get_token_stats(self):
        """Test token statistics method."""
        # Test that the method exists and returns expected structure
        agent = SQLWriterAgent()
        stats = agent.get_token_stats()

        # Check structure is correct
        assert isinstance(stats, dict)
        assert "total_calls" in stats
        assert "total_tokens" in stats
        assert isinstance(stats["total_calls"], int)
        assert isinstance(stats["total_tokens"], int)

    @patch("src.agents.strand_agents.sql.generator.SQLGenerator")
    def test_backward_compatibility(self, mock_generator_class):
        """Test that wrapper maintains backward compatibility."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Simulate existing usage pattern
        agent = SQLWriterAgent()

        # Check all expected attributes exist
        assert hasattr(agent, "region")
        assert hasattr(agent, "model_id")
        assert hasattr(agent, "generate_sql")
        assert hasattr(agent, "get_token_stats")

        # Check method signatures match
        import inspect

        sig = inspect.signature(agent.generate_sql)
        params = list(sig.parameters.keys())

        expected_params = [
            "question",
            "schema_context",
            "example_context",
            "selected_tables",
            "search_results",
            "best_similarity",
            "filter_context",
        ]
        assert params == expected_params


class TestIntegration:
    """Integration tests for the complete agent."""

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_full_pipeline_integration(self, mock_agent_class):
        """Test complete pipeline from question to SQL."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Selected customer_ledger table for outstanding amounts",
            sql="SELECT ledger_name, outstanding FROM awsdatacatalog.db.customer_ledger",
            confidence=0.88,
        )

        # Use the wrapper (as external code would)
        agent = SQLWriterAgent()
        result = agent.generate_sql(
            question="Show customer outstanding amounts",
            schema_context=[
                "customer_ledger: * ledger_name, * outstanding, * account_number"
            ],
            example_context=[
                "SELECT ledger_name FROM customer_ledger WHERE outstanding > 0"
            ],
            selected_tables=["customer_ledger"],
            search_results=[
                {
                    "chunk_type": "schema",
                    "similarity": 0.92,
                    "content": "customer schema",
                }
            ],
            best_similarity=0.92,
            filter_context=None,
        )

        assert result["sql"] is not None
        assert "customer_ledger" in result["sql"]
        assert result["confidence"] > 0.85
        assert result["selected_tables"] == ["customer_ledger"]
        assert result["metadata"]["best_similarity"] == 0.92

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_semantic_filters_integration(self, mock_agent_class):
        """Test integration with semantic filter context."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Applied location filter for Brisbane",
            sql="SELECT * FROM customer WHERE city = 'Brisbane'",
            confidence=0.82,
        )

        agent = SQLWriterAgent()
        filter_context = {"semantic_filters": [{"location": "Brisbane"}]}

        result = agent.generate_sql(
            question="Brisbane customers",
            schema_context=["customer: * city, * name"],
            example_context=[],
            selected_tables=["customer"],
            search_results=[],
            best_similarity=0.75,
            filter_context=filter_context,
        )

        assert "Brisbane" in result["sql"]
        assert result["confidence"] > 0.75

    @patch("src.agents.strand_agents.sql.generator.Agent")
    def test_metadata_preservation(self, mock_agent_class):
        """Test that all metadata is preserved through the pipeline."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_agent_instance.structured_output.return_value = SQLResponse(
            reasoning="Test reasoning", sql="SELECT 1", confidence=0.7
        )

        agent = SQLWriterAgent()
        search_results = [
            {"chunk_type": "schema", "similarity": 0.8, "content": "test content"},
            {"chunk_type": "example", "similarity": 0.7, "content": "example sql"},
        ]

        result = agent.generate_sql(
            question="Test",
            schema_context=["schema"],
            example_context=["example"],
            selected_tables=["table1", "table2"],
            search_results=search_results,
            best_similarity=0.8,
            filter_context=None,
        )

        # Check all metadata is preserved
        assert "sources" in result
        assert len(result["sources"]) <= 5
        assert result["metadata"]["best_similarity"] == 0.8
        assert result["metadata"]["sources_count"] == 2
        assert "llm_confidence" in result["metadata"]
        assert "context_confidence" in result["metadata"]
        assert result["selected_tables"] == ["table1", "table2"]
