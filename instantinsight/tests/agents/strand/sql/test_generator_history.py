"""Tests for SQL generator with conversation history."""

from unittest.mock import patch

from src.agents.strand_agents.sql.generator import SQLGenerator, SQLWriterAgent


class TestSQLGeneratorHistory:
    """Test SQL generator history prompt functionality."""

    def test_build_history_prompt_empty(self):
        """Test history prompt with empty prior turns."""
        generator = SQLGenerator()

        # Empty list
        prompt1 = generator._build_history_prompt([])
        assert prompt1 == ""

        # None
        prompt2 = generator._build_history_prompt(None)
        assert prompt2 == ""

    def test_build_history_prompt_with_turns(self):
        """Test history prompt with actual turns."""
        generator = SQLGenerator()

        prior_turns = [
            {
                "content": "Show top 10 products by revenue",
                "sql": "SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue DESC LIMIT 10",
                "visualization": {"type": "bar"},
            },
            {
                "content": "Make it a pie chart",
                "sql": "SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue DESC LIMIT 10",
                "visualization": {"type": "pie"},
            },
        ]

        prompt = generator._build_history_prompt(prior_turns)

        # Should include conversation structure
        assert "CONVERSATION HISTORY:" in prompt
        assert "Turn 1:" in prompt
        assert "Turn 2:" in prompt
        assert "User: Show top 10 products by revenue" in prompt
        assert "User: Make it a pie chart" in prompt

        # Should include abbreviated SQL
        assert (
            "SQL: SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue DESC LIMIT 10"
            in prompt
        )

        # Should include visualization types
        assert "Visualization: bar chart" in prompt
        assert "Visualization: pie chart" in prompt

        # Should include instructions
        assert "CURRENT REQUEST:" in prompt
        assert "continuing the conversation" in prompt
        assert "beginning visual" in prompt
        assert "Turn 1" in prompt

    def test_build_history_prompt_turn_indexing(self):
        """Test history prompt shows correct turn numbers."""
        generator = SQLGenerator()

        prior_turns = [
            {"content": "First query", "sql": "SELECT 1"},
            {
                "content": "Second query",
                "sql": "SELECT 2",
                "visualization": {"type": "bar"},
            },
            {"content": "Third query", "sql": "SELECT 3"},
        ]

        prompt = generator._build_history_prompt(prior_turns)

        # Check turn indexing
        assert "Turn 1:" in prompt
        assert "Turn 2:" in prompt
        assert "Turn 3:" in prompt

        # Check order is preserved
        turn1_pos = prompt.find("Turn 1:")
        turn2_pos = prompt.find("Turn 2:")
        turn3_pos = prompt.find("Turn 3:")

        assert turn1_pos < turn2_pos < turn3_pos

    def test_build_history_prompt_sql_truncation(self):
        """Test SQL is truncated to first 100 characters."""
        generator = SQLGenerator()

        long_sql = "SELECT * FROM very_long_table_name WHERE some_very_long_condition = 'some_very_long_value' AND another_condition = 'another_value' AND yet_more_conditions"

        prior_turns = [{"content": "Test query", "sql": long_sql}]

        prompt = generator._build_history_prompt(prior_turns)

        # Should include truncated SQL with ellipsis
        assert long_sql[:100] in prompt
        assert "..." in prompt
        # Should not include the full long SQL (check that end of long SQL is not in prompt)
        assert "yet_more_conditions" not in prompt

    def test_build_history_prompt_visualization_context(self):
        """Test visualization context is included."""
        generator = SQLGenerator()

        prior_turns = [
            {
                "content": "Show sales data",
                "sql": "SELECT * FROM sales",
                "visualization": {
                    "type": "line",
                    "config": {"x": "date", "y": "amount"},
                },
            }
        ]

        prompt = generator._build_history_prompt(prior_turns)

        # Should include visualization type
        assert "Visualization: line chart" in prompt

    def test_sql_writer_agent_passes_prior_turns(self):
        """Test SQLWriterAgent passes prior_turns to generator."""
        writer = SQLWriterAgent()

        # Mock the generator to track calls
        with patch.object(writer.generator, "generate_sql") as mock_generate:
            mock_generate.return_value = {"sql": "SELECT * FROM test"}

            prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM other"}]

            writer.generate_sql(
                question="New query",
                schema_context=[],
                example_context=[],
                selected_tables=["test"],
                search_results=[],
                best_similarity=0.8,
                prior_turns=prior_turns,
            )

            # Verify generator was called with prior_turns
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args.kwargs["prior_turns"] == prior_turns

    def test_sql_generator_handles_missing_fields(self):
        """Test generator handles turns with missing fields gracefully."""
        generator = SQLGenerator()

        prior_turns = [
            {"content": "Query with only content"},  # Missing SQL and visualization
            {"sql": "SELECT * FROM test"},  # Missing content and visualization
            {
                "content": "Full query",
                "sql": "SELECT * FROM test",
                "visualization": {"type": "bar"},
            },
        ]

        prompt = generator._build_history_prompt(prior_turns)

        # Should handle missing fields gracefully
        assert "Turn 1:" in prompt
        assert "User: Query with only content" in prompt
        assert (
            "SQL:" not in prompt.split("Turn 1:")[1].split("Turn 2:")[0]
        )  # No SQL in turn 1

        assert "Turn 2:" in prompt
        # Turn 2 should have SQL but no User content (empty user content)
        turn2_section = prompt.split("Turn 2:")[1].split("Turn 3:")[0]
        assert "SQL: SELECT * FROM test" in turn2_section
        # Should have empty User field for turn 2
        assert "User: \n  SQL:" in turn2_section
