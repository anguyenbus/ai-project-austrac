"""Tests for query normalizer with conversation context."""

from src.agents.strand_agents.query.normalizer import QueryNormalizer


class TestNormalizerContext:
    """Test normalizer context awareness."""

    def test_normalizer_with_prior_turns(self):
        """Test normalizer uses prior turn context."""
        normalizer = QueryNormalizer()

        # Turn 1 - establishes context
        result1 = normalizer.normalize("Show top 10 products by revenue", [])
        assert (
            "products" in result1.main_clause.lower()
            or "revenue" in result1.main_clause.lower()
        )

        # Turn 2 - references previous context
        prior_turns = [
            {
                "content": "Show top 10 products by revenue",
                "sql": "SELECT product, revenue FROM products ORDER BY revenue DESC LIMIT 10",
            }
        ]
        result2 = normalizer.normalize("Now show me top 20", prior_turns)

        # Should preserve context from Turn 1 (products/revenue mentioned)
        assert (
            "products" in result2.main_clause.lower()
            or "revenue" in result2.main_clause.lower()
        )
        # Should indicate increased limit in details_for_filterings
        assert any(
            "limit" in detail.lower() or "increased" in detail.lower()
            for detail in result2.details_for_filterings
        )

    def test_normalizer_without_context_fails_gracefully(self):
        """Test normalizer handles ambiguous queries without context."""
        normalizer = QueryNormalizer()

        # Ambiguous query without context
        result = normalizer.normalize("Show me top 20", [])

        # Should normalize to just the basic query without context
        assert "top 20" in result.main_clause.lower()
        # Should not have contextual filters without prior context
        assert len(result.details_for_filterings) == 0

    def test_normalizer_context_prompt_building(self):
        """Test context prompt building."""
        from src.agents.prompt_builders.query.normalizer import (
            QueryNormalizationPrompts,
        )

        prior_turns = [
            {"content": "Show sales data", "sql": "SELECT * FROM sales"},
            {"content": "Make it a bar chart", "sql": "SELECT * FROM sales"},
        ]

        prompt = QueryNormalizationPrompts.build_prompt("Show me more", prior_turns)

        # Should include context section
        assert "CONVERSATION CONTEXT:" in prompt
        assert "Turn 1: Show sales data" in prompt
        assert "Turn 2: Make it a bar chart" in prompt
        assert "CURRENT REQUEST:" in prompt
        assert "continuing the conversation" in prompt

    def test_normalizer_empty_prior_turns(self):
        """Test normalizer with empty prior turns."""
        normalizer = QueryNormalizer()

        # Empty list
        result1 = normalizer.normalize("Show sales data", [])

        # None
        result2 = normalizer.normalize("Show sales data", None)

        # Should produce same result
        assert result1.main_clause == result2.main_clause

    def test_normalizer_context_with_visualization_reference(self):
        """Test normalizer handles visualization references in context."""
        normalizer = QueryNormalizer()

        prior_turns = [
            {
                "content": "Show revenue by region",
                "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
                "visualization": {"type": "bar"},
            }
        ]

        result = normalizer.normalize("Change to pie chart", prior_turns)

        # Should understand this is about the same data with different visualization
        assert (
            "revenue" in result.main_clause.lower()
            or "region" in result.main_clause.lower()
        )
        assert result.required_visuals == "pie chart"
