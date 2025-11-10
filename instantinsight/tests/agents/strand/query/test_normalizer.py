"""Tests for the Strand-based query normalization agent."""

from unittest.mock import Mock

from src.agents.strand_agents.query.normalizer import (
    NormalizedQuery,
    QueryNormalizationCore,
)


class TestQueryNormalizerStrand:
    """Test Strand-based QueryNormalizer agent wiring."""

    def test_normalize_success_path(self) -> None:
        """Test successful query normalization with Strand agent."""
        mock_agent = Mock()
        mock_agent.structured_output.return_value = NormalizedQuery(
            main_clause="Sales performance",
            details_for_filterings=["for some period"],
            required_visuals="line chart",
            tables=["generalized dataset"],
        )
        normalizer = QueryNormalizationCore(agent=mock_agent)

        result = normalizer.normalize(
            "Show detailed sales performance for 2023 as a line chart"
        )

        assert result.main_clause == "Sales performance"
        assert result.details_for_filterings == ["for some period"]
        assert result.required_visuals == "line chart"
        assert result.tables == ["generalized dataset"]
        mock_agent.structured_output.assert_called_once()

    def test_normalize_fallback(self) -> None:
        """Test query normalization fallback behavior with Strand agent."""
        mock_agent = Mock()
        mock_agent.structured_output.side_effect = RuntimeError("Provider outage")
        normalizer = QueryNormalizationCore(agent=mock_agent)

        result = normalizer.normalize("Display churn metrics")

        assert result.main_clause == "General business request"
        assert result.details_for_filterings == []
        assert result.required_visuals is None
        assert result.tables is None
