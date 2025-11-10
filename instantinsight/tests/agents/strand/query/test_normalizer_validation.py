"""Test Pydantic validation fix for NormalizedQuery model."""

import pytest
from pydantic import ValidationError

from src.agents.strand_agents.query.normalizer import NormalizedQuery


class TestNormalizedQueryValidation:
    """Test NormalizedQuery model validation behavior."""

    def test_details_for_filterings_accepts_none(self) -> None:
        """Test that details_for_filterings field accepts None and converts to empty list."""
        # This simulates what the LLM might return
        data = {
            "main_clause": "product count by category",
            "details_for_filterings": None,  # LLM returned null
            "required_visuals": None,
            "tables": None,
        }

        # Should not raise ValidationError
        result = NormalizedQuery.model_validate(data)

        assert result.main_clause == "product count by category"
        assert result.details_for_filterings == []  # Converted from None
        assert result.required_visuals is None
        assert result.tables is None

    def test_details_for_filterings_accepts_list(self) -> None:
        """Test that details_for_filterings field accepts a list."""
        data = {
            "main_clause": "product count by category",
            "details_for_filterings": ["for some period", "with filters"],
            "required_visuals": "bar chart",
            "tables": ["products"],
        }

        result = NormalizedQuery.model_validate(data)

        assert result.main_clause == "product count by category"
        assert result.details_for_filterings == ["for some period", "with filters"]
        assert result.required_visuals == "bar chart"
        assert result.tables == ["products"]

    def test_details_for_filterings_uses_default_factory(self) -> None:
        """Test that details_for_filterings uses default_factory when field is missing."""
        data = {
            "main_clause": "product count by category",
            # details_for_filterings field is completely missing
        }

        result = NormalizedQuery.model_validate(data)

        assert result.main_clause == "product count by category"
        assert result.details_for_filterings == []  # Uses default_factory
        assert result.required_visuals is None
        assert result.tables is None

    def test_main_clause_required(self) -> None:
        """Test that main_clause is required and cannot be None."""
        data = {
            "main_clause": None,
            "details_for_filterings": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            NormalizedQuery.model_validate(data)

        assert "main_clause" in str(exc_info.value)

    def test_direct_construction_with_none(self) -> None:
        """Test direct construction with None for details_for_filterings."""
        # This should also work via the validator
        result = NormalizedQuery(
            main_clause="test query",
            details_for_filterings=None,  # type: ignore
        )

        assert result.details_for_filterings == []
