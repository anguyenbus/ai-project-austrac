"""Unit tests for BedrockTokenTracker."""

from unittest.mock import Mock

import pytest

from src.utils.bedrock_token_tracker import BedrockTokenTracker


class TestBedrockTokenTracker:
    """Test cases for BedrockTokenTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker instance for each test."""
        return BedrockTokenTracker()

    def test_track_instructor_usage_with_standard_format(self, tracker):
        """Test tracking token usage from instructor response with standard format."""
        # Create mock response with usage in standard format
        mock_response = Mock()
        mock_response._raw_response = {
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }

        usage = tracker.track_instructor_usage(
            mock_response, "claude-sonnet", metadata={"operation": "test"}
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model_id == "claude-sonnet"
        assert usage.metadata["operation"] == "test"
        assert len(tracker.usage_history) == 1

    def test_track_instructor_usage_with_camel_case_format(self, tracker):
        """Test tracking token usage with camelCase format."""
        # Create mock response with camelCase format
        mock_response = Mock()
        mock_response._raw_response = {
            "usage": {"inputTokens": 200, "outputTokens": 75}
        }

        usage = tracker.track_instructor_usage(mock_response, "claude-haiku")

        assert usage.input_tokens == 200
        assert usage.output_tokens == 75
        assert usage.total_tokens == 275
        assert usage.model_id == "claude-haiku"

    def test_track_instructor_usage_with_missing_tokens(self, tracker):
        """Test handling of response without token information."""
        # Create mock response without token information
        mock_response = Mock()
        mock_response._raw_response = {"result": "some_data"}

        usage = tracker.track_instructor_usage(mock_response, "claude-opus")

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model_id == "claude-opus"
        assert len(tracker.usage_history) == 1

    def test_get_usage_stats_with_multiple_calls(self, tracker):
        """Test getting aggregated usage statistics."""
        # Track multiple usages
        mock_response1 = Mock()
        mock_response1._raw_response = {
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        mock_response2 = Mock()
        mock_response2._raw_response = {
            "usage": {"input_tokens": 150, "output_tokens": 75}
        }

        tracker.track_instructor_usage(mock_response1, "claude-sonnet")
        tracker.track_instructor_usage(mock_response2, "claude-sonnet")
        tracker.track_instructor_usage(mock_response1, "claude-haiku")

        # Get stats for all models
        stats = tracker.get_usage_stats()
        assert stats["total_calls"] == 3
        assert stats["total_input_tokens"] == 350
        assert stats["total_output_tokens"] == 175
        assert stats["total_tokens"] == 525
        assert stats["average_tokens_per_call"] == 175

        # Get stats filtered by model
        sonnet_stats = tracker.get_usage_stats(model_id="claude-sonnet")
        assert sonnet_stats["total_calls"] == 2
        assert sonnet_stats["total_input_tokens"] == 250
        assert sonnet_stats["total_output_tokens"] == 125
        assert sonnet_stats["total_tokens"] == 375

    def test_reset_tracking(self, tracker):
        """Test resetting the tracker clears all history."""
        # Add some usage
        mock_response = Mock()
        mock_response._raw_response = {
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        tracker.track_instructor_usage(mock_response, "claude-sonnet")

        assert len(tracker.usage_history) == 1

        # Reset and verify
        tracker.reset_tracking()
        assert len(tracker.usage_history) == 0
        stats = tracker.get_usage_stats()
        assert stats["total_calls"] == 0
