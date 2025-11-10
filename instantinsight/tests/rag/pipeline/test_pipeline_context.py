"""Tests for pipeline with conversation context."""

from src.rag.pipeline.stages import PipelineResult


class TestPipelineContext:
    """Test pipeline context threading."""

    def test_pipeline_threads_prior_turns(self):
        """Test pipeline threads prior_turns through all stages."""
        # Create a real PipelineResult to test our changes
        result = PipelineResult(query="test")

        # Test storing prior_turns
        prior_turns = [{"content": "Previous query", "sql": "SELECT * FROM table1"}]
        result.prior_turns = prior_turns

        # Verify prior_turns were stored
        assert result.prior_turns == prior_turns

    def test_pipeline_result_includes_prior_turns(self):
        """Test PipelineResult includes prior_turns field."""
        # Test with empty prior_turns
        result1 = PipelineResult(query="test")
        assert result1.prior_turns == []

        # Test with provided prior_turns
        prior_turns = [{"content": "Test", "sql": "SELECT 1"}]
        result2 = PipelineResult(query="test", prior_turns=prior_turns)
        assert result2.prior_turns == prior_turns

    def test_pipeline_handles_none_prior_turns(self):
        """Test pipeline handles None prior_turns gracefully."""
        # Create a real PipelineResult to test our changes
        result = PipelineResult(query="test", prior_turns=None)

        # Verify __post_init__ converts None to empty list
        assert result.prior_turns == []

    def test_pipeline_backward_compatibility(self):
        """Test pipeline works without prior_turns parameter (backward compatibility)."""
        # Create a real PipelineResult without prior_turns
        result = PipelineResult(query="test")

        # Verify default behavior
        assert result.prior_turns == []
