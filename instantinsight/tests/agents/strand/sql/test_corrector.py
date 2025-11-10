"""
Tests for Strands-based SQLCorrector - LLM-based SQL Error Correction.

This test suite covers the Strands-based SQLCorrector functionality using mocked Agent responses.
"""

from unittest.mock import Mock, patch

from src.agents.strand_agents.sql.corrector import (
    SQLCorrectionResponse,
    SQLCorrector,
    SQLFixer,
    SQLFixResult,
    fix_sql_error,
)


class TestStrandsSQLCorrector:
    """Test the Strands-based SQLCorrector class."""

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_init_with_agent(self, mock_agent_class):
        """Test initialization with Strands Agent."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        corrector = SQLCorrector()

        assert corrector.agent == mock_agent_instance
        mock_agent_class.assert_called_once()

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_empty_sql(self, mock_agent_class):
        """Test refine_sql with empty SQL."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        corrector = SQLCorrector()

        result = corrector.refine_sql("", "some error")

        assert not result.success
        assert result.error_message == "Empty SQL query provided"
        assert result.corrected_sql == ""

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_empty_error(self, mock_agent_class):
        """Test refine_sql with empty error message."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        corrector = SQLCorrector()

        result = corrector.refine_sql("SELECT * FROM table", "")

        assert not result.success
        assert result.error_message == "No error message provided"
        assert result.corrected_sql == "SELECT * FROM table"

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_success(self, mock_agent_class):
        """Test successful SQL refinement."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent response
        mock_agent_instance.structured_output.return_value = SQLCorrectionResponse(
            corrected_sql="SELECT * FROM users WHERE name = 'John'",
            confidence=0.95,
            changes_summary="Fixed table name from 'user' to 'users'",
        )

        corrector = SQLCorrector()

        result = corrector.refine_sql(
            "SELECT * FROM user WHERE name = 'John'", "Table 'user' doesn't exist"
        )

        assert result.success
        assert result.corrected_sql == "SELECT * FROM users WHERE name = 'John'"
        assert result.confidence == 0.95
        assert hasattr(result, "llm_response")

        # Verify agent was called with structured_output
        mock_agent_instance.structured_output.assert_called_once()

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_with_schema_context(self, mock_agent_class):
        """Test refine_sql with schema context."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent response
        mock_agent_instance.structured_output.return_value = SQLCorrectionResponse(
            corrected_sql="SELECT user_id, name FROM users",
            confidence=0.90,
            changes_summary="Fixed column reference using schema context",
        )

        corrector = SQLCorrector()
        schema_context = {
            "tables": ["users", "orders"],
            "columns": {"users": ["user_id", "name", "email"]},
        }

        result = corrector.refine_sql(
            "SELECT id, name FROM users", "Column 'id' not found", schema_context
        )

        assert result.success
        assert result.corrected_sql == "SELECT user_id, name FROM users"
        assert result.confidence == 0.90

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_no_changes(self, mock_agent_class):
        """Test when LLM returns same SQL (no changes made)."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent response with same SQL
        original_sql = "SELECT * FROM users"
        mock_agent_instance.structured_output.return_value = SQLCorrectionResponse(
            corrected_sql=original_sql,  # Same as input
            confidence=0.50,
            changes_summary="No changes needed",
        )

        corrector = SQLCorrector()

        result = corrector.refine_sql(original_sql, "Some error")

        assert not result.success
        assert result.error_message == "No changes made to SQL"
        assert result.corrected_sql == original_sql

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_refine_sql_agent_exception(self, mock_agent_class):
        """Test handling of agent exceptions."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.structured_output.side_effect = Exception(
            "Agent run failed"
        )

        corrector = SQLCorrector()

        result = corrector.refine_sql("SELECT * FROM table", "Some error")

        assert not result.success
        assert "Agent run failed" in result.error_message
        assert result.corrected_sql == "SELECT * FROM table"

    @patch("src.agents.strand_agents.sql.corrector.Agent")
    def test_build_fix_prompt_compatibility(self, mock_agent_class):
        """Test _build_fix_prompt compatibility method."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        corrector = SQLCorrector()

        # NOTE: _build_fix_prompt is an internal method from Agno implementation
        # In Strands, prompts are built directly in refine_sql method
        # This test verifies the core functionality works without the private method
        result = corrector.refine_sql("SELECT * FROM table", "Error")
        assert isinstance(result, SQLFixResult)


class TestStrandsSQLFixer:
    """Test compatibility wrapper SQLFixer."""

    @patch("src.agents.strand_agents.sql.corrector.SQLCorrector")
    def test_init_with_corrector(self, mock_corrector_class):
        """Test wrapper initialization."""
        mock_corrector = Mock()
        mock_corrector_class.return_value = mock_corrector

        fixer = SQLFixer()
        assert fixer.sql_corrector == mock_corrector
        mock_corrector_class.assert_called_once()

    @patch("src.agents.strand_agents.sql.corrector.SQLCorrector")
    def test_refine_sql_delegation(self, mock_corrector_class):
        """Test method delegation to core corrector."""
        mock_corrector = Mock()
        mock_corrector_class.return_value = mock_corrector

        mock_result = SQLFixResult(
            corrected_sql="SELECT * FROM users", success=True, confidence=0.9
        )
        mock_corrector.refine_sql.return_value = mock_result

        fixer = SQLFixer()
        result = fixer.refine_sql("SELECT * FROM user", "Table doesn't exist")

        assert result.success
        assert result.corrected_sql == "SELECT * FROM users"
        mock_corrector.refine_sql.assert_called_once_with(
            "SELECT * FROM user", "Table doesn't exist", None
        )


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.agents.strand_agents.sql.corrector.SQLFixer")
    def test_fix_sql_error_success(self, mock_fixer_class):
        """Test fix_sql_error convenience function success."""
        mock_fixer_instance = Mock()
        mock_fixer_class.return_value = mock_fixer_instance

        mock_result = SQLFixResult(
            corrected_sql="SELECT * FROM users", success=True, confidence=0.9
        )
        mock_fixer_instance.refine_sql.return_value = mock_result

        result = fix_sql_error("SELECT * FROM user", "Table doesn't exist")

        assert result == "SELECT * FROM users"
        mock_fixer_instance.refine_sql.assert_called_once_with(
            "SELECT * FROM user", "Table doesn't exist", None
        )

    @patch("src.agents.strand_agents.sql.corrector.SQLFixer")
    def test_fix_sql_error_failure(self, mock_fixer_class):
        """Test fix_sql_error convenience function failure."""
        mock_fixer_instance = Mock()
        mock_fixer_class.return_value = mock_fixer_instance

        mock_result = SQLFixResult(
            corrected_sql="SELECT * FROM user",
            success=False,
            error_message="Fix failed",
        )
        mock_fixer_instance.refine_sql.return_value = mock_result

        # Should return original SQL on failure
        result = fix_sql_error("SELECT * FROM user", "Table doesn't exist")

        assert result == "SELECT * FROM user"


class TestDataClasses:
    """Test data class functionality."""

    def test_sql_fix_result_success(self):
        """Test SQLFixResult for successful correction."""
        result = SQLFixResult(
            corrected_sql="SELECT * FROM users", success=True, confidence=0.95
        )

        assert result.corrected_sql == "SELECT * FROM users"
        assert result.success
        assert result.confidence == 0.95
        assert result.error_message == ""

    def test_sql_fix_result_failure(self):
        """Test SQLFixResult for failed correction."""
        result = SQLFixResult(
            corrected_sql="SELECT * FROM table",
            success=False,
            error_message="Correction failed",
        )

        assert result.corrected_sql == "SELECT * FROM table"
        assert not result.success
        assert result.error_message == "Correction failed"
        assert result.confidence == 0.0
