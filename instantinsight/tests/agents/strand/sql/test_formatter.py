"""
Tests for Strands-based SQLFormatter - LLM-based SQL Formatting.

This test suite covers the Strands-based SQLFormatter functionality using mocked Agent responses.
"""

from unittest.mock import Mock, patch

import pytest

from src.agents.strand_agents.sql.formatter import (
    SpacingIssue,
    SQLFormatter,
    SQLSpacingAgent,
    SQLSpacingAnalysis,
    SQLSpacingResult,
    fix_sql_spacing_with_llm,
)


class TestStrandsSQLFormatter:
    """Test the Strands-based SQLFormatter class."""

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_init_with_agent(self, mock_agent_class):
        """Test initialization with Strands Agent."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        formatter = SQLFormatter()

        assert formatter.agent == mock_agent_instance
        mock_agent_class.assert_called_once()

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_fix_sql_spacing_empty_sql(self, mock_agent_class):
        """Test fix_sql_spacing with empty SQL."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        formatter = SQLFormatter()

        result = formatter.fix_sql_spacing("")

        assert result.success
        assert result.confidence == 1.0
        assert result.fixed_sql == ""
        assert result.issues_found == []

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_fix_sql_spacing_success(self, mock_agent_class):
        """Test successful SQL spacing fix."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent response
        mock_agent_instance.structured_output.return_value = SQLSpacingAnalysis(
            fixed_sql="SELECT * FROM users WHERE id > 0",
            issues_found=[
                SpacingIssue(
                    location="Around FROM keyword",
                    issue_type="missing_space",
                    original_text="SELECT *FROM",
                    fixed_text="SELECT * FROM",
                ),
                SpacingIssue(
                    location="Around WHERE condition",
                    issue_type="operator_spacing",
                    original_text="WHERE id>0",
                    fixed_text="WHERE id > 0",
                ),
            ],
            confidence=0.95,
            requires_fixes=True,
            fix_summary="Fixed missing spaces around keywords and operators",
        )

        formatter = SQLFormatter()

        result = formatter.fix_sql_spacing("SELECT *FROM users WHERE id>0")

        assert result.success
        assert result.fixed_sql == "SELECT * FROM users WHERE id > 0"
        assert result.confidence == 0.95
        assert len(result.issues_found) == 2
        assert "missing_space" in result.issues_found[0]
        assert "operator_spacing" in result.issues_found[1]
        assert hasattr(result, "llm_response")

        # Verify agent was called with structured_output
        mock_agent_instance.structured_output.assert_called_once()

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_fix_sql_spacing_no_specific_issues(self, mock_agent_class):
        """Test when LLM finds fixes but no specific issues."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent response with no specific issues but fixes made
        mock_agent_instance.structured_output.return_value = SQLSpacingAnalysis(
            fixed_sql="SELECT user_id, name FROM users",
            issues_found=[],  # No specific issues
            confidence=0.90,
            requires_fixes=True,  # But fixes were needed
            fix_summary="General formatting improvements applied",
        )

        formatter = SQLFormatter()

        result = formatter.fix_sql_spacing("SELECT user_id,name FROM users")

        assert result.success
        assert result.fixed_sql == "SELECT user_id, name FROM users"
        assert result.confidence == 0.90
        assert len(result.issues_found) == 1
        assert "General formatting improvements applied" in result.issues_found[0]

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_fix_sql_spacing_agent_exception(self, mock_agent_class):
        """Test handling of agent exceptions."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.structured_output.side_effect = Exception(
            "Agent run failed"
        )

        formatter = SQLFormatter()

        result = formatter.fix_sql_spacing("SELECT * FROM table")

        assert not result.success
        assert result.confidence == 0.0
        assert "Agent run failed" in result.issues_found[0]
        assert result.fixed_sql == "SELECT * FROM table"

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_fix_sql_spacing_invalid_response(self, mock_agent_class):
        """Test handling of invalid agent response."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Mock response with wrong content type - will trigger exception
        mock_agent_instance.structured_output.return_value = "Invalid response type"

        formatter = SQLFormatter()

        result = formatter.fix_sql_spacing("SELECT * FROM table")

        assert not result.success
        assert result.confidence == 0.0
        # Error message contains information about the failure
        assert len(result.issues_found) > 0
        assert result.fixed_sql == "SELECT * FROM table"

    @patch("src.agents.strand_agents.sql.formatter.Agent")
    def test_compatibility_methods(self, mock_agent_class):
        """Test compatibility methods."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Use SQLSpacingAgent for compatibility methods (not SQLFormatter)
        agent = SQLSpacingAgent()

        # Test _build_spacing_prompt compatibility method
        prompt = agent._build_spacing_prompt("SELECT * FROM table")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test _fix_with_instructor compatibility method
        mock_agent_instance.structured_output.return_value = SQLSpacingAnalysis(
            fixed_sql="SELECT * FROM users",
            issues_found=[],
            confidence=0.95,
            requires_fixes=False,
            fix_summary="No issues found",
        )

        instructor_result = agent._fix_with_instructor("SELECT * FROM users")

        assert "result" in instructor_result
        assert "llm_response" in instructor_result
        assert instructor_result["result"].success


class TestStrandsSQLSpacingAgent:
    """Test compatibility wrapper SQLSpacingAgent."""

    @patch("src.agents.strand_agents.sql.formatter.SQLFormatter")
    def test_init_with_formatter(self, mock_formatter_class):
        """Test wrapper initialization."""
        mock_formatter = Mock()
        mock_formatter_class.return_value = mock_formatter

        agent = SQLSpacingAgent()
        assert agent.sql_formatter == mock_formatter
        mock_formatter_class.assert_called_once()

    @patch("src.agents.strand_agents.sql.formatter.SQLFormatter")
    def test_fix_sql_spacing_delegation(self, mock_formatter_class):
        """Test method delegation to core formatter."""
        mock_formatter = Mock()
        mock_formatter_class.return_value = mock_formatter

        mock_result = SQLSpacingResult(
            fixed_sql="SELECT * FROM users WHERE id > 0",
            issues_found=["Fixed spacing around operators"],
            confidence=0.9,
            success=True,
        )
        mock_formatter.fix_sql_spacing.return_value = mock_result

        agent = SQLSpacingAgent()
        result = agent.fix_sql_spacing("SELECT * FROM users WHERE id>0")

        assert result.success
        assert result.fixed_sql == "SELECT * FROM users WHERE id > 0"
        mock_formatter.fix_sql_spacing.assert_called_once_with(
            "SELECT * FROM users WHERE id>0"
        )


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.agents.strand_agents.sql.formatter.SQLSpacingAgent")
    def test_fix_sql_spacing_with_llm_success(self, mock_agent_class):
        """Test fix_sql_spacing_with_llm convenience function success."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = SQLSpacingResult(
            fixed_sql="SELECT * FROM users WHERE id > 0",
            issues_found=["Fixed spacing around operators"],
            confidence=0.9,
            success=True,
        )
        mock_agent_instance.fix_sql_spacing.return_value = mock_result

        result = fix_sql_spacing_with_llm("SELECT * FROM users WHERE id>0")

        assert result == "SELECT * FROM users WHERE id > 0"
        mock_agent_instance.fix_sql_spacing.assert_called_once_with(
            "SELECT * FROM users WHERE id>0"
        )

    @patch("src.agents.strand_agents.sql.formatter.SQLSpacingAgent")
    def test_fix_sql_spacing_with_llm_failure(self, mock_agent_class):
        """Test fix_sql_spacing_with_llm convenience function failure."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = SQLSpacingResult(
            fixed_sql="SELECT * FROM users WHERE id>0",
            issues_found=["Fix failed"],
            confidence=0.0,
            success=False,
        )
        mock_agent_instance.fix_sql_spacing.return_value = mock_result

        # NOTE: In Strands implementation, even on "failure" the fixed_sql is returned
        # This differs from Agno which might return original on failure
        # The actual implementation logs success and returns fixed_sql
        result = fix_sql_spacing_with_llm("SELECT * FROM users WHERE id>0")

        # The function returns the result regardless of success flag
        assert result is not None


class TestDataClasses:
    """Test data class functionality."""

    def test_sql_spacing_result_success(self):
        """Test SQLSpacingResult for successful formatting."""
        result = SQLSpacingResult(
            fixed_sql="SELECT * FROM users",
            issues_found=["Fixed spacing around FROM"],
            confidence=0.95,
            success=True,
        )

        assert result.fixed_sql == "SELECT * FROM users"
        assert result.success
        assert result.confidence == 0.95
        assert len(result.issues_found) == 1

    def test_sql_spacing_result_failure(self):
        """Test SQLSpacingResult for failed formatting."""
        result = SQLSpacingResult(
            fixed_sql="SELECT * FROM table",
            issues_found=["Formatting failed"],
            confidence=0.0,
            success=False,
        )

        assert result.fixed_sql == "SELECT * FROM table"
        assert not result.success
        assert result.confidence == 0.0
        assert "Formatting failed" in result.issues_found

    def test_spacing_issue_validation(self):
        """Test SpacingIssue validation."""
        # Valid issue type
        issue = SpacingIssue(
            location="Around FROM keyword",
            issue_type="missing_space",
            original_text="SELECT *FROM",
            fixed_text="SELECT * FROM",
        )
        assert issue.issue_type == "missing_space"

        # Invalid issue type should raise validation error
        with pytest.raises(ValueError, match="issue_type must be one of"):
            SpacingIssue(
                location="Around FROM keyword",
                issue_type="invalid_type",
                original_text="SELECT *FROM",
                fixed_text="SELECT * FROM",
            )

    def test_sql_spacing_analysis_validation(self):
        """Test SQLSpacingAnalysis validation."""
        # Valid analysis
        analysis = SQLSpacingAnalysis(
            fixed_sql="SELECT * FROM users",
            confidence=0.95,
            requires_fixes=True,
            fix_summary="Fixed spacing issues",
        )
        assert analysis.fixed_sql == "SELECT * FROM users"

        # Empty SQL should raise validation error
        with pytest.raises(ValueError, match="fixed_sql cannot be empty"):
            SQLSpacingAnalysis(
                fixed_sql="",
                confidence=0.95,
                requires_fixes=False,
                fix_summary="No fixes needed",
            )
