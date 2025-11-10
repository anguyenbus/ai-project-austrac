"""
Smoke tests for Strands-based SQL agents.

These tests verify basic functionality and imports without mocking.
Full test suite will be created in a separate testing phase.
"""

import pytest


class TestSQLAgentsImports:
    """Test that all SQL agents can be imported successfully."""

    def test_import_sql_generator(self):
        """Test SQLGenerator imports correctly."""
        from src.agents.strand_agents.sql.generator import (
            SQLGenerator,
            SQLResponse,
            SQLWriterAgent,
        )

        assert SQLGenerator is not None
        assert SQLResponse is not None
        assert SQLWriterAgent is not None

    def test_import_sql_corrector(self):
        """Test SQLCorrector imports correctly."""
        from src.agents.strand_agents.sql.corrector import (
            SQLCorrectionResponse,
            SQLCorrector,
            SQLFixer,
            SQLFixResult,
        )

        assert SQLCorrector is not None
        assert SQLCorrectionResponse is not None
        assert SQLFixer is not None
        assert SQLFixResult is not None

    def test_import_sql_formatter(self):
        """Test SQLFormatter imports correctly."""
        from src.agents.strand_agents.sql.formatter import (
            SpacingIssue,
            SQLFormatter,
            SQLSpacingAgent,
            SQLSpacingAnalysis,
            SQLSpacingResult,
        )

        assert SQLFormatter is not None
        assert SQLSpacingAgent is not None
        assert SQLSpacingAnalysis is not None
        assert SQLSpacingResult is not None
        assert SpacingIssue is not None

    def test_import_sql_package(self):
        """Test SQL package imports correctly."""
        from src.agents.strand_agents import sql

        # Check main classes are accessible
        assert hasattr(sql, "SQLGenerator")
        assert hasattr(sql, "SQLWriterAgent")
        assert hasattr(sql, "SQLCorrector")
        assert hasattr(sql, "SQLFixer")
        assert hasattr(sql, "SQLFormatter")
        assert hasattr(sql, "SQLSpacingAgent")


class TestSQLResponseSchemas:
    """Test Pydantic schemas for SQL agents."""

    def test_sql_response_schema(self):
        """Test SQLResponse schema validation."""
        from src.agents.strand_agents.sql.generator import SQLResponse

        # Valid response
        response = SQLResponse(
            reasoning="Test reasoning",
            sql="SELECT * FROM table",
            confidence=0.9,
        )

        assert response.reasoning == "Test reasoning"
        assert response.sql == "SELECT * FROM table"
        assert response.confidence == 0.9

        # Test confidence bounds
        with pytest.raises(ValueError):
            SQLResponse(
                reasoning="Test",
                sql="SELECT 1",
                confidence=1.5,  # > 1.0
            )

    def test_sql_correction_response_schema(self):
        """Test SQLCorrectionResponse schema validation."""
        from src.agents.strand_agents.sql.corrector import SQLCorrectionResponse

        response = SQLCorrectionResponse(
            corrected_sql="SELECT * FROM users",
            confidence=0.95,
            changes_summary="Fixed table name",
        )

        assert response.corrected_sql == "SELECT * FROM users"
        assert response.confidence == 0.95
        assert response.changes_summary == "Fixed table name"

    def test_sql_spacing_analysis_schema(self):
        """Test SQLSpacingAnalysis schema validation."""
        from src.agents.strand_agents.sql.formatter import (
            SpacingIssue,
            SQLSpacingAnalysis,
        )

        issue = SpacingIssue(
            location="Around FROM",
            issue_type="missing_space",
            original_text="SELECT*FROM",
            fixed_text="SELECT * FROM",
        )

        analysis = SQLSpacingAnalysis(
            fixed_sql="SELECT * FROM table",
            issues_found=[issue],
            confidence=0.95,
            requires_fixes=True,
            athena_compatible=True,
            fix_summary="Fixed spacing",
        )

        assert analysis.fixed_sql == "SELECT * FROM table"
        assert len(analysis.issues_found) == 1
        assert analysis.confidence == 0.95

        # Test empty SQL validation
        with pytest.raises(ValueError, match="fixed_sql cannot be empty"):
            SQLSpacingAnalysis(
                fixed_sql="",
                confidence=0.9,
                requires_fixes=False,
                athena_compatible=True,
                fix_summary="None",
            )


class TestCompatibilityWrappers:
    """Test compatibility wrapper classes."""

    def test_sql_writer_agent_wrapper(self):
        """Test SQLWriterAgent wrapper initialization."""
        from src.agents.strand_agents.sql.generator import SQLWriterAgent

        # NOTE: This will make actual model initialization
        # In production, this would be mocked
        agent = SQLWriterAgent()

        assert agent.generator is not None
        assert hasattr(agent, "generate_sql")

    def test_sql_fixer_wrapper(self):
        """Test SQLFixer wrapper initialization."""
        from src.agents.strand_agents.sql.corrector import SQLFixer

        fixer = SQLFixer()

        assert fixer.sql_corrector is not None
        assert hasattr(fixer, "refine_sql")

    def test_sql_spacing_agent_wrapper(self):
        """Test SQLSpacingAgent wrapper initialization."""
        from src.agents.strand_agents.sql.formatter import SQLSpacingAgent

        agent = SQLSpacingAgent()

        assert agent.sql_formatter is not None
        assert hasattr(agent, "fix_sql_spacing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
