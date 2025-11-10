"""
SQL Spacing Agent - Fix SQL spacing issues using Strands framework.

This agent uses LLM with structured outputs to fix SQL spacing issues for database compatibility.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    update_langfuse_with_usage,
)

# Import the existing prompt builder
from ...prompt_builders.sql.formatter import SQLFormatterPrompts
from ..model_config import get_agent_config


class SpacingIssue(BaseModel):
    """Individual spacing issue found in SQL."""

    location: str = Field(
        description="Where the issue was found (e.g., 'Before FROM keyword', 'Around operator =')"
    )
    issue_type: str = Field(
        description="Type of issue: 'missing_space', 'extra_space', 'operator_spacing', 'keyword_spacing', 'quote_conversion'"
    )
    original_text: str = Field(description="The problematic text snippet")
    fixed_text: str = Field(description="The corrected text snippet")

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, v: str) -> str:
        """Validate issue_type is one of the expected types."""
        valid_types = {
            "missing_space",
            "extra_space",
            "operator_spacing",
            "keyword_spacing",
            "quote_conversion",
        }
        if v not in valid_types:
            raise ValueError(f"issue_type must be one of {valid_types}")
        return v


class SQLSpacingAnalysis(BaseModel):
    """Structured analysis of SQL spacing issues and fixes."""

    fixed_sql: str = Field(
        description="The SQL query with all spacing issues corrected for database compatibility"
    )
    issues_found: list[SpacingIssue] = Field(
        default=[], description="List of specific spacing issues found and fixed"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence level in the fixes applied (0.0 to 1.0)"
    )
    requires_fixes: bool = Field(
        description="Whether the SQL required any spacing fixes"
    )
    fix_summary: str = Field(description="Brief summary of what was fixed")

    @field_validator("fixed_sql")
    @classmethod
    def validate_sql_not_empty(cls, v: str) -> str:
        """Validate fixed_sql is not empty."""
        if not v or not v.strip():
            raise ValueError("fixed_sql cannot be empty")
        return v.strip()


@dataclass
class SQLSpacingResult:
    """Result from SQL spacing agent - maintains backward compatibility."""

    fixed_sql: str
    issues_found: list[str]
    confidence: float
    success: bool


class SQLFormatter:
    """Strands-based SQL formatter that fixes spacing issues using structured outputs."""

    def __init__(
        self,
        aws_region: str | None = None,
        model_id: str | None = None,
        session_id: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize the SQLFormatter.

        Args:
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM
            session_id: Session identifier for tracking
            debug_mode: Enable debug logging

        """
        # Get configuration from centralized config
        config = get_agent_config("SQLFormatter", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or config["aws_region"]
        self.model_id = model_id or config["model_id"]
        self.debug_mode = debug_mode
        self.session_id = session_id

        # Create callback for usage tracking
        self._usage_callback, self._usage_container = create_usage_callback()

        # Initialize the agent
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        """Initialize the Strands agent with structured output."""
        # Get configuration
        config = get_agent_config("SQLFormatter", self.aws_region)

        # Create the Bedrock model using config values
        model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
        )

        # Base instructions - the actual prompt will be generated per query
        base_instructions = """Fix SQL spacing issues for database compatibility.
        Analyze spacing problems and return corrected SQL with detailed issue tracking."""

        return Agent(
            model=model,
            system_prompt=base_instructions,
            callback_handler=self._usage_callback,
        )

    @observe(as_type="generation")
    def fix_sql_spacing(self, sql: str) -> SQLSpacingResult:
        """
        Fix SQL spacing issues using Strands agent with structured output.

        Args:
            sql: The SQL query with potential spacing issues

        Returns:
            SQLSpacingResult with fixed SQL and metadata

        """
        if not sql or not sql.strip():
            return SQLSpacingResult(
                fixed_sql=sql, issues_found=[], confidence=1.0, success=True
            )

        # Build prompt using existing prompt builder
        prompt = SQLFormatterPrompts.build_spacing_fix_prompt(sql)

        try:
            # Reset usage tracking
            self._usage_container["last_usage"] = None

            # Run the agent with the built prompt
            result = self.agent.structured_output(SQLSpacingAnalysis, prompt)

            # Update Langfuse with token usage
            update_langfuse_with_usage(
                self._usage_container, self.model_id, "SQLFormatter", langfuse_context
            )

            # Extract structured result
            analysis = result

            # Convert SpacingIssue objects to strings for backward compatibility
            issues_found = []
            for issue in analysis.issues_found:
                issue_str = f"{issue.location}: {issue.issue_type} - '{issue.original_text}' → '{issue.fixed_text}'"
                issues_found.append(issue_str)

            # Add summary if no specific issues but fixes were made
            if not issues_found and analysis.requires_fixes:
                issues_found.append(analysis.fix_summary)

            spacing_result = SQLSpacingResult(
                fixed_sql=analysis.fixed_sql,
                issues_found=issues_found,
                confidence=analysis.confidence,
                success=True,
            )
            # Attach the LLM response for token tracking compatibility
            spacing_result.llm_response = analysis
            return spacing_result

        except Exception as e:
            return SQLSpacingResult(
                fixed_sql=sql,
                issues_found=[f"Agent error: {str(e)}"],
                confidence=0.0,
                success=False,
            )

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "aws_region": self.aws_region,
            "model_id": self.model_id,
            "debug_mode": self.debug_mode,
        }

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        """
        for key, value in kwargs.items():
            if key == "aws_region":
                self.aws_region = value
            elif key == "model_id":
                self.model_id = value
            elif key == "debug_mode":
                self.debug_mode = value


class SQLSpacingAgent:
    """
    Drop-in replacement for the existing SQLSpacingAgent using Strands framework.

    Maintains compatibility with existing code while using Strands under the hood.
    """

    def __init__(self, model_id: str | None = None, **kwargs) -> None:
        """
        Initialize the SQLSpacingAgent.

        Args:
            model_id: Optional model ID to use (overrides config)
            **kwargs: Additional configuration parameters

        """
        debug_mode = kwargs.get("debug_mode", True)
        aws_region = kwargs.get("aws_region", None)

        self.sql_formatter = SQLFormatter(
            model_id=model_id,
            aws_region=aws_region,
            debug_mode=debug_mode,
        )

        # Store model_id for compatibility
        self.model_id = model_id
        # Compatibility attributes
        self.llm = None
        self.instructor_client = self.sql_formatter.agent  # For compatibility

    def fix_sql_spacing(self, sql: str) -> SQLSpacingResult:
        """
        Fix SQL spacing issues using Strands-based formatter.

        Args:
            sql: The SQL query with potential spacing issues

        Returns:
            SQLSpacingResult with fixed SQL and metadata

        """
        return self.sql_formatter.fix_sql_spacing(sql)

    def _fix_with_instructor(self, sql: str) -> dict:
        """
        Compatibility method for existing code that might call this method.

        Args:
            sql: The SQL query to fix

        Returns:
            Dictionary with result and llm_response for token tracking

        """
        result = self.sql_formatter.fix_sql_spacing(sql)
        return {"result": result, "llm_response": getattr(result, "llm_response", None)}

    def _build_spacing_prompt(self, sql: str) -> str:
        """
        Compatibility method for existing code that might call this method.

        Args:
            sql: The SQL query to analyze

        Returns:
            Formatted prompt string

        """
        return SQLFormatterPrompts.build_spacing_fix_prompt(sql)


# Global agent instance
_sql_spacing_agent: SQLSpacingAgent | None = None


def get_sql_spacing_agent() -> SQLSpacingAgent:
    """Get the global SQL spacing agent instance."""
    global _sql_spacing_agent
    if _sql_spacing_agent is None:
        _sql_spacing_agent = SQLSpacingAgent()
    return _sql_spacing_agent


def fix_sql_spacing_with_llm(sql: str) -> str:
    """
    Fix SQL spacing using LLM agent.

    Args:
        sql: The SQL query to fix

    Returns:
        The SQL query with proper spacing

    """
    agent = get_sql_spacing_agent()
    result = agent.fix_sql_spacing(sql)

    if result.success:
        # NOTE: Using print instead of logger to avoid import issues
        print(f"SQL spacing fixed by LLM agent (confidence: {result.confidence:.2f})")
        if result.issues_found:
            print(f"Issues fixed: {result.issues_found}")
        return result.fixed_sql
    else:
        print("SQL spacing agent failed, returning original SQL")
        return sql


def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL query from text that might contain explanations or markdown.

    Args:
        text: Text potentially containing SQL

    Returns:
        Extracted SQL query

    """
    import re

    # Remove markdown code blocks
    if "```sql" in text.lower():
        match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Find the first SQL keyword and extract from there
    sql_keywords = [
        "SELECT",
        "WITH",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
    ]

    lines = text.split("\n")
    sql_start_idx = None

    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        if any(line_upper.startswith(kw) for kw in sql_keywords):
            sql_start_idx = i
            break

    if sql_start_idx is not None:
        # Extract from the SQL start to the end
        sql_lines = lines[sql_start_idx:]

        # Find where SQL ends (usually at an empty line or line starting with text)
        sql_end_idx = len(sql_lines)
        for i, line in enumerate(sql_lines):
            if (
                i > 0
                and line.strip()
                and not any(
                    char in line
                    for char in [
                        ";",
                        "(",
                        ")",
                        ",",
                        "SELECT",
                        "FROM",
                        "WHERE",
                        "GROUP",
                        "ORDER",
                        "HAVING",
                        "JOIN",
                        "ON",
                        "AND",
                        "OR",
                    ]
                )
            ):
                # This line doesn't look like SQL
                sql_end_idx = i
                break

        sql = "\n".join(sql_lines[:sql_end_idx])
    else:
        # No SQL found, return original text
        sql = text

    # Apply spacing enforcement
    sql = fix_sql_spacing_with_llm(sql.strip())

    return sql


# Backward compatibility aliases
def enforce_sql_spacing(sql: str) -> str:
    """Backward compatibility alias for fix_sql_spacing_with_llm."""
    return fix_sql_spacing_with_llm(sql)
