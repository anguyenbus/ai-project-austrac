"""
SQL Corrector Agent - Extract structured SQL corrections using Strands framework.

This agent uses LLM-based SQL error correction with structured outputs to fix SQL queries.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    update_langfuse_with_usage,
)

# Import the existing prompt builder
from ...prompt_builders.sql.corrector import SQLCorrectorPrompts
from ..model_config import get_agent_config


class SQLCorrectionResponse(BaseModel):
    """Structured response for SQL correction."""

    corrected_sql: str = Field(
        description="The corrected SQL query that fixes the error"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the correction (0.0 to 1.0)"
    )
    changes_summary: str = Field(
        description="Brief summary of what was changed to fix the error"
    )


@dataclass
class SQLFixResult:
    """Result from SQL fixing."""

    corrected_sql: str
    success: bool
    error_message: str = ""
    confidence: float = 0.0


class SQLCorrector:
    """Strands-based SQL corrector that fixes SQL queries using structured outputs."""

    def __init__(
        self,
        aws_region: str | None = None,
        model_id: str | None = None,
        session_id: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize the SQLCorrector.

        Args:
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM
            session_id: Session identifier for tracking
            debug_mode: Enable debug logging

        """
        # Get configuration from centralized config
        config = get_agent_config("SQLCorrector", aws_region)

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
        config = get_agent_config("SQLCorrector", self.aws_region)

        # Create model
        model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
        )

        # Base instructions - the actual prompt will be generated per query
        base_instructions = """Fix SQL errors by analyzing the query and error message.
        Return corrected SQL with confidence score and summary of changes made."""

        return Agent(
            model=model,
            system_prompt=base_instructions,
            callback_handler=self._usage_callback,
        )

    @observe(as_type="generation")
    def refine_sql(
        self, sql: str, error: str, schema_context: dict | None = None
    ) -> SQLFixResult:
        """
        Fix SQL error using Strands agent with structured output.

        Args:
            sql: The broken SQL query
            error: Error message from database execution
            schema_context: Optional schema information with tables/columns

        Returns:
            SQLFixResult with corrected SQL

        """
        if not sql or not sql.strip():
            return SQLFixResult(
                corrected_sql=sql,
                success=False,
                error_message="Empty SQL query provided",
            )

        if not error or not error.strip():
            return SQLFixResult(
                corrected_sql=sql,
                success=False,
                error_message="No error message provided",
            )

        # Build prompt using existing prompt builder
        prompt = SQLCorrectorPrompts.build_sql_fix_prompt(sql, error, schema_context)

        try:
            # Reset usage tracking
            self._usage_container["last_usage"] = None

            # Run the agent with the built prompt
            result = self.agent.structured_output(SQLCorrectionResponse, prompt)

            # Update Langfuse with token usage
            update_langfuse_with_usage(
                self._usage_container, self.model_id, "SQLFixer", langfuse_context
            )

            # Extract structured result
            result_data = result

            corrected_sql = result_data.corrected_sql
            confidence = result_data.confidence

            if not corrected_sql or corrected_sql.strip() == sql.strip():
                return SQLFixResult(
                    corrected_sql=sql,
                    success=False,
                    error_message="No changes made to SQL",
                    confidence=confidence,
                )

            fix_result = SQLFixResult(
                corrected_sql=corrected_sql,
                success=True,
                confidence=confidence,
            )
            # Attach the LLM response for token tracking compatibility
            fix_result.llm_response = result_data
            return fix_result

        except Exception as e:
            return SQLFixResult(
                corrected_sql=sql,
                success=False,
                error_message=f"Error correction failed: {str(e)}",
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


class SQLFixer:
    """
    Drop-in replacement for the existing SQLFixer using Strands framework.

    Maintains compatibility with existing code while using Strands under the hood.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the SQLFixer.

        Args:
            **kwargs: Configuration parameters (ignored for compatibility)

        """
        # Extract any relevant parameters from kwargs if needed
        debug_mode = kwargs.get("debug_mode", True)

        self.sql_corrector = SQLCorrector(
            debug_mode=debug_mode,
        )

    def refine_sql(
        self, sql: str, error: str, schema_context: dict | None = None
    ) -> SQLFixResult:
        """
        Fix SQL error using Strands-based corrector.

        Args:
            sql: The broken SQL query
            error: Error message from database execution
            schema_context: Optional schema information with tables/columns

        Returns:
            SQLFixResult with corrected SQL

        """
        return self.sql_corrector.refine_sql(sql, error, schema_context)

    def _build_fix_prompt(
        self, sql: str, error: str, schema_context: dict | None
    ) -> str:
        """
        Compatibility method for existing code that might call this method.

        Args:
            sql: Broken SQL query
            error: Database error message
            schema_context: Optional schema information

        Returns:
            Focused prompt string

        """
        return SQLCorrectorPrompts.build_sql_fix_prompt(sql, error, schema_context)


# Convenience function for quick usage
def fix_sql_error(sql: str, error: str, schema_context: dict | None = None) -> str:
    """
    Quick function to fix SQL error.

    Args:
        sql: Broken SQL query
        error: Database error message
        schema_context: Optional schema information

    Returns:
        Corrected SQL query (or original if fix failed)

    """
    fixer = SQLFixer()
    result = fixer.refine_sql(sql, error, schema_context)

    if result.success:
        return result.corrected_sql
    else:
        # NOTE: Using print instead of logger to avoid import issues
        print(f"SQL fix failed: {result.error_message}")
        return sql


# For backwards compatibility
def create_sql_refiner_agent(**kwargs):
    """Create SQLFixer (SQLRefinerAgent is deprecated)."""
    return SQLFixer(**kwargs)
