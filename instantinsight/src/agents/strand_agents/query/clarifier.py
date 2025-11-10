"""
ClarificationAgent - Intelligent Query Clarification System using Strand framework.

This agent determines when user queries are too vague or ambiguous to generate accurate SQL,
and provides helpful clarification responses to guide users toward more specific queries.
Migrated to use Strand framework for enhanced LLM-powered clarification generation.
"""

import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.prompts import Prompts
from src.agents.prompt_builders.query.clarifier import QueryClarificationPrompts
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ..model_config import get_agent_config


class ClarificationConfig:
    """Configuration for ClarificationAgent."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        enable_llm_clarification: bool = True,
        aws_region: str = None,
        model_id: str = None,
        max_tables_to_mention: int = 3,
        max_examples: int = 3,
    ):
        """
        Initialize clarification configuration.

        Args:
            confidence_threshold: Minimum confidence threshold for table selection
            enable_llm_clarification: Enable LLM-based clarification generation
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM
            max_tables_to_mention: Maximum tables to mention in clarification
            max_examples: Maximum example queries to provide

        """
        self.confidence_threshold = confidence_threshold
        self.enable_llm_clarification = enable_llm_clarification
        self.aws_region = aws_region
        self.model_id = model_id
        self.max_tables_to_mention = max_tables_to_mention
        self.max_examples = max_examples


class ClarificationResponse(BaseModel):
    """Structured response from LLM for clarification."""

    message: str = Field(
        description="A helpful clarification message explaining why SQL cannot be generated"
    )
    examples: list[str] = Field(
        description="2-3 example queries the user could try instead",
        default_factory=list,
    )
    reasoning: str = Field(
        description="Internal reasoning for why clarification was needed"
    )

    @field_validator("examples", mode="before")
    @classmethod
    def parse_examples(cls, v):
        """Parse examples if they come as a JSON string."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v


class ClarificationAgent:
    """Strand-based clarification implementation."""

    def __init__(
        self,
        config: ClarificationConfig | None = None,
        *,
        model_id: str = None,
        aws_region: str = None,
        debug_mode: bool = False,
        session_id: str = None,
    ):
        """
        Initialize ClarificationAgent with Strand Agent.

        Args:
            config: Clarification configuration object
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            debug_mode: Enable debug logging
            session_id: Session identifier for tracking

        """
        self.config = config or ClarificationConfig()

        # Get configuration from centralized config
        agent_config = get_agent_config("ClarificationAgent", aws_region)

        # Use provided values, then config values, then defaults
        self.aws_region = (
            aws_region or self.config.aws_region or agent_config["aws_region"]
        )
        self.model_id = model_id or self.config.model_id or agent_config["model_id"]
        self.debug_mode = debug_mode
        self.session_id = session_id

        # Model configuration for Strand
        # Note: Bedrock region is set via AWS_REGION env var or AWS credentials
        self._cache_prompt_type = agent_config.get("cache_prompt")

        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            streaming=False,
            cache_prompt=self._cache_prompt_type,
        )

        # Base instructions for clarification generation
        base_instructions = Prompts.QUERY_CLARIFIER

        # Initialize Strand Agent
        # Create callback for usage tracking

        callback, self._usage_container = create_usage_callback()

        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

        logger.info("✓ ClarificationAgent (strand) initialized")

    def needs_clarification(self, table_selection) -> bool:
        """
        Check if clarification is needed based on table selection confidence scores.

        This method analyses various signals from the table selection process to determine
        if the user query is too ambiguous to generate reliable SQL.

        Args:
            table_selection: Result from table selection process

        Returns:
            True if clarification is needed, False otherwise

        """
        if not table_selection:
            logger.info("📊 No table selection result - clarification needed")
            return True

        # Check if confidence scores are too low
        confidence_threshold = self.config.confidence_threshold

        # Get average confidence score for selected tables
        if table_selection.confidence_scores and table_selection.selected_tables:
            avg_confidence = sum(
                table_selection.confidence_scores.get(table, 0.0)
                for table in table_selection.selected_tables
            ) / len(table_selection.selected_tables)

            if avg_confidence < confidence_threshold:
                logger.info(
                    f"📊 Average confidence {avg_confidence:.2f} below threshold {confidence_threshold}"
                )
                return True

        # Check if too many similar tables (ambiguity)
        if (
            table_selection.duplicate_analysis
            and table_selection.duplicate_analysis.get("has_duplicates")
        ):
            logger.info("🔄 Duplicate tables detected - clarification needed")
            return True

        # Check if query is too complex and multiple interpretations exist
        if hasattr(table_selection, "metadata") and table_selection.metadata.get(
            "ambiguous_query"
        ):
            logger.info("❓ Ambiguous query detected - clarification needed")
            return True

        # Check if no tables were selected with confidence
        if not table_selection.selected_tables:
            logger.info("📊 No tables selected - clarification needed")
            return True

        return False

    @observe(as_type="generation")
    def generate_clarification_response(
        self, table_selection_or_context, question: str
    ) -> str:
        """
        Generate a clarification response using Strand Agent for enhanced responses.

        Args:
            table_selection_or_context: Either a TableSelectionResult object or a dict with context info
            question: Original user question

        Returns:
            Natural language response explaining why SQL cannot be generated

        """
        # Handle different input types
        if hasattr(table_selection_or_context, "selected_tables"):
            # TableSelectionResult object
            context = self._extract_table_selection_context(
                table_selection_or_context, question
            )
        elif isinstance(table_selection_or_context, dict):
            # Dictionary context (for SQL generation, etc.)
            context = self._extract_dict_context(table_selection_or_context, question)
        else:
            # Fallback - treat as string reasoning
            context = {
                "question": question,
                "reasoning": (
                    str(table_selection_or_context)
                    if table_selection_or_context
                    else "Query needs clarification"
                ),
                "context_type": "general",
            }

        # Log the context analysis
        logger.info("📝 Clarification Context Analysis:")
        logger.info(f"   - Context type: {context.get('context_type', 'unknown')}")
        logger.info(
            f"   - Reasoning: {context.get('reasoning', 'No reasoning provided')}"
        )

        # Try Strand generation if enabled
        if self.config.enable_llm_clarification:
            try:
                clarification_text = self._generate_strand_clarification(context)
                if clarification_text:
                    return self._format_as_sql_comments(clarification_text)
            except Exception as e:
                logger.warning(f"Strand clarification generation failed: {e}")

        # Fallback to template response
        clarification_text = self._generate_template_clarification(context)
        return self._format_as_sql_comments(clarification_text)

    @observe(as_type="generation")
    def _generate_strand_clarification(self, context: dict[str, Any]) -> str | None:
        """
        Generate clarification using Strand Agent with structured prompt.

        Args:
            context: Analysis context from table selection

        Returns:
            Generated clarification text or None if failed

        """
        try:
            # Build structured prompt using existing prompt builder
            prompt = QueryClarificationPrompts.build_clarification_prompt(
                context, self.config.max_examples
            )

            # Call Strand Agent with structured output
            # Reset usage

            self._usage_container["last_usage"] = None

            # Run agent

            llm_result = self.agent.structured_output(ClarificationResponse, prompt)

            # Update Langfuse with usage and costs

            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "ClarificationAgent",
                langfuse_context,
            )
            self._log_prompt_cache_metrics()
            # Format the response nicely
            message = llm_result.message
            if llm_result.examples:
                message += "\n\nHere are some example queries you could try:\n"
                for i, example in enumerate(llm_result.examples[:3], 1):
                    message += f"{i}. {example}\n"

            logger.info("✅ Generated Strand clarification response")
            return message

        except Exception as e:
            logger.warning(f"Strand clarification failed: {e}")
            return None

    def _extract_table_selection_context(self, table_selection, question: str) -> dict:
        """Extract context from TableSelectionResult object."""
        return {
            "question": question,
            "context_type": "table_selection",
            "selected_tables": getattr(table_selection, "selected_tables", []),
            "confidence_scores": getattr(table_selection, "confidence_scores", {}),
            "reasoning": getattr(
                table_selection,
                "selection_reasoning",
                "Table selection had low confidence",
            ),
            "duplicate_analysis": getattr(table_selection, "duplicate_analysis", {}),
            "related_tables": getattr(table_selection, "related_tables", []),
        }

    def _extract_dict_context(self, context_dict: dict, question: str) -> dict:
        """Extract context from dictionary (for SQL generation, schema validation, etc.)."""
        context = {
            "question": question,
            "context_type": context_dict.get("type", "sql_generation"),
            "reasoning": context_dict.get(
                "reasoning", "Low confidence or validation failure"
            ),
        }

        # Add specific fields based on context type
        if "confidence" in context_dict:
            context["confidence"] = context_dict["confidence"]
        if "sql" in context_dict:
            context["sql"] = context_dict["sql"]
        if "tables" in context_dict:
            context["selected_tables"] = context_dict["tables"]
        if "error" in context_dict:
            context["error"] = context_dict["error"]

        return context

    def _generate_template_clarification(self, context: dict[str, Any]) -> str:
        """
        Generate template clarification response as fallback.

        Args:
            context: Analysis context from various sources

        Returns:
            Template-based clarification text

        """
        context_type = context.get("context_type", "general")

        if context_type == "table_selection":
            return self._generate_table_selection_template(context)
        elif context_type == "sql_generation":
            return self._generate_sql_generation_template(context)
        else:
            return self._generate_general_template(context)

    def _generate_table_selection_template(self, context: dict[str, Any]) -> str:
        """Generate template for table selection issues."""
        question = context.get("question", "")
        selected_tables = context.get("selected_tables", [])
        reasoning = context.get("reasoning", "Table selection had low confidence")

        tables_mentioned = ""
        if selected_tables:
            table_list = ", ".join(selected_tables[: self.config.max_tables_to_mention])
            tables_mentioned = f"\nPotentially related tables found: {table_list}"

        return f"""Your query "{question}" couldn't identify the right tables with confidence.

ISSUE: {reasoning}{tables_mentioned}

Please provide more specific details:
- What specific data or metrics do you need?
- Which business area or domain are you asking about?
- Any specific time periods or filters?

Examples of clearer queries:
- "Show total sales amount by product category for Q4 2023"
- "List all customers with account balance > 1000 in Sydney"
- "Find top 10 suppliers by order count this month\""""

    def _generate_sql_generation_template(self, context: dict[str, Any]) -> str:
        """Generate template for SQL generation issues."""
        question = context.get("question", "")
        reasoning = context.get("reasoning", "SQL generation had low confidence")
        confidence = context.get("confidence", 0.0)

        return f"""Your query "{question}" resulted in low confidence SQL ({confidence:.1f}).

ISSUE: {reasoning}

This usually means:
- The query might be too vague or ambiguous
- Required columns or relationships might not be clearly identified
- Multiple interpretations of the query are possible

Please provide more specific details:
- What specific data or metrics do you need?
- Clarify any ambiguous terms in your query
- Specify time periods, filters, or conditions if relevant

Examples of clearer queries:
- "Show total sales by region for Q4 2023"
- "List customers with overdue payments > 30 days"
- "Find top 10 products by revenue this month\""""

    def _generate_general_template(self, context: dict[str, Any]) -> str:
        """Generate template for general clarification issues."""
        question = context.get("question", "")
        reasoning = context.get("reasoning", "Query needs clarification")

        return f"""Your query "{question}" needs clarification.

ISSUE: {reasoning}

Please provide more specific details:
- What specific information are you looking for?
- Which business area or domain does this relate to?
- Any specific time periods, filters, or conditions?

Examples of clearer queries:
- "Show total sales by product category for Q4 2023"
- "List all customers with account balance > 1000 in Sydney"
- "Find top 10 suppliers by order count this month\""""

    def _format_as_sql_comments(self, text: str) -> str:
        """
        Format clarification text as SQL comments for consistency.

        Args:
            text: Raw clarification text

        Returns:
            Text formatted as SQL comments

        """
        if not text.startswith("--"):
            lines = text.strip().split("\n")
            text = "\n".join(
                f"-- {line}" if line and not line.startswith("--") else line
                for line in lines
            )

        return text

    def get_config(self) -> ClarificationConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def _log_prompt_cache_metrics(self) -> None:
        """Log prompt caching metrics for observability when enabled."""
        if not self._cache_prompt_type:
            logger.debug(
                "ClarificationAgent prompt cache metrics skipped (cache_prompt disabled)"
            )
            return None

        return log_prompt_cache_status("ClarificationAgent", self._usage_container)
