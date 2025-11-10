"""
SQL Writer Agent for generating SQL queries using Strands framework.

Handles SQL generation with schema context, examples, and filter processing.
Uses Strands Agent for structured LLM responses and reliable SQL generation.
"""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.strand_agents.query.clarifier import ClarificationAgent
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    update_langfuse_with_usage,
)

from ...prompt_builders.prompts import Prompts
from ..model_config import get_agent_config


class SQLResponse(BaseModel):
    """Structured SQL generation response with reasoning and confidence."""

    reasoning: str = Field(
        description="Explanation of column choices and mapping logic"
    )
    sql: str = Field(description="The generated SQL query")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the generated SQL"
    )


class SQLGenerator:
    """Core Strands-based SQL generation implementation."""

    def __init__(
        self,
        model_id: str | None = None,
        aws_region: str | None = None,
        debug_mode: bool = False,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize SQLGenerator with Strands Agent.

        Args:
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            debug_mode: Enable debug logging
            session_id: Session identifier for tracking

        """
        # Get configuration from centralized config
        config = get_agent_config("SQLGenerator", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or config["aws_region"]
        self.model_id = model_id or config["model_id"]
        self.debug_mode = debug_mode
        self.session_id = session_id

        # Create callback for usage tracking
        self._usage_callback, self._usage_container = create_usage_callback()

        # Initialize Strands Agent
        self.agent = self._build_agent(config)

        logger.info("✓ SQLGenerator (strand) initialized")

    def _build_agent(self, config: dict) -> Agent:
        """Build Strands agent with BedrockModel."""
        model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
        )

        # Base instructions from prompts
        base_instructions = Prompts.SQL_GENERATOR

        return Agent(
            model=model,
            system_prompt=base_instructions,
            callback_handler=self._usage_callback,
        )

    @observe(as_type="generation")
    def generate_sql(
        self,
        question: str,
        schema_context: list[str],
        example_context: list[str],
        selected_tables: list[str],
        search_results: list[dict],
        best_similarity: float,
        filter_context: dict[str, Any] | None = None,
        prior_turns: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate SQL with conversation history.

        Args:
            question: Natural language query
            schema_context: List of schema information strings
            example_context: List of example SQL queries
            selected_tables: List of selected table names
            search_results: List of search result dictionaries
            best_similarity: Best similarity score from search
            filter_context: Optional filter information
            prior_turns: Optional list of prior conversation turns

        Returns:
            Dict containing SQL, confidence, reasoning, and metadata

        """
        try:
            # Combine contexts
            combined_context = []
            if schema_context:
                # Ensure schema_context is a list, not a string
                if isinstance(schema_context, str):
                    schema_list = [schema_context]
                else:
                    schema_list = schema_context
                combined_context.append(
                    "SCHEMA INFORMATION:\n" + "\n".join(schema_list)
                )
            if example_context:
                # Ensure example_context is a list, not a string
                if isinstance(example_context, str):
                    example_list = [example_context]
                else:
                    example_list = example_context
                combined_context.append("SQL EXAMPLES:\n" + "\n".join(example_list))
            context_text = "\n\n".join(combined_context)

            # Process filter context
            filter_prompt = self._build_filter_prompt(filter_context)

            # Build conversation history prompt
            history_prompt = self._build_history_prompt(prior_turns)

            # Build user message with dynamic content only (system prompt is in agent)
            user_message = f"""{history_prompt}USER QUESTION: {question}

SELECTED TABLES: {selected_tables if selected_tables else "Auto-selected"}

{context_text}

{filter_prompt if filter_prompt else ""}"""

            # Reset usage tracking
            self._usage_container["last_usage"] = None

            # Call Strands Agent
            result = self.agent.structured_output(SQLResponse, user_message)

            # Update Langfuse with token usage
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "SQLGenerator",
                langfuse_context,
            )

            # Extract structured result
            llm_result = result

            # Handle low confidence responses
            if llm_result.confidence < 0.5:
                return self._handle_low_confidence_response(
                    llm_result, question, context_text, selected_tables
                )

            # Post-process SQL
            processed_sql = self._post_process_sql(llm_result.sql)

            # Calculate confidence combining LLM confidence with context quality
            context_confidence = min(0.7 + (best_similarity * 0.3), 1.0)
            if selected_tables:
                context_confidence = min(
                    context_confidence + 0.1, 1.0
                )  # Boost for table selection

            # Combine LLM confidence with context confidence
            combined_confidence = min(
                (llm_result.confidence + context_confidence) / 2, 1.0
            )

            return {
                "sql": processed_sql,
                "confidence": combined_confidence,
                "reasoning": llm_result.reasoning,
                "sources": [
                    {
                        "chunk_type": r.get("chunk_type"),
                        "similarity": r.get("similarity"),
                        "content_preview": r.get("content", "")[:100],
                    }
                    for r in search_results[:5]
                ],
                "context_used": context_text,
                "selected_tables": selected_tables,
                "metadata": {
                    "best_similarity": best_similarity,
                    "context_length": len(context_text),
                    "sources_count": len(search_results),
                    "llm_confidence": llm_result.confidence,
                    "context_confidence": context_confidence,
                },
            }

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise e

    def _build_history_prompt(self, prior_turns: list[dict[str, str]] | None) -> str:
        """
        Build conversation history prompt.

        Includes SQL and visualization from prior turns for context.
        """
        if not prior_turns or len(prior_turns) == 0:
            return ""

        history_lines = ["CONVERSATION HISTORY:"]

        for i, turn in enumerate(prior_turns, 1):
            content = turn.get("content", "")
            sql = turn.get("sql", "")
            viz = turn.get("visualization")

            history_lines.append(f"\nTurn {i}:")
            history_lines.append(f"  User: {content}")

            if sql:
                # Show abbreviated SQL (first 100 chars)
                sql_preview = sql[:100] + ("..." if len(sql) > 100 else "")
                history_lines.append(f"  SQL: {sql_preview}")

            if viz:
                viz_type = viz.get("type", "unknown")
                history_lines.append(f"  Visualization: {viz_type} chart")

        history_lines.append("\nCURRENT REQUEST:")
        history_lines.append("The user is continuing the conversation above.")
        history_lines.append(
            "Consider references to 'beginning', 'first', 'original', etc."
        )
        history_lines.append(
            "If the user says 'beginning visual', use the visualization from Turn 1."
        )
        history_lines.append("If unrelated to history, generate fresh SQL.\n")

        return "\n".join(history_lines) + "\n\n"

    def _build_filter_prompt(self, filter_context: dict[str, Any] | None) -> str:
        """Build filter prompt section based on filter context."""
        if not filter_context:
            return ""

        filter_prompt = ""
        if "mapped_filters" in filter_context:
            # ColumnAgent has mapped filters to exact columns
            filter_prompt = "\n\nEXACT FILTERS (Already mapped to columns):\n"
            for filter_item in filter_context["mapped_filters"]:
                for col, val in filter_item.items():
                    if isinstance(val, list):
                        # Multiple values - use IN clause
                        values_str = ", ".join([f"'{v}'" for v in val])
                        filter_prompt += f"- {col} IN ({values_str})\n"
                    else:
                        # Single value - use equality
                        filter_prompt += f"- {col} = '{val}'\n"
            filter_prompt += "\nINCORPORATE THESE EXACT FILTERS IN THE WHERE CLAUSE. Use IN clause for multiple values, = for single values."

        elif "semantic_filters" in filter_context:
            # FilteringAgent has extracted semantic filters
            filter_prompt = "\n\nSEMANTIC FILTERS (Map to appropriate columns):\n"
            for filter_item in filter_context["semantic_filters"]:
                for key, val in filter_item.items():
                    filter_prompt += f"- {key}: {val}\n"
            filter_prompt += (
                "\nMATCH THESE FILTERS TO THE APPROPRIATE COLUMNS IN THE WHERE CLAUSE."
            )

        return filter_prompt

    def _handle_low_confidence_response(
        self,
        response: SQLResponse,
        question: str,
        context_text: str,
        selected_tables: list[str],
    ) -> dict[str, Any]:
        """Handle low confidence responses with clarification."""
        clarification_result = ClarificationAgent().generate_clarification_response(
            response.sql, question
        )
        logger.info(f"Clarification result: {clarification_result}")

        # Add "CANNOT FIND TABLES" to the clarification result
        clarification_result = "CANNOT FIND TABLES: " + clarification_result

        return {
            "sql": clarification_result,
            "confidence": response.confidence,
            "reasoning": f"Low confidence ({response.confidence:.2f}) - user clarification needed: {response.reasoning}",
            "sources": [],
            "context_used": context_text,
            "selected_tables": selected_tables,
            "metadata": {
                "context_length": len(context_text),
                "llm_confidence": response.confidence,
            },
        }

    def _post_process_sql(self, sql: str) -> str:
        """Post-process SQL with spacing and formatting."""
        if not sql:
            return ""

        try:
            from src.agents.strand_agents.sql.formatter import SQLSpacingAgent

            spacing_result = SQLSpacingAgent().fix_sql_spacing(sql)
            return spacing_result.fixed_sql
        except Exception as e:
            logger.warning(f"SQL spacing failed: {e}")
            return sql

    def _create_error_response(self, error_msg: str) -> dict[str, Any]:
        """Create standardized error response."""
        return {
            "sql": "",
            "confidence": 0.0,
            "reasoning": f"Error during SQL generation: {error_msg}",
            "sources": [],
            "context_used": "",
            "selected_tables": [],
            "metadata": {"error": error_msg},
        }


class SQLWriterAgent:
    """Compatibility wrapper maintaining original API."""

    def __init__(self) -> None:
        """Initialize with Strands implementation."""
        # Initialize core Strands implementation with config defaults
        self.generator = SQLGenerator(debug_mode=False)

        # Expose attributes for backward compatibility
        self.region = self.generator.aws_region
        self.model_id = self.generator.model_id

        logger.info("✓ SQLWriterAgent initialized")

    def get_token_stats(self) -> dict:
        """
        Get token usage statistics.

        Returns:
            Dict with total_calls and total_tokens

        """
        # NOTE: Token tracking is now handled by Langfuse via callback handler
        # Return empty stats for backward compatibility
        return {
            "total_calls": 0,
            "total_tokens": 0,
        }

    def generate_sql(
        self,
        question: str,
        schema_context: list[str],
        example_context: list[str],
        selected_tables: list[str],
        search_results: list[dict],
        best_similarity: float,
        filter_context: dict[str, Any] | None = None,
        prior_turns: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate SQL query using LLM with structured context.

        Delegates to core Strands implementation.
        """
        return self.generator.generate_sql(
            question=question,
            schema_context=schema_context,
            example_context=example_context,
            selected_tables=selected_tables,
            search_results=search_results,
            best_similarity=best_similarity,
            filter_context=filter_context,
            prior_turns=prior_turns,
        )
