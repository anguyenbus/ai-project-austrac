"""
FilteringAgent - Extract structured filtering criteria from natural language queries using Strands framework.

This agent parses natural language queries to identify and extract filtering conditions
(temporal, categorical, numeric) that can be used to enhance SQL generation accuracy.
"""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ...prompt_builders.prompts import Prompts
from ...prompt_builders.schema.filter_builder import SchemaFilterPrompts
from ..model_config import get_agent_config


class FilteringResult(BaseModel):
    """Structured result from filtering analysis."""

    filterings: list[dict[str, Any]] = Field(
        description="List of filter key-value pairs with grouped values"
    )
    query_analysis: str = Field(description="Brief analysis of what filters were found")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in extracted filters"
    )
    question: str = Field(
        description="The original question that was used to extract the filters"
    )


class FilterAgent:
    """Strands-based FilterAgent that extracts structured filtering criteria from queries."""

    def __init__(
        self,
        aws_region: str | None = None,
        model_id: str | None = None,
        confidence_threshold: float = 0.6,
        session_id: str | None = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the FilterAgent.

        Args:
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM
            confidence_threshold: Minimum confidence threshold for filter extraction
            session_id: Session identifier for tracking
            debug_mode: Enable debug logging

        """
        # Get configuration from centralized config
        config = get_agent_config("SchemaFilterBuilder", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or config["aws_region"]
        self.model_id = model_id or config["model_id"]
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode

        # Create Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
            cache_prompt=config.get("cache_prompt"),
        )

        self._cache_prompt_type = config.get("cache_prompt")

        # Initialize the agent
        self.agent: Agent | None = None
        self._initialize_agent(session_id)

    def _initialize_agent(self, session_id: str | None = None):
        """Initialize the Strands agent with structured output."""
        # Use the system prompt directly as instructions
        base_instructions = Prompts.SCHEMA_FILTER_EXTRACTION

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

    @observe(as_type="generation")
    def extract_filters(
        self, query: str, filter_hints: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Extract filtering criteria from natural language query.

        Args:
            query: Natural language query
            filter_hints: Optional list of normalized hints guiding extraction

        Returns:
            Dictionary containing extracted filters and metadata

        """
        # Get the prompt from the existing prompt builder
        prompt = SchemaFilterPrompts.build_extraction_prompt(
            query, normalized_hints=filter_hints
        )

        try:
            # Reset usage
            self._usage_container["last_usage"] = None

            # Run the agent with structured output
            result = self.agent.structured_output(FilteringResult, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "FilteringAgent",
                langfuse_context,
            )

            if self._cache_prompt_type:
                log_prompt_cache_status("FilteringAgent", self._usage_container)

            # The result should be a FilteringResult instance
            if isinstance(result, FilteringResult):
                return {
                    "filterings": result.filterings,
                    "query_analysis": result.query_analysis,
                    "confidence": result.confidence,
                    "question": result.question,
                    "llm_response": result,  # Include for token tracking compatibility
                }
            else:
                # Fallback
                return {
                    "filterings": [],
                    "error": "Response not in expected format",
                    "raw_response": str(result),
                }

        except Exception as e:
            logger.error(f"Filter extraction failed: {e}")
            return {
                "filterings": [],
                "error": str(e),
            }

    def get_config(self):
        """Get the current configuration."""
        return {
            "aws_region": self.aws_region,
            "model_id": self.model_id,
            "confidence_threshold": self.confidence_threshold,
            "debug_mode": self.debug_mode,
        }

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        """
        for key, value in kwargs.items():
            if key == "aws_region":
                self.aws_region = value
                self.model.aws_region = value
            elif key == "model_id":
                self.model_id = value
                self.model.model_id = value
            elif key == "confidence_threshold":
                self.confidence_threshold = value
            elif key == "debug_mode":
                self.debug_mode = value


class FilterConfig:
    """Configuration for FilteringAgent - compatible with existing code."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        aws_region: str | None = None,
        model_id: str | None = None,
        max_filters: int = 10,
    ):
        """
        Initialize FilterConfig with filtering parameters.

        Args:
            confidence_threshold: Minimum confidence score for filter extraction
            aws_region: AWS region for Bedrock services
            model_id: Bedrock model identifier for LLM operations
            max_filters: Maximum number of filters to extract

        """
        self.confidence_threshold = confidence_threshold
        self.aws_region = aws_region
        self.model_id = model_id
        self.max_filters = max_filters


class FilteringAgent:
    """
    Drop-in replacement for the existing FilteringAgent using Strands framework.

    Maintains compatibility with existing code while using Strands under the hood.
    """

    def __init__(self, config: FilterConfig | None = None):
        """
        Initialize the FilteringAgent.

        Args:
            config: Configuration object for the agent

        """
        self.config = config or FilterConfig()
        self.filter_agent = FilterAgent(
            aws_region=self.config.aws_region,
            model_id=self.config.model_id,
            confidence_threshold=self.config.confidence_threshold,
            debug_mode=True,  # Enable for testing
        )

    @observe(as_type="generation")
    def extract_filters(
        self, query: str, normalized_hint: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Extract filtering criteria from natural language query.

        Args:
            query: Natural language query
            normalized_hint: Optional normalized query payload providing filter cues

        Returns:
            Dictionary containing extracted filters and metadata

        """
        hints = []
        if isinstance(normalized_hint, dict):
            details = normalized_hint.get("details_for_filterings")
            if isinstance(details, list):
                seen = set()
                for item in details:
                    if isinstance(item, str):
                        text = item.strip()
                        if text and text not in seen:
                            hints.append(text)
                            seen.add(text)
        return self.filter_agent.extract_filters(query, hints or None)

    def get_config(self) -> FilterConfig:
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
                # Also update the filter agent
                self.filter_agent.update_config(**{key: value})
