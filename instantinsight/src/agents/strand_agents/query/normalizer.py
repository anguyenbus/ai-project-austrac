"""Query normalization agent powered by Strand SDK."""

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.query.normalizer import QueryNormalizationPrompts
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    update_langfuse_with_usage,
)

from ..model_config import get_agent_config


class NormalizedQuery(BaseModel):
    """Structured normalized query representation."""

    main_clause: str = Field(description="Directive stripped of execution verbs")
    details_for_filterings: list[str] = Field(
        default_factory=list,
        description="Abstract filtering hints",
    )
    required_visuals: str | None = Field(
        default=None, description="Optional visualization requirement"
    )
    tables: list[str] | None = Field(
        default=None, description="Optional generalized table references"
    )

    @field_validator("details_for_filterings", mode="before")
    @classmethod
    def validate_details_for_filterings(cls, value: list[str] | None) -> list[str]:
        """
        Convert None to empty list for details_for_filterings.

        Args:
            value: The raw value from LLM output

        Returns:
            Empty list if None, otherwise the original value

        """
        if value is None:
            return []
        return value


class QueryNormalizationCore:
    """Strand-based normalization core."""

    def __init__(
        self,
        agent: Agent | None = None,
        model_id: str | None = None,
        aws_region: str | None = None,
        session_id: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize query normalization core.

        Args:
            agent: Optional pre-configured agent instance
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            session_id: Session identifier for tracking
            debug_mode: Enable debug logging

        """
        agent_config = get_agent_config("QueryNormalizer", aws_region)
        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]
        self.debug_mode = debug_mode

        # Initialize usage container (needed even when mock agent provided)
        self._usage_container = {"last_usage": None}

        self.agent = agent or self._build_agent(agent_config)
        logger.info("✓ QueryNormalizer (strand) initialized")

    def _build_agent(self, config: dict) -> Agent:
        model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
        )

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        return Agent(
            model=model,
            system_prompt=self._base_instructions(),
            callback_handler=callback,
        )

    @observe(as_type="generation")
    def normalize(
        self, query: str, prior_turns: list[dict] | None = None
    ) -> NormalizedQuery:
        """
        Normalize query into structured representation.

        Args:
            query: Natural language query to normalize
            prior_turns: Optional list of prior conversation turns

        Returns:
            Normalized query with structured components

        """
        prompt = self._build_prompt(query, prior_turns)
        try:
            # Reset usage
            self._usage_container["last_usage"] = None

            # Run agent
            result = self.agent.structured_output(NormalizedQuery, prompt)

            # NOTE: Diagnostic logging for debugging validation issues
            if self.debug_mode:
                logger.debug(f"Raw LLM output type: {type(result)}")
                logger.debug(f"Raw LLM output: {result}")

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "QueryNormalizer",
                langfuse_context,
            )

            return result

        except Exception as error:
            logger.error(f"Query normalization failed: {error}")
            if self.debug_mode:
                logger.exception("Full traceback:")
            return NormalizedQuery(
                main_clause="General business request",
                details_for_filterings=[],
            )

    def _build_prompt(self, query: str, prior_turns: list[dict] | None = None) -> str:
        """Build prompt for normalization with context."""
        return QueryNormalizationPrompts.build_prompt(query, prior_turns)

    def _base_instructions(self) -> str:
        return (
            "You normalize analytical questions into structured directives."
            "\nExtract the essence, abstract filters, note visualization needs,"
            " and output JSON matching NormalizedQuery."
        )


class QueryNormalizer:
    """Simple wrapper exposing normalize."""

    def __init__(self) -> None:
        """Initialize query normalizer with core implementation."""
        self.core = QueryNormalizationCore()

    def normalize(
        self, query: str, prior_turns: list[dict] | None = None
    ) -> NormalizedQuery:
        """
        Normalize query into structured representation.

        Args:
            query: Natural language query to normalize
            prior_turns: Optional list of prior conversation turns

        Returns:
            Normalized query with structured components

        """
        return self.core.normalize(query, prior_turns)
