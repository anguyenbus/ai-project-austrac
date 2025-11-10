"""
ModificationDecisionAgent - Intelligent decision making for visualization modifications using Strand framework.

This agent analyzes user modification requests and determines whether to:
1. Modify only the Plotly schema (visual changes) - calls VisualizationAgent
2. Modify the SQL query (data changes) - runs full Pipeline
3. Modify both - runs full Pipeline

Migrated to use Strand framework for enhanced performance and reliability.
"""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.prompts import Prompts
from src.agents.prompt_builders.query.modification_decider import (
    ModificationDecisionPrompts,
)
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ..model_config import get_agent_config


class ModificationDecision(BaseModel):
    """Structured response for modification decision."""

    modify_sql: str | None = Field(
        default=None,
        description="SQL modification instruction if data changes are needed, None if no SQL changes",
    )
    modify_schema: str | None = Field(
        default=None,
        description="Schema modification instruction if only visual changes are needed, None if no schema-only changes",
    )
    reasoning: str = Field(
        description="Explanation of the decision and what type of modification is needed"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the decision (0.0 to 1.0)"
    )
    modification_type: str = Field(
        description="Type of modification: 'schema_only', 'sql_only', 'both', or 'unclear'"
    )
    is_simple_sql_modification: bool = Field(
        default=False,
        description="Whether this is a simple SQL modification that can be handled directly without full pipeline",
    )
    modified_sql: str | None = Field(
        default=None,
        description="The directly modified SQL query for simple modifications",
    )
    is_simple_visual_modification: bool = Field(
        default=False,
        description="Whether this is a simple visual modification that can be handled by directly updating the Plotly schema",
    )
    modified_plotly_config: dict[str, Any] | None = Field(
        default=None,
        description='Direct Plotly configuration updates for simple visual modifications (e.g., {"layout": {"title": "New Title"}})',
    )


class ModificationDecisionCore:
    """Core Strand-based modification decision implementation."""

    def __init__(
        self,
        model_id: str = None,
        aws_region: str = None,
        debug_mode: bool = False,
        session_id: str = None,
    ):
        """
        Initialize ModificationDecisionCore with Strand Agent.

        Args:
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            debug_mode: Enable debug logging
            session_id: Session identifier for tracking

        """
        # Get configuration from centralized config
        agent_config = get_agent_config("QueryModificationDecider", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]
        self.debug_mode = debug_mode

        # Model configuration for Strand
        # Note: Bedrock region is set via AWS_REGION env var or AWS credentials
        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            streaming=False,
            cache_prompt=agent_config.get("cache_prompt"),
        )

        self._cache_prompt_type = agent_config.get("cache_prompt")

        # Load cached system prompt optimized for prompt caching
        base_instructions = Prompts.QUERY_MODIFICATION_DECIDER

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        # Initialize Strand Agent
        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

        logger.info("✓ ModificationDecisionCore (strand) initialized")

    @observe(as_type="generation")
    def analyze_modification_request(
        self,
        historical_context: str,
        user_message: str,
        current_sql: str,
        current_plotly_schema: dict[str, Any],
        user_id: str | None = None,
    ) -> ModificationDecision:
        """
        Analyze user modification request to determine processing approach using Strand Agent.

        Args:
            historical_context: Full conversation history
            user_message: Current user modification request
            current_sql: Current SQL query generating the data
            current_plotly_schema: Current Plotly chart schema
            user_id: Identifier used to scope agent memory and history

        Returns:
            ModificationDecision with instructions for processing

        """
        try:
            # Build analysis prompt using prompt builder
            prompt = ModificationDecisionPrompts.build_decision_prompt(
                historical_context, user_message, current_sql, current_plotly_schema
            )

            # Reset usage
            self._usage_container["last_usage"] = None

            # Call Strand Agent with structured output
            llm_result = self.agent.structured_output(ModificationDecision, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "ModificationDecider",
                langfuse_context,
            )

            if self._cache_prompt_type:
                log_prompt_cache_status("ModificationDecider", self._usage_container)

            logger.info(
                f"Decision: {llm_result.modification_type} (confidence: {llm_result.confidence})"
            )
            logger.debug(f"Reasoning: {llm_result.reasoning}")
            return llm_result

        except Exception as e:
            logger.error(f"Strand modification decision analysis failed: {e}")
            # Fallback to safe default (full pipeline)
            return ModificationDecision(
                modify_sql="Process full pipeline due to analysis error",
                modify_schema=None,
                reasoning=f"Analysis failed ({str(e)}), defaulting to full pipeline for safety",
                confidence=0.1,
                modification_type="sql_only",
            )

    @observe(as_type="generation")
    def decide_processing_approach(
        self,
        historical_context: str,
        user_message: str,
        current_sql: str,
        current_plotly_schema: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Decide how to process a modification request.

        Returns:
            Dict with processing instructions and metadata

        """
        try:
            decision = self.analyze_modification_request(
                historical_context=historical_context,
                user_message=user_message,
                current_sql=current_sql,
                current_plotly_schema=current_plotly_schema,
                user_id=user_id,
            )

            result = {
                "modify_sql": decision.modify_sql,
                "modify_schema": decision.modify_schema,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
                "modification_type": decision.modification_type,
                "is_simple_sql_modification": decision.is_simple_sql_modification,
                "modified_sql": decision.modified_sql,
                "is_simple_visual_modification": decision.is_simple_visual_modification,
                "modified_plotly_config": decision.modified_plotly_config,
                "success": True,
            }

            # Add processing recommendation
            if (
                decision.is_simple_visual_modification
                and decision.modified_plotly_config
            ):
                result["processing_method"] = "simple_visual_modification"
                result["explanation"] = (
                    "Simple visual update - modifying Plotly directly"
                )
            elif decision.is_simple_sql_modification and decision.modified_sql:
                result["processing_method"] = "simple_sql_modification"
                result["explanation"] = "Simple SQL modification - executing directly"
            elif decision.modify_sql and not decision.modify_schema:
                result["processing_method"] = "full_pipeline"
                result["explanation"] = "Complex data changes - running full pipeline"
            elif decision.modify_schema and not decision.modify_sql:
                result["processing_method"] = "visualization_agent_only"
                result["explanation"] = "Visual changes only - using VisualizationAgent"
            elif decision.modify_sql and decision.modify_schema:
                result["processing_method"] = "full_pipeline"
                result["explanation"] = (
                    "Both data and visual changes - running full pipeline"
                )
            else:
                result["processing_method"] = "full_pipeline"
                result["explanation"] = "Unclear request - defaulting to full pipeline"

            return result

        except Exception as e:
            logger.error(f"Decision processing failed: {e}")
            return {
                "modify_sql": "Process with full pipeline due to error",
                "modify_schema": None,
                "reasoning": f"Error in decision processing: {str(e)}",
                "confidence": 0.1,
                "modification_type": "error",
                "processing_method": "full_pipeline",
                "explanation": "Error occurred - defaulting to full pipeline",
                "success": False,
                "error": str(e),
            }

    def _create_error_response(self, error_msg: str) -> dict[str, Any]:
        """Create standardized error response."""
        return {
            "modify_sql": "Process full pipeline due to error",
            "modify_schema": None,
            "reasoning": f"Error: {error_msg}",
            "confidence": 0.1,
            "modification_type": "error",
            "processing_method": "full_pipeline",
            "explanation": "Error occurred - defaulting to full pipeline",
            "success": False,
            "error": error_msg,
        }


class ModificationDecisionAgent:
    """Compatibility wrapper maintaining original API."""

    def __init__(self, aws_region: str = None, model_id: str = None):
        """
        Initialize with Strand implementation.

        Args:
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM

        """
        agent_config = get_agent_config("QueryModificationDecider", aws_region)

        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]

        # Initialize core Strand implementation
        self.core = ModificationDecisionCore(
            model_id=self.model_id, aws_region=self.aws_region, debug_mode=False
        )

        logger.info("✓ ModificationDecisionAgent initialized")

    def analyze_modification_request(
        self,
        historical_context: str,
        user_message: str,
        current_sql: str,
        current_plotly_schema: dict[str, Any],
        user_id: str | None = None,
    ) -> ModificationDecision:
        """
        Analyze user modification request to determine processing approach.

        Delegates to core Strand implementation.
        """
        return self.core.analyze_modification_request(
            historical_context=historical_context,
            user_message=user_message,
            current_sql=current_sql,
            current_plotly_schema=current_plotly_schema,
            user_id=user_id,
        )

    def decide_processing_approach(
        self,
        historical_context: str,
        user_message: str,
        current_sql: str,
        current_plotly_schema: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Decide how to process a modification request.

        Delegates to core Strand implementation.
        """
        return self.core.decide_processing_approach(
            historical_context=historical_context,
            user_message=user_message,
            current_sql=current_sql,
            current_plotly_schema=current_plotly_schema,
            user_id=user_id,
        )
