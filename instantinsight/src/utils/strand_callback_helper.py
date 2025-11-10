"""Helper utilities for Strands agent usage tracking and Langfuse integration."""

from collections.abc import Callable
from typing import Any

from loguru import logger

from .cost_accumulator import cost_accumulator
from .model_pricing import get_model_pricing


def create_usage_callback() -> tuple[Callable, dict[str, Any]]:
    """
    Create callback handler for Strands agent usage tracking.

    Returns:
        Tuple of (callback_function, usage_container_dict)

    """
    usage_container = {"last_usage": None}

    def usage_callback(**kwargs: Any) -> None:
        """
        Capture usage from Strands agent.

        Strands callbacks use **kwargs signature like PrintingCallbackHandler.

        Args:
            **kwargs: Callback event data including:
                - event: Event type (e.g., "on_llm_start", "on_llm_end")
                - usage: Token usage data (on LLM completion)
                - data: Text content
                - complete: Whether response is complete

        """
        # Check if usage data is present (typically on LLM completion)
        if "usage" in kwargs:
            usage_container["last_usage"] = kwargs["usage"]
            logger.debug(f"📊 Captured usage from Strands agent: {kwargs['usage']}")

    return usage_callback, usage_container


def update_langfuse_with_usage(
    usage_container: dict[str, Any],
    model_id: str,
    agent_name: str,
    langfuse_context,
) -> None:
    """
    Update Langfuse with usage metrics from Strands agent.

    Args:
        usage_container: Container with last_usage data
        model_id: Model identifier for cost calculation
        agent_name: Name of the agent for logging
        langfuse_context: Langfuse context module

    """
    try:
        if not langfuse_context or not usage_container.get("last_usage"):
            logger.debug(f"⚠️ {agent_name}: No usage data to update Langfuse")
            return

        usage = usage_container["last_usage"]

        # Extract token counts from Strands usage format
        # Strands usage format: {"input_tokens": X, "output_tokens": Y}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        logger.info(
            f"📊 {agent_name}: Extracted tokens - input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
        )

        # Get model-specific pricing
        pricing = get_model_pricing(model_id)

        input_cost = input_tokens * pricing["input_cost_per_token"]
        output_cost = output_tokens * pricing["output_cost_per_token"]

        # Add to cost accumulator for total tracking
        cost_accumulator.add_agent_cost(
            agent_name=agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            model_id=model_id,
        )

        # Store trace ID for later use (when we're outside the @observe context)
        if not cost_accumulator.trace_id and langfuse_context:
            try:
                trace_id = langfuse_context.get_current_trace_id()
                if trace_id:
                    cost_accumulator.set_trace_id(trace_id)
                    logger.debug(
                        f"📍 Stored trace ID for cost accumulation: {trace_id}"
                    )
            except Exception as e:
                logger.debug(f"Could not get trace ID: {e}")

        # Prepare usage_details in Langfuse format
        usage_details = {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "unit": "TOKENS",
        }

        # Update Langfuse observation
        logger.debug(
            f"🔄 {agent_name}: Calling langfuse_context.update_current_observation..."
        )
        logger.debug(f"   Model: {model_id}")
        logger.debug(f"   Usage details: {usage_details}")
        logger.debug(f"   Calculated cost: ${input_cost + output_cost:.6f}")

        try:
            # Try the decorator approach first - test different parameter names
            try:
                langfuse_context.update_current_observation(
                    model=model_id, usage=usage_details
                )
                logger.debug(
                    f"✅ {agent_name}: Langfuse usage update completed via decorator context"
                )
            except Exception:
                # Try the original format
                langfuse_context.update_current_observation(
                    model=model_id, usage_details=usage_details
                )
                logger.debug(
                    f"✅ {agent_name}: Langfuse usage_details update completed via decorator context"
                )

            # Add cost information as scores (visible in dashboard)
            try:
                from langfuse import Langfuse

                client = Langfuse()

                # Get current observation ID and trace ID for cost tracking
                obs_id = langfuse_context.get_current_observation_id()
                trace_id = langfuse_context.get_current_trace_id()

                if obs_id and trace_id:
                    # Add cost breakdown as individual scores for better visibility
                    client.score(
                        trace_id=trace_id,
                        observation_id=obs_id,
                        name=f"{agent_name.lower()}_cost_breakdown",
                        value=input_cost + output_cost,
                        comment=f"Input: ${input_cost:.6f} ({input_tokens} tokens) + Output: ${output_cost:.6f} ({output_tokens} tokens) | Model: {model_id}",
                    )

                    logger.debug(
                        f"✅ {agent_name}: Cost tracking added as scores - Total: ${input_cost + output_cost:.6f}"
                    )
                else:
                    logger.debug(f"⚠️ {agent_name}: Could not get IDs for cost tracking")

            except Exception as cost_error:
                logger.debug(
                    f"⚠️ {agent_name}: Cost tracking as scores failed: {cost_error}"
                )
                # Continue - cost tracking failure shouldn't break the application

            logger.debug(
                f"✅ {agent_name}: Langfuse update completed successfully via decorator context"
            )
        except Exception as update_error:
            logger.warning(f"⚠️ {agent_name}: Decorator update failed: {update_error}")
            # Try direct client approach as fallback
            try:
                from langfuse import Langfuse

                client = Langfuse()

                # Get current observation ID if available
                obs_id = None
                try:
                    obs_id = langfuse_context.get_current_observation_id()
                except Exception as e:
                    logger.error(f"🚨 {agent_name}: Could not get observation ID: {e}")

                if obs_id:
                    # Update existing observation
                    client.score(
                        trace_id=langfuse_context.get_current_trace_id(),
                        observation_id=obs_id,
                        name="token_usage",
                        value=total_tokens,
                        comment=f"Tokens: {input_tokens}+{output_tokens}={total_tokens}, Cost: ${input_cost + output_cost:.6f}, Model: {model_id}",
                    )
                    logger.info(
                        f"✅ {agent_name}: Langfuse updated via direct client score method"
                    )
                else:
                    logger.warning(
                        f"⚠️ {agent_name}: Could not get observation ID for direct update"
                    )

            except Exception as direct_error:
                logger.error(
                    f"🚨 {agent_name}: Both decorator and direct methods failed: {direct_error}"
                )
                # Continue execution - token tracking failure shouldn't break the application

        total_cost = input_cost + output_cost
        logger.info(
            f"✅ {agent_name}: Updated Langfuse - tokens: {input_tokens}+{output_tokens}={total_tokens}, cost: ${total_cost:.6f}, model: {model_id}"
        )
    except Exception as e:
        logger.error(
            f"🚨 {agent_name}: Could not update Langfuse with token usage: {e}"
        )
        logger.error(f"   Exception type: {type(e)}")
        logger.error(f"   Exception args: {e.args}")
        import traceback

        logger.error(f"   Traceback: {traceback.format_exc()}")


def log_prompt_cache_status(agent_name: str, usage_container: dict[str, Any]) -> None:
    """
    Log prompt cache hit/miss status for debugging.

    Args:
        agent_name: Name of the agent for logging
        usage_container: Container with last_usage data

    """
    try:
        if not usage_container.get("last_usage"):
            return

        usage = usage_container["last_usage"]

        # Check for cache-related fields in usage data
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)

        if cache_creation_tokens > 0:
            logger.info(
                f"💾 {agent_name}: Prompt cache CREATED - {cache_creation_tokens} tokens written to cache"
            )

        if cache_read_tokens > 0:
            logger.info(
                f"⚡ {agent_name}: Prompt cache HIT - {cache_read_tokens} tokens read from cache"
            )

        if cache_creation_tokens == 0 and cache_read_tokens == 0:
            logger.debug(f"📝 {agent_name}: No prompt cache activity")

    except Exception as e:
        logger.debug(f"⚠️ {agent_name}: Could not log cache status: {e}")
