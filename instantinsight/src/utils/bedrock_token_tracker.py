"""
Bedrock Token Usage Tracker.

Tracks token usage for AWS Bedrock API calls via instructor.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class TokenUsage:
    """Represents token usage for a single API call."""

    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BedrockTokenTracker:
    """
    Tracks token usage for Bedrock API calls via instructor.

    Usage:
        tracker = BedrockTokenTracker()

        # For instructor-wrapped calls
        response = client.chat.completions.create(...)
        usage = tracker.track_instructor_usage(response, model_id)

        # Get statistics
        stats = tracker.get_usage_stats()
    """

    def __init__(self):
        """Initialize token tracker with empty history."""
        self.usage_history: list[TokenUsage] = []

    def track_instructor_usage(
        self, response: Any, model_id: str, metadata: dict[str, Any] | None = None
    ) -> TokenUsage:
        """
        Track token usage from an instructor-wrapped Bedrock call.

        Args:
            response: Response from instructor client.chat.completions.create()
            model_id: Model identifier used
            metadata: Optional metadata to attach to the usage record

        Returns:
            TokenUsage object

        """
        try:
            input_tokens = 0
            output_tokens = 0

            # Instructor responses typically have a _raw_response attribute
            if hasattr(response, "_raw_response"):
                raw_response = response._raw_response

                # Try multiple formats for token extraction
                if "usage" in raw_response and isinstance(raw_response["usage"], dict):
                    # Usage object exists - try both snake_case and camelCase
                    usage_data = raw_response["usage"]

                    # Check for input tokens in order of preference
                    for key in ["input_tokens", "inputTokens", "inputTokenCount"]:
                        if key in usage_data:
                            input_tokens = usage_data[key]
                            break

                    # Check for output tokens in order of preference
                    for key in ["output_tokens", "outputTokens", "outputTokenCount"]:
                        if key in usage_data:
                            output_tokens = usage_data[key]
                            break
                elif "inputTokens" in raw_response and "outputTokens" in raw_response:
                    # Direct camelCase format at root level
                    input_tokens = raw_response.get("inputTokens", 0)
                    output_tokens = raw_response.get("outputTokens", 0)
                elif (
                    "inputTokenCount" in raw_response
                    and "outputTokenCount" in raw_response
                ):
                    # Alternative format at root level
                    input_tokens = raw_response.get("inputTokenCount", 0)
                    output_tokens = raw_response.get("outputTokenCount", 0)
                else:
                    # Check other possible nested locations
                    for usage_key in [
                        "token_usage",
                        "metrics",
                        "amazon-bedrock-invocationMetrics",
                    ]:
                        if usage_key in raw_response and isinstance(
                            raw_response[usage_key], dict
                        ):
                            usage_data = raw_response[usage_key]

                            # Check for input tokens in order of preference
                            for key in [
                                "input_tokens",
                                "inputTokens",
                                "inputTokenCount",
                            ]:
                                if key in usage_data:
                                    input_tokens = usage_data[key]
                                    break

                            # Check for output tokens in order of preference
                            for key in [
                                "output_tokens",
                                "outputTokens",
                                "outputTokenCount",
                            ]:
                                if key in usage_data:
                                    output_tokens = usage_data[key]
                                    break

                            # If we found tokens, break out of the outer loop
                            if input_tokens > 0 or output_tokens > 0:
                                break

            usage = TokenUsage(
                timestamp=datetime.now(),
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                metadata=metadata or {},
            )

            self.usage_history.append(usage)

            # Add debug info about token extraction
            if input_tokens == 0 and output_tokens == 0:
                logger.warning(
                    f"No tokens extracted from instructor response. "
                    f"Raw response keys: {list(raw_response.keys()) if hasattr(response, '_raw_response') and isinstance(response._raw_response, dict) else 'N/A'}"
                )

            logger.debug(
                f"Tracked instructor usage: {input_tokens} input, {output_tokens} output tokens"
            )

            return usage

        except Exception as e:
            logger.error(f"Error tracking instructor usage: {e}")
            return TokenUsage(
                timestamp=datetime.now(),
                model_id=model_id,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                metadata=metadata or {},
            )

    def get_usage_stats(self, model_id: str | None = None) -> dict[str, Any]:
        """
        Get usage statistics.

        Args:
            model_id: Optional model ID to filter stats

        Returns:
            Dictionary with usage statistics

        """
        filtered_usage = self.usage_history
        if model_id:
            filtered_usage = [u for u in self.usage_history if u.model_id == model_id]

        if not filtered_usage:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            }

        total_input = sum(u.input_tokens for u in filtered_usage)
        total_output = sum(u.output_tokens for u in filtered_usage)

        return {
            "total_calls": len(filtered_usage),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "average_tokens_per_call": (total_input + total_output)
            / len(filtered_usage),
        }

    def reset_tracking(self):
        """Reset all tracked usage data."""
        self.usage_history.clear()
        logger.info("Token tracking data reset")
