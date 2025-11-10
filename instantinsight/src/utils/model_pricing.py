"""
Model pricing configuration for AWS Bedrock models.

Prices are in USD per 1M tokens based on AWS Bedrock pricing.
Langfuse will use its own pricing database if available, this is fallback.
"""


def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing for a Bedrock model, with fallback to default pricing.

    Args:
        model_id: Model identifier string

    Returns:
        Dict with input_cost_per_token and output_cost_per_token

    """
    return MODEL_PRICING.get(
        model_id,
        {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "name": "Default (Claude 3.5 Sonnet v2 pricing)",
        },
    )


# AWS Bedrock model pricing (USD per 1M tokens)
# Based on models with "Access granted" status
MODEL_PRICING = {
    # Claude Sonnet 4 (Latest)
    "anthropic.claude-sonnet-4-20250514-v1:0": {
        "input_cost_per_token": 0.000003,  # $3 per 1M tokens
        "output_cost_per_token": 0.000015,  # $15 per 1M tokens
        "name": "Claude Sonnet 4",
    },
    # Claude 3.5 Sonnet v2 (Current production)
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_cost_per_token": 0.000003,  # $3 per 1M tokens
        "output_cost_per_token": 0.000015,  # $15 per 1M tokens
        "name": "Claude 3.5 Sonnet v2",
    },
    # Claude 3 Haiku (Fast and cheap)
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_cost_per_token": 0.00000025,  # $0.25 per 1M tokens
        "output_cost_per_token": 0.00000125,  # $1.25 per 1M tokens
        "name": "Claude 3 Haiku",
    },
    # Amazon Nova Lite (Cheapest option)
    "amazon.nova-lite-v1:0": {
        "input_cost_per_token": 0.00000006,  # $0.06 per 1M tokens
        "output_cost_per_token": 0.00000024,  # $0.24 per 1M tokens (150 output chars)
        "name": "Nova Lite",
    },
    # Amazon Nova Micro
    "amazon.nova-micro-v1:0": {
        "input_cost_per_token": 0.000000035,  # $0.035 per 1M tokens
        "output_cost_per_token": 0.00000014,  # $0.14 per 1M tokens (150 output chars)
        "name": "Nova Micro",
    },
    # Amazon Nova Pro
    "amazon.nova-pro-v1:0": {
        "input_cost_per_token": 0.0000008,  # $0.80 per 1M tokens
        "output_cost_per_token": 0.0000032,  # $3.20 per 1M tokens
        "name": "Nova Pro",
    },
    # Amazon Titan Text Embeddings V2
    "amazon.titan-embed-text-v2:0": {
        "input_cost_per_token": 0.00000002,  # $0.02 per 1M tokens
        "output_cost_per_token": 0.0,  # Embeddings don't have output
        "name": "Titan Text Embeddings V2",
    },
    # Amazon Titan Text G1 - Express
    "amazon.titan-text-express-v1": {
        "input_cost_per_token": 0.0000002,  # $0.20 per 1M tokens
        "output_cost_per_token": 0.0000006,  # $0.60 per 1M tokens
        "name": "Titan Text G1 Express",
    },
    # Amazon Titan Text G1 - Lite
    "amazon.titan-text-lite-v1": {
        "input_cost_per_token": 0.00000015,  # $0.15 per 1M tokens
        "output_cost_per_token": 0.0000002,  # $0.20 per 1M tokens
        "name": "Titan Text G1 Lite",
    },
}
