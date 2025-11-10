"""Test model pricing functionality."""

from src.utils.model_pricing import get_model_pricing


def test_get_model_pricing_known_model():
    """Test getting pricing for a known model."""
    pricing = get_model_pricing("anthropic.claude-3-haiku-20240307-v1:0")

    assert pricing["input_cost_per_token"] == 0.00000025
    assert pricing["output_cost_per_token"] == 0.00000125
    assert pricing["name"] == "Claude 3 Haiku"


def test_get_model_pricing_unknown_model():
    """Test getting pricing for an unknown model falls back to default."""
    pricing = get_model_pricing("unknown-model-id")

    assert pricing["input_cost_per_token"] == 0.000003
    assert pricing["output_cost_per_token"] == 0.000015
    assert pricing["name"] == "Default (Claude 3.5 Sonnet v2 pricing)"


def test_get_model_pricing_nova_lite():
    """Test getting pricing for Amazon Nova Lite."""
    pricing = get_model_pricing("amazon.nova-lite-v1:0")

    assert pricing["input_cost_per_token"] == 0.00000006
    assert pricing["output_cost_per_token"] == 0.00000024
    assert pricing["name"] == "Nova Lite"
