"""
Model configuration for strand agents - uses framework-agnostic config.

Uses the centralized configuration system from src.config.agent_model_config.
"""

from src.config.agent_model_config import (
    ModelConfig,
    configure,
    get_agent_config,
    get_cache_system_prompt,
    get_global_config,
    show_config_summary,
    update_agent,
)

__all__ = [
    "ModelConfig",
    "get_global_config",
    "configure",
    "get_agent_config",
    "update_agent",
    "show_config_summary",
    "get_cache_system_prompt",
]
