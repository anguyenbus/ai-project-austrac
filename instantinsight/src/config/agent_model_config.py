"""
Framework-agnostic model configuration for agents.

This module provides centralized model configuration for all agents,
independent of any specific agent framework (Agno, Strands, etc.).
"""

import os
from typing import Any, Final


class ModelConfig:
    """Centralized model configuration for all agents."""

    # All agent configurations in one dictionary
    # Easy to modify settings for a single agent
    AGENT_CONFIGS = {
        "SQLFormatter": {
            "model_id": "amazon.nova-micro-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
            "description": "Simple SQL formatting and spacing fixes",
        },
        "SQLSpacingAgent": {
            "model_id": "amazon.nova-micro-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
            "description": "SQL spacing corrections",
        },
        "SchemaValidator": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
            "description": "Table and column existence validation",
        },
        "ClarificationAgent": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 8000,
            "description": "Generate helpful clarification messages",
            "cache_prompt": "default",
        },
        "QueryNormalizer": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 5000,
            "description": "Normalize user questions into canonical form",
        },
        "QueryIntentValidator": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "temperature": 0.1,
            "max_tokens": 3000,
            "description": "Validate query intent and ambiguity",
        },
        "SchemaColumnMapper": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 6000,
            "description": "Map user terms to database columns",
            "cache_prompt": "default",
        },
        "SchemaFilterBuilder": {
            "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 6000,
            "description": "Build SQL filters from user input",
            "cache_prompt": "default",
        },
        "PipelineEvaluationJudge": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 4000,
            "description": "LLM judge for comparing expected and generated SQL",
            "cache_prompt": "default",
        },
        "PipelineEvaluationJudgeNova": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 4000,
            "description": "Primary Nova Pro evaluation judge",
            "cache_prompt": "default",
        },
        "PipelineEvaluationJudgeClaude4": {
            "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 4000,
            "description": "Claude 4 evaluation judge",
            "cache_prompt": "default",
        },
        "SQLGenerator": {
            "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.3,
            "max_tokens": 6000,
            "cache_prompt": "default",
            "description": "Main SQL query generation from natural language",
        },
        "SQLWriterAgent": {
            "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 6000,
            "description": "SQL writing with context and examples",
        },
        "SQLCorrector": {
            "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.2,
            "max_tokens": 4000,
            "description": "Fix and correct SQL syntax errors",
        },
        "SchemaTableSelector": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 8000,
            "cache_prompt": "default",
            "description": "Select appropriate tables for queries",
        },
        "QueryModificationDecider": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "temperature": 0.1,
            "max_tokens": 8000,
            "cache_prompt": "default",
            "description": "Decide if query needs modification",
        },
        "OutputVisualizer": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.1,
            "max_tokens": 8000,
            "cache_prompt": "default",
            "description": "Generate visualization recommendations",
        },
    }

    # Default configuration for agents not explicitly configured
    DEFAULT_CONFIG: Final[dict[str, Any]] = {
        "model_id": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        "temperature": 0.3,
        "max_tokens": 3000,
        "cache_system_prompt": True,
        "description": "Default configuration",
    }

    def __init__(self, custom_configs: dict[str, dict[str, Any]] | None = None):
        """
        Initialize model configuration.

        Args:
            custom_configs: Optional custom configurations to override defaults
                           e.g., {"SQLFormatter": {"model_id": "...", "temperature": 0.2}}

        """
        # Create a copy of default configs
        self.agent_configs = self.AGENT_CONFIGS.copy()

        # Apply custom configurations
        if custom_configs:
            for agent_name, config in custom_configs.items():
                if agent_name in self.agent_configs:
                    # Update existing agent config
                    self.agent_configs[agent_name].update(config)
                else:
                    # Add new agent config
                    self.agent_configs[agent_name] = {**self.DEFAULT_CONFIG, **config}

        # Load environment variable overrides
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load model configurations from environment variables."""
        # Global default model override
        global_model = os.getenv("BEDROCK_MODEL_ID")
        if global_model:
            self.DEFAULT_CONFIG["model_id"] = global_model

        # Agent-specific overrides
        # Format: BEDROCK_MODEL_{AGENT_NAME} for model
        # Format: BEDROCK_TEMP_{AGENT_NAME} for temperature
        # Format: BEDROCK_TOKENS_{AGENT_NAME} for max_tokens
        # Format: BEDROCK_CACHE_{AGENT_NAME} for cache_system_prompt
        for agent_name in self.agent_configs.keys():
            env_prefix = agent_name.upper().replace(" ", "_")

            # Model override
            model_env = f"BEDROCK_MODEL_{env_prefix}"
            model_id = os.getenv(model_env)
            if model_id:
                self.agent_configs[agent_name]["model_id"] = model_id

            # Temperature override
            temp_env = f"BEDROCK_TEMP_{env_prefix}"
            temp = os.getenv(temp_env)
            if temp:
                try:
                    self.agent_configs[agent_name]["temperature"] = float(temp)
                except ValueError:
                    pass

            # Max tokens override
            tokens_env = f"BEDROCK_TOKENS_{env_prefix}"
            tokens = os.getenv(tokens_env)
            if tokens:
                try:
                    self.agent_configs[agent_name]["max_tokens"] = int(tokens)
                except ValueError:
                    pass

            # Cache override
            cache_env = f"BEDROCK_CACHE_{env_prefix}"
            cache_setting = os.getenv(cache_env)
            if cache_setting is not None:
                # Convert string to boolean
                cache_bool = cache_setting.lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                    "enabled",
                )
                self.agent_configs[agent_name]["cache_system_prompt"] = cache_bool

    def get_agent_config(
        self, agent_name: str, aws_region: str | None = None
    ) -> dict[str, Any]:
        """
        Get complete configuration for an agent.

        Args:
            agent_name: Name of the agent
            aws_region: AWS region (defaults to ap-southeast-2)

        Returns:
            Dictionary with model configuration

        """
        # Get agent-specific config or use default
        config = self.agent_configs.get(agent_name, self.DEFAULT_CONFIG).copy()

        # Add AWS region
        config["aws_region"] = aws_region or os.getenv("AWS_REGION", "ap-southeast-2")

        return config

    def get_model_id(self, agent_name: str) -> str:
        """Get just the model ID for an agent."""
        config = self.agent_configs.get(agent_name, self.DEFAULT_CONFIG)
        return config["model_id"]

    def get_temperature(self, agent_name: str) -> float:
        """Get just the temperature for an agent."""
        config = self.agent_configs.get(agent_name, self.DEFAULT_CONFIG)
        return config["temperature"]

    def get_max_tokens(self, agent_name: str) -> int:
        """Get just the max tokens for an agent."""
        config = self.agent_configs.get(agent_name, self.DEFAULT_CONFIG)
        return config["max_tokens"]

    def get_cache_system_prompt(self, agent_name: str) -> bool:
        """Get just the cache_system_prompt setting for an agent."""
        config = self.agent_configs.get(agent_name, self.DEFAULT_CONFIG)
        return config.get(
            "cache_system_prompt", True
        )  # Default to True for Claude models

    def update_agent_config(
        self,
        agent_name: str,
        model_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_system_prompt: bool | None = None,
        description: str | None = None,
    ) -> None:
        """
        Update configuration for a specific agent.

        Args:
            agent_name: Name of the agent
            model_id: New model ID
            temperature: New temperature
            max_tokens: New max tokens
            cache_system_prompt: New cache_system_prompt setting
            description: New description

        """
        if agent_name not in self.agent_configs:
            # Start with default if agent doesn't exist
            self.agent_configs[agent_name] = self.DEFAULT_CONFIG.copy()

        if model_id is not None:
            self.agent_configs[agent_name]["model_id"] = model_id
        if temperature is not None:
            self.agent_configs[agent_name]["temperature"] = temperature
        if max_tokens is not None:
            self.agent_configs[agent_name]["max_tokens"] = max_tokens
        if cache_system_prompt is not None:
            self.agent_configs[agent_name]["cache_system_prompt"] = cache_system_prompt
        if description is not None:
            self.agent_configs[agent_name]["description"] = description

    def list_agents(self) -> dict[str, dict[str, Any]]:
        """List all configured agents and their full configurations."""
        return self.agent_configs.copy()

    def get_summary(self) -> str:
        """Get a summary of all agent configurations."""
        summary = "Agent Model Configurations:\n"
        summary += "=" * 80 + "\n"

        for agent_name, config in self.agent_configs.items():
            model = config["model_id"].split(".")[-1].split("-")[0]  # Extract key part
            summary += f"{agent_name:30} | {model:15} | T={config['temperature']:.1f} | Tokens={config['max_tokens']:4}\n"

        return summary

    def get_cost_estimate(
        self, agent_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Estimate cost for a specific agent's usage.

        Args:
            agent_name: Name of the agent
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD

        """
        # Pricing per 1M tokens (based on AWS Bedrock pricing)
        # These are approximate - update with actual pricing
        cost_per_million = {
            "nova-micro": {"input": 0.035, "output": 0.14},  # Very cost-efficient
            "haiku": {"input": 0.25, "output": 1.25},
            "sonnet": {"input": 3.0, "output": 15.0},
            "opus": {"input": 15.0, "output": 75.0},
        }

        model_id = self.get_model_id(agent_name).lower()

        # Determine pricing tier from model ID
        if "nova-micro" in model_id:
            rates = cost_per_million["nova-micro"]
        elif "haiku" in model_id:
            rates = cost_per_million["haiku"]
        elif "opus" in model_id:
            rates = cost_per_million["opus"]
        else:
            # Default to sonnet pricing for all sonnet variants
            rates = cost_per_million["sonnet"]

        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost


# Global configuration instance
_global_config: ModelConfig | None = None


def get_global_config() -> ModelConfig:
    """Get or create the global model configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ModelConfig()
    return _global_config


def configure(custom_configs: dict[str, dict[str, Any]] | None = None) -> None:
    """
    Configure global model settings.

    Args:
        custom_configs: Optional custom configurations
                       e.g., {"SQLFormatter": {"model_id": "...", "temperature": 0.2}}

    """
    global _global_config
    _global_config = ModelConfig(custom_configs)


# Convenience functions
def get_agent_config(agent_name: str, aws_region: str | None = None) -> dict[str, Any]:
    """Get complete config for an agent using global config."""
    return get_global_config().get_agent_config(agent_name, aws_region)


def get_cache_system_prompt(agent_name: str) -> bool:
    """Get cache_system_prompt setting for an agent using global config."""
    return get_global_config().get_cache_system_prompt(agent_name)


def update_agent(agent_name: str, **kwargs: Any) -> None:
    """Update an agent's configuration."""
    get_global_config().update_agent_config(agent_name, **kwargs)


def show_config_summary() -> None:
    """Print a summary of all agent configurations."""
    print(get_global_config().get_summary())
