"""
Configuration loader for Athena RAG setup.

Handles loading and merging configurations from various sources.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class ConfigLoader:
    """Manages configuration loading and merging for Athena RAG setup."""

    def __init__(self, config_file: Path | None = None):
        """
        Initialize the configuration loader.

        Args:
            config_file: Optional path to configuration override file

        """
        self.config_override = {}

        if config_file and config_file.exists():
            self.config_override = self._load_config_file(config_file)

    def _load_config_file(self, config_file: Path) -> dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            config_file: Path to the configuration file

        Returns:
            Dict containing configuration data

        """
        try:
            with open(config_file) as f:
                config = json.load(f)
            logger.info(f"✓ Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise ValueError(f"Invalid configuration file: {e}") from e

    def get_athena_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        Get merged Athena configuration.

        Args:
            base_config: Base Athena configuration

        Returns:
            Merged configuration dictionary

        """
        return {**base_config, **self.config_override.get("athena", {})}

    def get_rag_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        Get merged RAG configuration.

        Args:
            base_config: Base RAG configuration

        Returns:
            Merged configuration dictionary

        """
        return {**base_config, **self.config_override.get("rag", {})}

    def get_aws_config(self) -> dict[str, Any]:
        """
        Get AWS-specific configuration.

        Returns:
            AWS configuration dictionary

        """
        return {
            "aws_profile": self.config_override.get("aws_profile"),
            "aws_region": self.config_override.get("aws_region"),
        }
