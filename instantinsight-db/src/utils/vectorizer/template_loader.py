"""
Template Loader for LLM Prompts.

Loads and formats YAML-based prompt templates for consistent LLM interactions.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class TemplateLoader:
    """
    Loads and formats YAML prompt templates.

    Provides a clean interface to load structured prompt templates
    and format them with runtime parameters.
    """

    def __init__(self):
        """Initialize the template loader."""
        self.templates_dir = Path(__file__).parent / "templates"
        self._template_cache = {}
        logger.debug("TemplateLoader initialized")

    def load_template(self, template_name: str) -> dict[str, Any]:
        """
        Load a YAML template file.

        Args:
            template_name: Name of the template (without .yaml extension)

        Returns:
            Dictionary containing template metadata and prompt

        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.templates_dir / f"{template_name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        try:
            with open(template_path, encoding="utf-8") as f:
                template = yaml.safe_load(f)

            self._template_cache[template_name] = template
            logger.debug(f"Loaded template: {template_name}")
            return template

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in template {template_name}: {e}") from e

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """
        Load and format a prompt template with parameters.

        Args:
            template_name: Name of the template
            **kwargs: Parameters to substitute in the template

        Returns:
            Formatted prompt string

        """
        template = self.load_template(template_name)

        try:
            return template["prompt"].format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required parameter for template {template_name}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error formatting template {template_name}: {e}") from e

    def get_template_parameters(self, template_name: str) -> list:
        """
        Get the list of required parameters for a template.

        Args:
            template_name: Name of the template

        Returns:
            List of parameter names

        """
        template = self.load_template(template_name)
        return template.get("parameters", [])
