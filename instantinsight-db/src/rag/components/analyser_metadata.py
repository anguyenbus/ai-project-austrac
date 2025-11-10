"""
Simple Analyser metadata handler for the RAG system.

Reads analyser definitions from YAML and enriches table metadata.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class AnalyserMetadata:
    """Simple handler for Analyser metadata from YAML configuration."""

    def __init__(self, config_path: str = "config/analyser_definitions.yaml"):
        """Initialize with path to YAML configuration."""
        self.config_path = Path(config_path)
        self.definitions = self._load_definitions()
        self.default_config = self.definitions.get("_default", {})

    def _load_definitions(self) -> dict:
        """Load Analyser definitions from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Analyser definitions not found at {self.config_path}")
            return {}

        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load analyser definitions: {e}")
            return {}

    def get_all_analysers(self) -> dict[str, Any]:
        """
        Get all analyser definitions (excluding _default).

        Returns:
            Dictionary of analyser definitions keyed by table name

        """
        return {k: v for k, v in self.definitions.items() if k != "_default"}

    def get_metadata(self, table_name: str) -> dict[str, Any]:
        """
        Get enriched metadata for a table.

        Args:
            table_name: Name of the Analyser table

        Returns:
            Dictionary with metadata including unique key, grain, and warnings

        """
        # Start with defaults
        metadata = {
            "table_name": table_name,
            "is_analyser": True,
            "safe_to_join_with": [],
            "warnings": ["This is a Analyser - avoid joins unless verified safe"],
        }

        # Apply default configuration
        metadata.update(self.default_config)

        # Apply specific table configuration if exists
        if table_name in self.definitions:
            table_config = self.definitions[table_name]
            metadata.update(table_config)

            # Format unique key as readable string
            if "unique_key" in metadata and metadata["unique_key"]:
                metadata["unique_key_str"] = " + ".join(metadata["unique_key"])

        return metadata

    def is_join_safe(self, table1: str, table2: str) -> tuple[bool, str | None]:
        """
        Check if joining two tables is safe.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            Tuple of (is_safe, join_key) where join_key is the column to join on

        """
        # Get metadata for first table
        metadata1 = self.get_metadata(table1)

        # Check if table2 is in the safe join list
        for safe_join in metadata1.get("safe_to_join_with", []):
            if isinstance(safe_join, dict) and safe_join.get("table") == table2:
                return True, safe_join.get("join_key")
            elif isinstance(safe_join, str) and safe_join == table2:
                return True, None

        # Check reverse direction
        metadata2 = self.get_metadata(table2)
        for safe_join in metadata2.get("safe_to_join_with", []):
            if isinstance(safe_join, dict) and safe_join.get("table") == table1:
                return True, safe_join.get("join_key")
            elif isinstance(safe_join, str) and safe_join == table1:
                return True, None

        return False, None

    def get_join_warning(self, table1: str, table2: str) -> str:
        """
        Get a warning message about joining two tables.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            Warning message

        """
        is_safe, join_key = self.is_join_safe(table1, table2)

        if is_safe:
            if join_key:
                return (
                    f"✅ Safe join verified between {table1} and {table2} on {join_key}"
                )
            else:
                return f"✅ Safe join verified between {table1} and {table2}"
        else:
            metadata1 = self.get_metadata(table1)
            metadata2 = self.get_metadata(table2)

            grain1 = metadata1.get("grain", "unknown grain")
            grain2 = metadata2.get("grain", "unknown grain")

            return (
                f"⚠️ DANGEROUS: Joining {table1} ({grain1}) with {table2} ({grain2}) "
                f"is not verified safe. This may cause double-counting, cartesian products, "
                f"or incorrect results. Consider using a single Analyser or creating a "
                f"combined dataset at the model level."
            )

    def enrich_ddl_metadata(self, table_name: str, existing_metadata: dict) -> dict:
        """
        Enrich existing DDL metadata with Analyser-specific information.

        Args:
            table_name: Table name
            existing_metadata: Existing metadata from DDL

        Returns:
            Enriched metadata

        """
        # Get Analyser metadata
        analyser_meta = self.get_metadata(table_name)

        # Merge with existing metadata
        enriched = existing_metadata.copy()
        enriched.update(
            {
                "is_analyser": True,
                "unique_key": analyser_meta.get("unique_key", []),
                "grain": analyser_meta.get("grain", "Row-level data"),
                "safe_joins": analyser_meta.get("safe_to_join_with", []),
                "analyser_warnings": analyser_meta.get("warnings", []),
            }
        )

        # Add a natural language description for embedding
        enriched["analyser_description"] = self._create_description(
            table_name, analyser_meta
        )

        return enriched

    def _create_description(self, table_name: str, metadata: dict) -> str:
        """Create a natural language description of the Analyser."""
        parts = [
            f"{table_name} is a Analyser.",
        ]

        if metadata.get("grain"):
            parts.append(f"Grain: {metadata['grain']}.")

        if metadata.get("unique_key"):
            key_str = " + ".join(metadata["unique_key"])
            parts.append(f"Unique key: {key_str}.")

        if metadata.get("safe_to_join_with"):
            safe_tables = [
                j["table"] if isinstance(j, dict) else j
                for j in metadata["safe_to_join_with"]
            ]
            parts.append(f"Safe to join with: {', '.join(safe_tables)}.")
        else:
            parts.append("Should not be joined with other Analysers.")

        if metadata.get("warnings"):
            parts.append(f"Warnings: {' '.join(metadata['warnings'])}")

        return " ".join(parts)
