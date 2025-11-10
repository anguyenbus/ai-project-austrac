"""
Analyser filtering for Athena RAG setup.

Manages inclusion and exclusion of analysers based on YAML configuration files.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class AnalyserFilter:
    """Manages analyser inclusion and exclusion logic."""

    def __init__(self, project_root: Path):
        """
        Initialize the analyser filter.

        Args:
            project_root: Path to the project root directory

        """
        self.project_root = project_root
        self.included_analysers = self._load_included_analysers()
        self.excluded_analysers = self._load_excluded_analysers()

    def _load_included_analysers(self) -> dict[str, list[str]]:
        """
        Load analyser inclusion list from artifacts/analysers_to_be_included.yaml file.

        If the file is empty or doesn't exist, all analysers will be processed
        (except those in the exclusion list).

        Returns:
            Dict mapping database names to lists of analyser names

        """
        inclusion_file = (
            self.project_root / "artifacts" / "analysers_to_be_included.yaml"
        )

        if not inclusion_file.exists():
            logger.info(f"📄 Analyser inclusion file not found: {inclusion_file}")
            logger.info("💡 Will process all available analysers (minus exclusions).")
            return {}

        try:
            # Check if file is empty (size 0 or only whitespace)
            file_content = inclusion_file.read_text().strip()
            if not file_content:
                logger.info(f"📄 Analyser inclusion file is empty: {inclusion_file}")
                logger.info(
                    "💡 Will process all available analysers (minus exclusions)."
                )
                return {}

            with open(inclusion_file) as f:
                inclusion_data = yaml.safe_load(f) or {}

            # Check if the loaded content is empty
            if not inclusion_data:
                logger.info(f"📄 No analyser inclusions found in: {inclusion_file}")
                logger.info(
                    "💡 Will process all available analysers (minus exclusions)."
                )
                return {}

            # Handle both formats: direct list or database-mapped lists
            included_analysers = {}

            # If it's a simple list of analysers
            if "analysers" in inclusion_data:
                analyser_list = inclusion_data.get("analysers", [])
                if analyser_list:
                    # Use default database name (will be set by caller)
                    included_analysers["default"] = analyser_list
                    logger.info(f"📋 Loaded {len(analyser_list)} analysers to include")
                    logger.debug(
                        f"   Analysers: {', '.join(analyser_list[:5])}{'...' if len(analyser_list) > 5 else ''}"
                    )
            # If it's mapped by database
            else:
                for db_name, analyser_list in inclusion_data.items():
                    if analyser_list:  # Only include non-empty lists
                        included_analysers[db_name] = analyser_list
                        logger.info(
                            f"📋 Loaded {len(analyser_list)} analysers to include for database '{db_name}'"
                        )

            # If no valid inclusions remain, log that all analysers will be processed
            if not included_analysers:
                logger.info(
                    "💡 No specific analyser inclusions found. Will process all available analysers (minus exclusions)."
                )

            return included_analysers

        except Exception as e:
            logger.error(
                f"❌ Failed to load analyser inclusions from {inclusion_file}: {e}"
            )
            logger.info(
                "💡 Will process all available analysers due to inclusion loading error."
            )
            return {}

    def _load_excluded_analysers(self) -> list[str]:
        """
        Load analyser exclusion list from artifacts/analysers_to_be_excluded.yaml file.

        Returns:
            List of analyser names to exclude

        """
        exclusion_file = (
            self.project_root / "artifacts" / "analysers_to_be_excluded.yaml"
        )

        if not exclusion_file.exists():
            logger.warning(f"⚠️ Analyser exclusion file not found: {exclusion_file}")
            logger.info("💡 No analysers will be excluded.")
            return []

        try:
            with open(exclusion_file) as f:
                exclusion_data = yaml.safe_load(f) or {}

            excluded_analysers = exclusion_data.get("analysers", [])

            if excluded_analysers:
                logger.info(f"📋 Loaded {len(excluded_analysers)} analysers to exclude")
                logger.debug(
                    f"   Excluded analysers: {', '.join(excluded_analysers[:5])}{'...' if len(excluded_analysers) > 5 else ''}"
                )
            else:
                logger.info("📄 No analysers found in exclusion file")

            return excluded_analysers

        except Exception as e:
            logger.error(
                f"❌ Failed to load analyser exclusion list from {exclusion_file}: {e}"
            )
            logger.info("💡 No analysers will be excluded due to loading error.")
            return []

    def filter_schema_tables(
        self,
        schema_data: dict[str, Any],
        database_name: str,
        athena_database: str = None,
    ) -> dict[str, Any]:
        """
        Filter schema data based on analyser inclusion/exclusion lists.

        Logic:
        1. If analysers_to_be_included.yaml exists and has content for this database:
           - Only include analysers from that list
        2. Otherwise:
           - Include all analysers EXCEPT those in analysers_to_be_excluded.yaml

        Args:
            schema_data: Original schema data from extractor
            database_name: Name of the database
            athena_database: Athena database name (for mapping default inclusions)

        Returns:
            Filtered schema data

        """
        original_tables = schema_data.get("tables", {})
        filtered_tables = {}

        # Get included analysers for this database (if any)
        included_analysers = self.included_analysers.get(database_name, [])

        # Also check for "default" key and athena_database key
        if not included_analysers and "default" in self.included_analysers:
            included_analysers = self.included_analysers["default"]

        if (
            not included_analysers
            and athena_database
            and athena_database in self.included_analysers
        ):
            included_analysers = self.included_analysers[athena_database]

        # Determine which tables to process based on inclusion/exclusion logic
        if included_analysers:
            # Case 1: Inclusion list exists - only include specified analysers
            tables_to_process = {
                table_name: table_data
                for table_name, table_data in original_tables.items()
                if table_name in included_analysers
            }

            # Log any analysers that were requested but not found
            for analyser_name in included_analysers:
                if analyser_name not in original_tables:
                    logger.warning(
                        f"⚠️ Analyser '{analyser_name}' not found in {database_name} schema"
                    )

            # No need to apply exclusions when using inclusion list
            filtered_tables = tables_to_process
            excluded_count = 0
        else:
            # Case 2: No inclusion list - use all analysers except excluded ones
            excluded_count = 0
            for table_name, table_data in original_tables.items():
                # Check if the table name matches any excluded analyser
                if table_name in self.excluded_analysers:
                    excluded_count += 1
                    logger.debug(f"   - Excluding analyser: {table_name}")
                else:
                    filtered_tables[table_name] = table_data

        # Log filtering results
        original_count = len(original_tables)
        filtered_count = len(filtered_tables)
        filtered_out_count = original_count - filtered_count

        logger.info(f"🔍 Analyser filtering for {database_name}:")
        logger.info(f"   - Original analysers: {original_count}")

        if included_analysers:
            logger.info(
                f"   - Using inclusion list: {len(included_analysers)} analysers specified"
            )
            logger.info(f"   - Analysers found: {len(filtered_tables)}")
        else:
            logger.info(
                f"   - Using exclusion list: {len(self.excluded_analysers)} analysers to exclude"
            )
            logger.info(f"   - Excluded analysers: {excluded_count}")

        logger.info(f"   - Final analysers: {filtered_count}")
        logger.info(f"   - Total filtered out: {filtered_out_count}")

        if filtered_out_count > 0:
            filtered_out_tables = set(original_tables.keys()) - set(
                filtered_tables.keys()
            )
            # Show a sample of excluded analysers if there are many
            if len(filtered_out_tables) > 10:
                sample = sorted(filtered_out_tables)[:10]
                logger.debug(
                    f"   - Sample of excluded analysers: {', '.join(sample)}... ({len(filtered_out_tables)} total)"
                )
            else:
                logger.debug(
                    f"   - Excluded analysers: {', '.join(sorted(filtered_out_tables))}"
                )

        # Update schema data with filtered tables
        filtered_schema_data = schema_data.copy()
        filtered_schema_data["tables"] = filtered_tables
        filtered_schema_data["table_count"] = len(filtered_tables)

        return filtered_schema_data
