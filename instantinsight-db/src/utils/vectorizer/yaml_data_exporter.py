"""
YAML Data Export Module.

Handles export and management of validated SQL examples in YAML format.
Extracted from the monolithic AthenaSchemaVectorizer for better maintainability.
"""

from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger


class YAMLDataExporter:
    """
    Exports validated SQL examples to YAML format for training data.

    This class handles:
    - Individual table example exports
    - Bulk export of all validated examples
    - Appending to main training YAML files
    - Proper formatting according to sql_examples.yaml structure
    """

    def __init__(self, training_data_dir: Path = None):
        """
        Initialize the YAML data exporter.

        Args:
            training_data_dir: Directory for training data (defaults to src/training/data)

        """
        if training_data_dir is None:
            # Default to src/training/data relative to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            self.training_data_dir = project_root / "src" / "training" / "data"
        else:
            self.training_data_dir = training_data_dir

        # Ensure directory exists
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"YAMLDataExporter initialized with directory: {self.training_data_dir}"
        )

    def export_table_examples(
        self, examples: list[dict[str, str]], database_name: str, table_name: str
    ) -> Path:
        """
        Export SQL examples for a specific table to YAML format.

        Args:
            examples: List of SQL examples with question/sql pairs
            database_name: Name of the database
            table_name: Name of the table

        Returns:
            Path to the created YAML file

        """
        if not examples:
            logger.warning(f"No examples to export for {table_name}")
            return None

        filename = f"sql_examples_{database_name}_{table_name}.yaml"
        output_file = self.training_data_dir / filename

        # Format examples according to sql_examples.yaml structure
        yaml_data = {"examples": {"queries": []}}

        for example in examples:
            # Ensure we have the required fields
            if "question" in example and "sql" in example:
                yaml_example = {"question": example["question"], "sql": example["sql"]}
                yaml_data["examples"]["queries"].append(yaml_example)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Add header comment
                f.write(f"# SQL Examples for {database_name}.{table_name}\n")
                f.write(
                    f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"# Contains {len(examples)} validated examples\n\n")

                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            logger.info(
                f"💾 Exported {len(examples)} examples for {table_name} to: {output_file}"
            )
            return output_file

        except Exception as e:
            logger.error(f"Failed to export examples for {table_name}: {e}")
            return None

    def export_all_validated_examples(
        self, validated_examples: list[dict[str, str]]
    ) -> Path:
        """
        Export all validated examples to a timestamped YAML file.

        Args:
            validated_examples: List of all validated SQL examples

        Returns:
            Path to the created YAML file

        """
        if not validated_examples:
            logger.warning("No validated examples to export")
            return None

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.training_data_dir / f"validated_athena_examples_{timestamp}.yaml"
        )

        # Format examples according to sql_examples.yaml structure
        yaml_data = {"examples": {"queries": []}}

        for example in validated_examples:
            if "question" in example and "sql" in example:
                yaml_example = {"question": example["question"], "sql": example["sql"]}
                yaml_data["examples"]["queries"].append(yaml_example)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Add comprehensive header
                f.write("# ATHENA-VALIDATED SQL Examples\n")
                f.write(
                    f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(
                    f"# Contains {len(validated_examples)} ATHENA-VALIDATED examples\n"
                )
                f.write(
                    "# These examples have been tested against AWS Athena and confirmed to execute successfully\n\n"
                )

                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            logger.info(
                f"💾 EXPORTED {len(validated_examples)} validated examples to: {output_file}"
            )

            # Also append to the main sql_examples.yaml file
            self._append_to_main_yaml_file(validated_examples)

            return output_file

        except Exception as e:
            logger.error(f"Failed to export validated examples: {e}")
            return None

    def _append_to_main_yaml_file(self, validated_examples: list[dict[str, str]]):
        """Append validated examples to the main sql_examples.yaml file."""
        try:
            main_yaml_file = self.training_data_dir / "sql_examples.yaml"

            if not main_yaml_file.exists():
                logger.warning(f"Main YAML file not found: {main_yaml_file}")
                return

            # Load existing data
            with open(main_yaml_file, encoding="utf-8") as f:
                existing_data = yaml.safe_load(f) or {"examples": {"queries": []}}

            # Ensure proper structure
            if "examples" not in existing_data:
                existing_data["examples"] = {"queries": []}
            elif "queries" not in existing_data["examples"]:
                existing_data["examples"]["queries"] = []

            # Count new additions
            initial_count = len(existing_data["examples"]["queries"])

            # Add validated examples
            for example in validated_examples:
                if "question" in example and "sql" in example:
                    yaml_example = {
                        "question": example["question"],
                        "sql": example["sql"],
                    }
                    existing_data["examples"]["queries"].append(yaml_example)

            # Write back to file
            with open(main_yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    existing_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            added_count = len(existing_data["examples"]["queries"]) - initial_count
            logger.info(
                f"📝 APPENDED {added_count} validated examples to main sql_examples.yaml"
            )

        except Exception as e:
            logger.error(f"Failed to append to main YAML file: {e}")

    def load_existing_examples(self) -> list[dict[str, str]]:
        """
        Load existing SQL examples from all YAML files in the training directory.

        Returns:
            List of existing SQL examples

        """
        try:
            # Find all YAML files in training directory and subdirectories
            yaml_files = list(self.training_data_dir.glob("*.yaml")) + list(
                self.training_data_dir.glob("**/*.yaml")
            )

            if not yaml_files:
                logger.warning(f"No YAML files found in {self.training_data_dir}")
                return []

            logger.info(
                f"🔄 Loading predefined SQL examples from {len(yaml_files)} YAML files..."
            )

            all_examples = []
            files_loaded = 0

            for yaml_file in yaml_files:
                try:
                    logger.info(f"📖 Reading: {yaml_file.name}")

                    with open(yaml_file, encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f)

                    if not yaml_data:
                        logger.warning(f"Empty YAML file: {yaml_file.name}")
                        continue

                    # Handle different YAML structures
                    examples = []
                    if "examples" in yaml_data and "queries" in yaml_data["examples"]:
                        examples = yaml_data["examples"]["queries"]
                    elif isinstance(yaml_data, list):
                        examples = yaml_data
                    elif "queries" in yaml_data:
                        examples = yaml_data["queries"]
                    else:
                        logger.warning(f"No examples found in: {yaml_file.name}")
                        continue

                    # Process examples
                    file_examples = []
                    for example in examples:
                        if isinstance(example, dict):
                            # Ensure required fields exist
                            if "question" in example and "sql" in example:
                                clean_example = {
                                    "question": example["question"],
                                    "sql": example["sql"],
                                    # Add explanation if missing
                                    "explanation": example.get(
                                        "explanation", f"Example from {yaml_file.name}"
                                    ),
                                }
                                file_examples.append(clean_example)

                    if file_examples:
                        all_examples.extend(file_examples)
                        files_loaded += 1
                        logger.info(
                            f"✅ Loaded {len(file_examples)} examples from {yaml_file.name}"
                        )
                    else:
                        logger.warning(f"No valid examples found in: {yaml_file.name}")

                except Exception as e:
                    logger.error(f"Failed to load examples from {yaml_file.name}: {e}")
                    continue

            # Remove duplicates based on SQL content
            unique_examples = []
            seen_sql = set()

            for example in all_examples:
                sql_normalized = example["sql"].strip().lower()
                if sql_normalized not in seen_sql:
                    unique_examples.append(example)
                    seen_sql.add(sql_normalized)

            logger.info(
                f"🎯 TOTAL: Loaded {len(unique_examples)} unique SQL examples from {files_loaded}/{len(yaml_files)} YAML files"
            )

            return unique_examples

        except Exception as e:
            logger.error(f"Failed to load existing examples: {e}")
            return []

    def get_training_data_dir(self) -> Path:
        """Get the training data directory path."""
        return self.training_data_dir
