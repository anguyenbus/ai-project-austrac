#!/usr/bin/env python3
"""
SchemaLogic Setup Script.

This script sets up the SchemaLogic system for hybrid database architecture
with intelligent query routing.

The complex logic uses modular components:
- ConfigLoader: Configuration management
- PrerequisiteValidator: System validation
- AnalyserFilter: Table filtering logic
- SetupOrchestrator: Main workflow coordination

Usage:
    poetry run python scripts/setup_schema_logic.py --setup-all
    poetry run python scripts/setup_schema_logic.py --databases db1,db2 --update
    poetry run python scripts/setup_schema_logic.py --validate-only
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.setup import ConfigLoader, SetupOrchestrator  # noqa: E402


class SchemaLogicSetup:
    """
    Setup manager for SchemaLogic system.

    This class provides a clean interface for setting up the SchemaLogic
    system using modular components.
    """

    def __init__(self, config_override: dict[str, Any] = None):
        """
        Initialize the setup manager.

        Args:
            config_override: Optional configuration overrides

        """
        # Create temporary config file if override provided
        config_file = None
        if config_override:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(config_override, f)
                config_file = Path(f.name)

        # Initialize components
        self.config_loader = ConfigLoader(config_file)
        self.orchestrator = SetupOrchestrator(self.config_loader, project_root)

        # Expose setup report for compatibility
        self.setup_report = self.orchestrator.setup_report

        logger.info("🚀 SchemaLogic Setup Manager initialized")

        # Clean up temp file
        if config_file and config_file.exists():
            config_file.unlink()

    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites for SchemaLogic setup."""
        return self.orchestrator.validate_prerequisites()

    def setup_complete_pipeline(
        self,
        database_names: list[str] = None,
        force_rebuild: bool = False,
        generate_examples: bool = False,
    ) -> bool:
        """Set up the complete SchemaLogic pipeline."""
        return self.orchestrator.run_full_setup(
            database_names, force_rebuild, generate_examples
        )

    def update_existing_setup(
        self, database_names: list[str] = None, generate_examples: bool = False
    ) -> bool:
        """Update existing SchemaLogic setup with new or changed data."""
        return self.orchestrator.run_update(database_names, generate_examples)

    def update_with_new_tables_only(
        self, database_names: list[str] = None, generate_examples: bool = False
    ) -> bool:
        """Update SchemaLogic system with only tables that don't already exist."""
        return self.orchestrator.run_new_tables_only_update(
            database_names, generate_examples
        )

    def generate_examples_for_specific_tables(
        self, table_names: list[str], database_names: list[str] = None
    ) -> bool:
        """Generate SQL examples for specific tables by name."""
        return self.orchestrator.run_specific_table_examples(
            table_names, database_names
        )

    def cleanup(self):
        """Clean up resources."""
        self.orchestrator.cleanup()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Setup SchemaLogic integration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python scripts/setup_schema_logic.py --setup-all                          # Setup with all databases, load existing examples
  poetry run python scripts/setup_schema_logic.py --databases sales,marketing          # Setup specific databases, load existing examples
  poetry run python scripts/setup_schema_logic.py --validate-only                      # Only validate prerequisites
  poetry run python scripts/setup_schema_logic.py --update --databases analytics       # Update specific database, load existing examples
  poetry run python scripts/setup_schema_logic.py --update-new-tables-only             # Update only with new tables not in system (efficient)
  poetry run python scripts/setup_schema_logic.py --table-names table1,table2 --generate-examples  # Generate examples for specific tables only
  poetry run python scripts/setup_schema_logic.py --setup-all --generate-examples      # Setup with NEW SQL examples generation
  poetry run python scripts/setup_schema_logic.py --force-rebuild --generate-examples  # Force rebuild with NEW SQL examples
  poetry run python scripts/setup_schema_logic.py --force-rebuild                      # Force rebuild, load existing examples from src/training/data/

Analyser Filtering:
  Create artifacts/analysers_to_be_included.yaml to specify which analysers to include:
  
  analysers:
    - analyser1
    - analyser2
    - analyser3
  
  OR for database-specific inclusions:
  
  cleaned_dataformats_data:
    - analyser1
    - analyser2
  
  If this file is empty or doesn't exist, all analysers will be processed 
  except those listed in artifacts/analysers_to_be_excluded.yaml
        """,
    )

    parser.add_argument(
        "--setup-all",
        action="store_true",
        help="Setup pipeline for all available databases",
    )

    parser.add_argument(
        "--databases",
        "-d",
        help="Comma-separated list of database names to process",
        default="text_to_sql",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate prerequisites without setup",
    )

    parser.add_argument(
        "--update", action="store_true", help="Update existing setup (incremental mode)"
    )

    parser.add_argument(
        "--update-new-tables-only",
        action="store_true",
        help="Update only with new tables that don't exist in local RAG (more efficient than --update)",
    )

    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild of existing data"
    )

    parser.add_argument(
        "--generate-examples",
        action="store_true",
        help="Generate SQL examples using LLM during setup (REQUIRES LLM - no fallback, fails if LLM unavailable). "
        "Without this flag, existing examples from src/training/data/ will be loaded instead.",
    )

    parser.add_argument(
        "--table-names",
        "-t",
        help="Comma-separated list of specific table names to process for example generation. "
        "When used with --generate-examples, only these tables will have examples generated.",
    )

    parser.add_argument(
        "--config-file", type=Path, help="Path to configuration override file"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def configure_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    logger.remove()  # Remove default handler

    level = "DEBUG" if verbose else "INFO"

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )


def parse_comma_separated_list(value: str) -> list:
    """Parse a comma-separated string into a list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",")]


def validate_arguments(args) -> bool:
    """Validate argument combinations."""
    # Validate that --table-names requires --generate-examples
    if args.table_names and not args.generate_examples:
        logger.error("❌ --table-names requires --generate-examples to be specified")
        logger.info("💡 Use: --table-names table1,table2 --generate-examples")
        return False

    return True


def run_setup_workflow(args, orchestrator: SetupOrchestrator) -> bool:
    """Run the appropriate setup workflow based on arguments."""
    # Parse database names
    database_names = None
    if args.databases:
        database_names = parse_comma_separated_list(args.databases)

    # Parse table names if provided
    table_names = None
    if args.table_names:
        table_names = parse_comma_separated_list(args.table_names)

    # Run setup, update, or specific table processing
    if table_names and args.generate_examples:
        # Special mode: Generate examples for specific tables only
        logger.info("🎯 Specific table example generation mode")
        success = orchestrator.run_specific_table_examples(table_names, database_names)
    elif args.update_new_tables_only:
        success = orchestrator.run_new_tables_only_update(
            database_names, generate_examples=args.generate_examples
        )
    elif args.update:
        success = orchestrator.run_update(
            database_names, generate_examples=args.generate_examples
        )
    else:
        success = orchestrator.run_full_setup(
            database_names,
            force_rebuild=args.force_rebuild,
            generate_examples=args.generate_examples,
        )

    return success


def handle_setup_issues(orchestrator: SetupOrchestrator) -> None:
    """Handle setup issues with enhanced debugging."""
    logger.info("🔧 Setup had issues, enabling debug logging for troubleshooting...")
    configure_logging(verbose=True)

    # Run validation again with debug logging
    logger.debug("Re-running validation with debug logging enabled...")

    # Check if we should expect new documents based on the setup report
    new_tables_added = orchestrator.setup_report.get("new_tables_added", -1)
    if new_tables_added >= 0:
        # We're in update-new-tables-only mode
        orchestrator._validate_setup(expect_new_documents=(new_tables_added > 0))
    else:
        # Default behavior for other modes
        orchestrator._validate_setup()


def main() -> int:
    """Execute command-line setup workflow."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging(args.verbose)

    # Validate arguments
    if not validate_arguments(args):
        return 1

    # Get project root
    project_root = Path(__file__).parent.parent

    # Initialize configuration loader
    try:
        config_loader = ConfigLoader(args.config_file)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Initialize setup orchestrator
    orchestrator = SetupOrchestrator(config_loader, project_root)

    try:
        # Validate prerequisites
        if not orchestrator.validate_prerequisites():
            logger.error(
                "❌ Prerequisites validation failed. Please fix the issues and try again."
            )
            return 1

        if args.validate_only:
            logger.info("✅ Prerequisites validation completed successfully")
            return 0

        # Run the appropriate workflow
        success = run_setup_workflow(args, orchestrator)

        # Check if setup had issues and needs debugging
        new_tables_added = orchestrator.setup_report.get("new_tables_added", -1)
        is_no_new_tables_scenario = (
            new_tables_added == 0 and args.update_new_tables_only
        )
        has_real_warnings = (
            orchestrator.setup_report.get("warnings") and not is_no_new_tables_scenario
        )

        if not success or has_real_warnings:
            handle_setup_issues(orchestrator)

        if success:
            logger.info("🎉 SchemaLogic setup completed successfully!")
            logger.info(
                "You can now use the hybrid database architecture with intelligent query routing."
            )
            return 0
        else:
            logger.error("❌ Setup failed. Check the logs for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("\nSetup cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        return 1
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
