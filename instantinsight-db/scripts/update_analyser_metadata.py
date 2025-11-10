#!/usr/bin/env python3
"""
Inject Analyser Safe Join Metadata into RAG System.

WORKFLOW:
1. Data Engineer/SME fills analyser_definitions.yaml with safe joins
2. Run this script to inject metadata into RAG database
3. AI agents now understand which joins are safe vs dangerous

DEFAULT BEHAVIOR: All Analysers are dangerous to join unless explicitly configured

Usage:
    # Apply safe joins from YAML to RAG system
    python scripts/update_analyser_metadata.py

    # Check specific Analyser current settings
    python scripts/update_analyser_metadata.py --table sales_analyser --show-only

    # Test join safety between two Analysers
    python scripts/update_analyser_metadata.py --check-join table1 table2
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
from loguru import logger
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.components.analyser_metadata import AnalyserMetadata


class AnalyserMetadataUpdater:
    """Updates Analyser metadata in the RAG PostgreSQL database."""

    def __init__(
        self, db_config: dict, config_path: str = "config/analyser_definitions.yaml"
    ):
        """
        Initialize the updater.

        Args:
            db_config: PostgreSQL connection configuration
            config_path: Path to analyser definitions YAML

        """
        self.db_config = db_config
        self.analyser_metadata = AnalyserMetadata(config_path)
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Disconnect from database."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_all_analysers(self) -> list:
        """Get all schema documents from the RAG database."""
        # First try to get documents with table_name in metadata (new format)
        query = """
        SELECT 
            id,
            metadata,
            doc_type
        FROM rag_documents 
        WHERE doc_type = 'schema'
        AND source = 'database'
        AND metadata->>'table_name' IS NOT NULL
        ORDER BY id DESC
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        analysers = []
        seen_tables = set()

        # Use documents with table_name in metadata
        for result in results:
            metadata = result["metadata"] or {}
            table_name = (
                metadata.get("table_name") if isinstance(metadata, dict) else None
            )

            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                analysers.append(
                    {
                        "table_name": table_name,
                        "metadata": metadata,
                        "doc_type": result["doc_type"],
                        "id": result["id"],
                    }
                )

        # ALWAYS also check DDL extraction for documents without table_name metadata
        logger.debug("Checking for additional tables in DDL content")
        fallback_query = """
        SELECT 
            id,
            metadata,
            doc_type,
            full_content
        FROM rag_documents 
        WHERE doc_type = 'schema'
        AND source = 'database'
        AND (metadata->>'table_name' IS NULL OR metadata->>'table_name' = '')
        ORDER BY id DESC
        """

        self.cursor.execute(fallback_query)
        fallback_results = self.cursor.fetchall()

        for result in fallback_results:
            metadata = result["metadata"] or {}

            # Extract table name from CREATE EXTERNAL TABLE statement
            table_name = None
            if result["full_content"]:
                import re

                # Try multiple patterns to match different DDL formats
                patterns = [
                    r"CREATE EXTERNAL TABLE `[^`]+`\.`([^`]+)`",  # backtick quoted
                    r"CREATE EXTERNAL TABLE [^.]+\.([^.\s(]+)",  # unquoted
                    r"CREATE TABLE `[^`]+`\.`([^`]+)`",  # CREATE TABLE variant
                    r"CREATE TABLE [^.]+\.([^.\s(]+)",  # unquoted variant
                ]

                for pattern in patterns:
                    match = re.search(pattern, result["full_content"], re.IGNORECASE)
                    if match:
                        table_name = match.group(1).strip('`"')
                        logger.debug(
                            f"Extracted table name '{table_name}' using pattern: {pattern}"
                        )
                        break

                if not table_name:
                    logger.debug(
                        f"Could not extract table name from: {result['full_content'][:100]}..."
                    )

            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                analysers.append(
                    {
                        "table_name": table_name,
                        "metadata": metadata,
                        "doc_type": result["doc_type"],
                        "id": result["id"],
                    }
                )

        return analysers

    def get_analyser(self, table_name: str) -> dict | None:
        """Get specific Analyser from the RAG database."""
        # Use the get_all_analysers method to find the table - this ensures consistency
        all_analysers = self.get_all_analysers()

        for analyser in all_analysers:
            if analyser["table_name"] == table_name:
                return analyser

        logger.debug(
            f"No schema document found for table '{table_name}' in {len(all_analysers)} total analysers"
        )
        return None

    def update_analyser_metadata(
        self, table_name: str, show_only: bool = False
    ) -> bool:
        """
        Update metadata for a specific Analyser.

        Args:
            table_name: Name of the Analyser table
            show_only: If True, only show current metadata without updating

        Returns:
            True if successful

        """
        # Get current metadata from database
        current = self.get_analyser(table_name)

        if not current:
            logger.info(
                f"Analyser '{table_name}' not found in RAG database (may not be imported yet)"
            )
            return False

        current_metadata = current["metadata"] or {}

        if show_only:
            logger.info(f"\nCurrent metadata for {table_name}:")
            print(json.dumps(current_metadata, indent=2))
            return True

        # Get enriched metadata from configuration
        enriched_metadata = self.analyser_metadata.get_metadata(table_name)

        # Start with existing metadata but clean out old analyser fields
        updated_metadata = current_metadata.copy()

        # Remove old/deprecated analyser fields for clean format
        deprecated_fields = [
            "analyser_warnings",
            "join_safety_level",
            "analyser_description",
        ]
        for field in deprecated_fields:
            updated_metadata.pop(field, None)

        # Add Analyser-specific fields (clean format)
        updated_metadata.update(
            {
                "table_name": table_name,
                "is_analyser": True,
                "unique_key": enriched_metadata.get("unique_key", []),
                "grain": enriched_metadata.get("grain", "Row-level data"),
                "safe_to_join_with": enriched_metadata.get("safe_to_join_with", []),
            }
        )

        # Update existing document metadata
        if current and current["id"] is not None:
            update_query = """
            UPDATE rag_documents 
            SET metadata = %s
            WHERE id = %s
            """
            self.cursor.execute(
                update_query, (json.dumps(updated_metadata), current["id"])
            )
        else:
            logger.warning(
                f"Cannot update {table_name} - no existing schema document found"
            )
            return False

        logger.info(f"Updated metadata for {table_name}")

        # Also update any related embeddings
        self._update_embeddings_metadata(table_name, updated_metadata)

        return True

    def _update_embeddings_metadata(self, table_name: str, metadata: dict):
        """Update metadata in the embeddings table as well."""
        update_query = """
        UPDATE rag_embeddings 
        SET metadata = metadata || %s
        WHERE metadata->>'table_name' = %s
        """

        # Only include key fields in embeddings metadata
        embedding_metadata = {
            "is_analyser": True,
            "grain": metadata.get("grain"),
            "safe_to_join_with": metadata.get("safe_to_join_with", []),
        }

        self.cursor.execute(update_query, (json.dumps(embedding_metadata), table_name))

    def _create_analyser_description(self, metadata: dict) -> str:
        """Create a minimal join safety description."""
        if metadata.get("safe_to_join_with"):
            return f"Can join with: {', '.join(metadata['safe_to_join_with'])}"
        else:
            return "DO NOT JOIN with other Analysers"

    def cleanup_duplicate_schemas(self):
        """Remove duplicate schema documents, keeping only the most recent ones."""
        # DISABLED: Cleanup is too risky - only clean if we can find exact matches
        logger.debug(
            "Cleanup disabled to prevent accidental deletion of analyser documents"
        )
        return

    def update_all_analysers(self):
        """Update metadata for all Analysers in the database."""
        # First clean up any duplicates
        self.cleanup_duplicate_schemas()

        analysers = self.get_all_analysers()

        # Get all defined analysers from YAML
        yaml_analysers = self.analyser_metadata.get_all_analysers()

        logger.info(f"Found {len(analysers)} tables in database")
        logger.info(f"Found {len(yaml_analysers)} analysers defined in YAML")

        # Update existing analysers
        success_count = 0
        failed_count = 0

        for analyser in analysers:
            table_name = analyser["table_name"]

            # Only update if defined in YAML
            if table_name in yaml_analysers:
                logger.info(f"Updating {table_name}...")
                if self.update_analyser_metadata(table_name):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                logger.debug(
                    f"Skipping {table_name} (not in analyser_definitions.yaml)"
                )

        # List any YAML-defined analysers not in database
        db_tables = {a["table_name"] for a in analysers}
        missing_tables = set(yaml_analysers.keys()) - db_tables
        if missing_tables:
            logger.warning(
                f"These analysers are defined in YAML but not found in database: {', '.join(missing_tables)}"
            )

        logger.info(f"Successfully updated {success_count} analysers")
        if failed_count > 0:
            logger.warning(f"Failed to update {failed_count} analysers")

        # Commit all changes
        self.conn.commit()

    def show_join_safety(self, table1: str, table2: str):
        """Check and display join safety between two Analysers."""
        is_safe, join_key = self.analyser_metadata.is_join_safe(table1, table2)
        warning = self.analyser_metadata.get_join_warning(table1, table2)

        print(f"\nJoin Safety Analysis: {table1} <-> {table2}")
        print(f"{'=' * 50}")
        print(warning)

        if is_safe and join_key:
            print(f"Join key: {join_key}")


def main():
    """Run the analyser metadata update workflow."""
    parser = argparse.ArgumentParser(
        description="Update Analyser metadata in RAG system"
    )
    parser.add_argument(
        "--table", help="Specific table to update (updates all if not specified)"
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Only show current metadata without updating",
    )
    parser.add_argument(
        "--check-join",
        nargs=2,
        metavar=("TABLE1", "TABLE2"),
        help="Check if joining two tables is safe",
    )
    parser.add_argument(
        "--config",
        default="config/analyser_definitions.yaml",
        help="Path to analyser definitions YAML",
    )

    args = parser.parse_args()

    # Database configuration - use environment variables in production
    import os

    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "instantinsight"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }

    # Create updater
    updater = AnalyserMetadataUpdater(db_config, args.config)

    try:
        updater.connect()

        if args.check_join:
            # Check join safety
            updater.show_join_safety(args.check_join[0], args.check_join[1])

        elif args.table:
            # Update specific table
            updater.update_analyser_metadata(args.table, args.show_only)
            if not args.show_only:
                updater.conn.commit()
                logger.info("Changes committed to database")

        else:
            # Update all tables
            if args.show_only:
                analysers = updater.get_all_analysers()
                logger.info(f"\nFound {len(analysers)} Analysers:")
                for a in analysers:
                    print(f"  - {a['table_name']}")
            else:
                updater.update_all_analysers()
                logger.info("All Analysers updated with latest metadata")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    finally:
        updater.disconnect()


if __name__ == "__main__":
    main()
