"""
Universal schema introspection using Ibis.

Replaces: scripts/extract_athena_schema.py (981 lines)

This module provides universal schema extraction that works with any
Ibis backend, replacing the 981-line Athena-specific extractor.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beartype import beartype
from icontract import require
from loguru import logger

from src.connectors.analytics_backend import AnalyticsConnector

if TYPE_CHECKING:
    pass


@beartype
class SchemaIntrospector:
    """
    Extract database schema using Ibis (works with any backend).

    This class replaces the 981-line AthenaSchemaExtractor with a
    universal approach that works with any Ibis-supported database.

    Optionally enriches with Glue metadata for Athena.

    Attributes:
        backend: Ibis backend instance
        glue_enricher: Optional Glue metadata enricher
        schema_cache: Cached schema data for performance

    Examples:
        >>> backend = AnalyticsConnector("athena://awsdatacatalog?region=ap-southeast-2&database=mydb")
        >>> introspector = SchemaIntrospector(backend)
        >>> schema = introspector.extract_database_schema("mydb")
        >>> print(f"Found {schema['table_count']} tables")

    """

    __slots__ = ("backend", "glue_enricher", "schema_cache")

    def __init__(self, backend: Any, glue_enricher: Any | None = None) -> None:
        """
        Initialize schema introspector.

        Args:
            backend: Ibis backend instance
            glue_enricher: Optional Glue enricher for Athena

        """
        self.backend = backend
        self.glue_enricher = glue_enricher
        self.schema_cache: dict[str, dict[str, Any]] = {}

        logger.info(
            f"SchemaIntrospector initialized for {backend.backend_type} backend"
        )
        if glue_enricher:
            logger.info("✓ Glue enrichment enabled")

    @beartype
    def get_available_databases(self) -> list[str]:
        """
        List available databases.

        Returns:
            List of database names

        Raises:
            RuntimeError: If database listing fails

        """
        try:
            databases = self.backend.list_databases()
            logger.info(f"Found {len(databases)} databases: {databases}")
            return databases
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            return []

    @beartype
    @require(
        lambda database_name: len(database_name) > 0, "Database name cannot be empty"
    )
    def extract_database_schema(self, database_name: str) -> dict[str, Any]:
        """
        Extract comprehensive schema for a database.

        This method replaces the complex Athena-specific schema extraction
        with a universal approach that works with any Ibis backend.

        Args:
            database_name: Database to extract

        Returns:
            Schema data dictionary with tables and metadata

        """
        # Check cache first
        if database_name in self.schema_cache:
            logger.info(f"Using cached schema for {database_name}")
            return self.schema_cache[database_name]

        logger.info(f"Extracting schema for database: {database_name}")

        try:
            # Get tables from Ibis (universal approach)
            # NOTE: Use _default_database if set for Athena backends
            db_param = (
                self.backend._default_database
                if hasattr(self.backend, "_default_database")
                and self.backend._default_database
                else database_name
            )
            tables = self.backend.list_tables(database=db_param)

            if not tables:
                logger.warning(f"No tables found in database: {database_name}")
                return {
                    "database_name": database_name,
                    "tables": {},
                    "table_count": 0,
                    "extracted_at": datetime.now().isoformat(),
                    "source": self.backend.backend_type,
                }

            # Extract schema for each table
            table_schemas = {}
            for table_name in tables:
                table_schema = self._extract_table_schema(database_name, table_name)
                if table_schema:
                    table_schemas[table_name] = table_schema

            # Build schema data
            schema_data = {
                "database_name": database_name,
                "tables": table_schemas,
                "table_count": len(table_schemas),
                "extracted_at": datetime.now().isoformat(),
                "source": self.backend.backend_type,
            }

            # Enrich with Glue metadata if available (Athena only)
            if self.glue_enricher:
                logger.debug("Applying Glue enrichment")
                schema_data = self.glue_enricher.enrich(schema_data)

            # Cache result
            self.schema_cache[database_name] = schema_data

            logger.info(f"✓ Extracted schema: {len(table_schemas)} tables")
            return schema_data

        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            return {
                "database_name": database_name,
                "tables": {},
                "table_count": 0,
                "extracted_at": datetime.now().isoformat(),
                "source": self.backend.backend_type,
                "error": str(e),
            }

    @beartype
    def _extract_table_schema(
        self, database_name: str, table_name: str
    ) -> dict[str, Any]:
        """
        Extract schema for a single table.

        Args:
            database_name: Database name
            table_name: Table name

        Returns:
            Table schema dictionary

        """
        try:
            # NOTE: Use _default_database if set for Athena backends
            db_param = (
                self.backend._default_database
                if hasattr(self.backend, "_default_database")
                and self.backend._default_database
                else database_name
            )
            schema_info = self.backend.get_table_schema(table_name, database=db_param)

            if not schema_info.get("columns"):
                logger.warning(
                    f"No columns found for table: {database_name}.{table_name}"
                )
                return {}

            return {
                "table_name": table_name,
                "database_name": database_name,
                "columns": schema_info["columns"],
                "column_count": schema_info["column_count"],
            }

        except Exception as e:
            logger.error(f"Failed to extract {database_name}.{table_name}: {e}")
            return {}

    @beartype
    def extract_multiple_databases(
        self, database_names: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Extract schemas for multiple databases.

        Args:
            database_names: List of database names to extract

        Returns:
            Dictionary mapping database names to their schemas

        """
        logger.info(f"Extracting schemas for {len(database_names)} databases")

        schemas = {}
        for database_name in database_names:
            schema = self.extract_database_schema(database_name)
            if schema and schema.get("table_count", 0) > 0:
                schemas[database_name] = schema

        logger.info(f"Successfully extracted schemas for {len(schemas)} databases")
        return schemas

    @beartype
    def get_schema_statistics(self, database_name: str) -> dict[str, Any]:
        """
        Get statistics about extracted schema.

        Args:
            database_name: Database name

        Returns:
            Statistics dictionary

        """
        if database_name not in self.schema_cache:
            logger.warning(f"No cached schema for database: {database_name}")
            return {}

        schema = self.schema_cache[database_name]
        tables = schema.get("tables", {})

        # Calculate statistics
        total_columns = sum(table.get("column_count", 0) for table in tables.values())

        column_types = {}
        for table in tables.values():
            for col in table.get("columns", []):
                col_type = col.get("type", "unknown")
                column_types[col_type] = column_types.get(col_type, 0) + 1

        return {
            "database_name": database_name,
            "table_count": len(tables),
            "total_columns": total_columns,
            "column_types": column_types,
            "backend_type": schema.get("source", "unknown"),
            "extracted_at": schema.get("extracted_at"),
        }

    @beartype
    def clear_cache(self, database_name: str | None = None) -> None:
        """
        Clear schema cache.

        Args:
            database_name: Specific database to clear, or None for all

        """
        if database_name:
            if database_name in self.schema_cache:
                del self.schema_cache[database_name]
                logger.info(f"Cleared cache for database: {database_name}")
        else:
            self.schema_cache.clear()
            logger.info("Cleared all schema cache")

    @beartype
    def save_schema_to_file(self, database_name: str, output_file: Path) -> bool:
        """
        Save extracted schema to JSON file.

        Args:
            database_name: Database name
            output_file: Output file path

        Returns:
            True if successful, False otherwise

        """
        try:
            import json

            schema = self.extract_database_schema(database_name)
            if not schema:
                logger.error(f"No schema to save for database: {database_name}")
                return False

            with open(output_file, "w") as f:
                json.dump(schema, f, indent=2, default=str)

            logger.info(f"✓ Schema saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save schema to {output_file}: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.backend:
            self.backend.close()
        logger.info("SchemaIntrospector cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SchemaIntrospector(backend='{self.backend.backend_type}')"


# Utility functions for backward compatibility
@beartype
def create_schema_introspector(
    connection_string: str,
    enable_glue_enrichment: bool = False,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> SchemaIntrospector:
    """
    Create SchemaIntrospector with optional Glue enrichment.

    Args:
        connection_string: Database connection string
        enable_glue_enrichment: Whether to enable Glue enrichment (Athena only)
        aws_profile: AWS profile for Glue enrichment
        aws_region: AWS region for Glue enrichment

    Returns:
        Configured SchemaIntrospector instance

    """
    backend = AnalyticsConnector(connection_string)

    glue_enricher = None
    if enable_glue_enrichment and "athena://" in connection_string:
        if not aws_profile or not aws_region:
            logger.warning("Glue enrichment requires AWS profile and region")
        else:
            from src.utils.glue_enricher import GlueMetadataEnricher

            glue_enricher = GlueMetadataEnricher(aws_profile, aws_region)

    return SchemaIntrospector(backend, glue_enricher)
