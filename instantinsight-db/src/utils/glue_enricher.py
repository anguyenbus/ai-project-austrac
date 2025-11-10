"""
AWS Glue metadata enrichment for Athena tables.

Adds S3 locations, partition keys, SERDE info to Ibis-extracted schemas.

This module provides optional Athena-specific metadata enrichment while keeping
the core schema extraction universal.
"""

from typing import Any

import boto3
from beartype import beartype
from loguru import logger


@beartype
class GlueMetadataEnricher:
    """
    Enrich Ibis schemas with AWS Glue Catalog metadata.

    This class adds Athena-specific metadata to universally extracted
    schemas, preserving S3 locations, partition keys, and SERDE info.

    Attributes:
        glue_client: Boto3 Glue client
        aws_profile: AWS profile name
        aws_region: AWS region

    Examples:
        >>> enricher = GlueMetadataEnricher("default", "ap-southeast-2")
        >>> enriched_schema = enricher.enrich(schema_data)
        >>> print(f"Enriched {len(enriched_schema['tables'])} tables")

    """

    __slots__ = ("glue_client", "aws_profile", "aws_region")

    def __init__(self, aws_profile: str, region: str) -> None:
        """
        Initialize Glue enricher.

        Args:
            aws_profile: AWS profile name
            region: AWS region

        Raises:
            ImportError: If boto3 is not available
            RuntimeError: If AWS session fails

        """
        self.aws_profile = aws_profile
        self.aws_region = region

        try:
            session = boto3.Session(profile_name=aws_profile, region_name=region)
            self.glue_client = session.client("glue")
            logger.info(
                f"✓ Glue enricher initialized (profile: {aws_profile}, region: {region})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Glue enricher: {e}")
            raise RuntimeError(f"Glue enricher initialization failed: {e}") from e

    @beartype
    def enrich(self, schema_data: dict[str, Any]) -> dict[str, Any]:
        """
        Add Glue metadata to schema.

        This method enriches universally extracted schema data with
        Athena-specific metadata from AWS Glue Catalog.

        Adds:
        - S3 location
        - Partition keys
        - SERDE info
        - Table descriptions
        - Creation times
        - Table parameters

        Args:
            schema_data: Schema dictionary from Ibis introspector

        Returns:
            Enriched schema dictionary

        Examples:
            >>> enricher = GlueMetadataEnricher("default", "ap-southeast-2")
            >>> schema = {"database_name": "mydb", "tables": {"users": {...}}}
            >>> enriched = enricher.enrich(schema)
            >>> "s3_location" in enriched["tables"]["users"]
            True

        """
        database_name = schema_data.get("database_name")
        if not database_name:
            logger.warning("No database name in schema data")
            return schema_data

        tables = schema_data.get("tables", {})
        enriched_count = 0

        for table_name, table_info in tables.items():
            try:
                glue_table = self._get_glue_table(database_name, table_name)

                if not glue_table:
                    logger.warning(f"Table {table_name} not found in Glue catalog")
                    continue

                # Extract storage descriptor
                storage_desc = glue_table.get("StorageDescriptor", {})

                # Add Glue-specific metadata
                table_info["s3_location"] = storage_desc.get("Location", "")
                table_info["partition_keys"] = glue_table.get("PartitionKeys", [])
                table_info["serde_info"] = storage_desc.get("SerdeInfo", {})
                table_info["description"] = glue_table.get("Description", "")
                table_info["table_type"] = glue_table.get("TableType", "")
                table_info["parameters"] = glue_table.get("Parameters", {})

                # Add timing information
                table_info["creation_time"] = glue_table.get("CreateTime")
                table_info["last_analyzed_time"] = glue_table.get("LastAnalyzedTime")
                table_info["retention"] = glue_table.get("Retention")

                # Add input/output format information
                table_info["input_format"] = storage_desc.get("InputFormat", "")
                table_info["output_format"] = storage_desc.get("OutputFormat", "")
                table_info["compressed"] = storage_desc.get("Compressed", False)

                # Add column-level comments from Glue
                self._enrich_columns_with_glue_comments(table_info, storage_desc)

                enriched_count += 1
                logger.debug(f"✓ Enriched {table_name} with Glue metadata")

            except Exception as e:
                logger.warning(f"Could not enrich {table_name}: {e}")
                continue

        logger.info(
            f"✓ Enriched {enriched_count}/{len(tables)} tables with Glue metadata"
        )

        # Add enrichment metadata
        schema_data["glue_enrichment"] = {
            "enabled": True,
            "enriched_tables": enriched_count,
            "total_tables": len(tables),
            "enriched_at": boto3.Session().region_name,
        }

        return schema_data

    @beartype
    def _get_glue_table(self, database_name: str, table_name: str) -> dict[str, Any]:
        """
        Get table metadata from Glue catalog.

        Args:
            database_name: Database name
            table_name: Table name

        Returns:
            Glue table dictionary or None if not found

        """
        try:
            response = self.glue_client.get_table(
                DatabaseName=database_name, Name=table_name
            )
            return response.get("Table", {})
        except Exception as e:
            logger.debug(f"Failed to get Glue table {database_name}.{table_name}: {e}")
            return {}

    @beartype
    def _enrich_columns_with_glue_comments(
        self, table_info: dict[str, Any], storage_desc: dict[str, Any]
    ) -> None:
        """
        Enrich column information with Glue comments.

        Args:
            table_info: Table information dictionary to modify
            storage_desc: Storage descriptor from Glue

        """
        glue_columns = storage_desc.get("Columns", [])
        if not glue_columns:
            return

        # Create mapping of column name to Glue metadata
        glue_column_map = {
            col["Name"]: {
                "comment": col.get("Comment", ""),
                "type": col.get("Type", ""),
            }
            for col in glue_columns
        }

        # Enrich existing columns with Glue comments
        for col in table_info.get("columns", []):
            col_name = col.get("name")
            if col_name in glue_column_map:
                glue_col = glue_column_map[col_name]
                col["glue_comment"] = glue_col["comment"]
                col["glue_type"] = glue_col["type"]

                # Use Glue comment if no existing comment
                if not col.get("comment") and glue_col["comment"]:
                    col["comment"] = glue_col["comment"]

    @beartype
    def get_table_location(self, database_name: str, table_name: str) -> str:
        """
        Get S3 location for a specific table.

        Args:
            database_name: Database name
            table_name: Table name

        Returns:
            S3 location string or empty string if not found

        """
        try:
            glue_table = self._get_glue_table(database_name, table_name)
            return glue_table.get("StorageDescriptor", {}).get("Location", "")
        except Exception as e:
            logger.error(f"Failed to get table location: {e}")
            return ""

    @beartype
    def get_partition_keys(
        self, database_name: str, table_name: str
    ) -> list[dict[str, str]]:
        """
        Get partition keys for a specific table.

        Args:
            database_name: Database name
            table_name: Table name

        Returns:
            List of partition key dictionaries

        """
        try:
            glue_table = self._get_glue_table(database_name, table_name)
            return glue_table.get("PartitionKeys", [])
        except Exception as e:
            logger.error(f"Failed to get partition keys: {e}")
            return []

    @beartype
    def test_glue_access(self, database_name: str) -> bool:
        """
        Test access to Glue catalog for a database.

        Args:
            database_name: Database name to test

        Returns:
            True if access is successful, False otherwise

        """
        try:
            self.glue_client.get_database(Name=database_name)
            logger.info(f"✓ Glue access confirmed for database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Glue access failed for {database_name}: {e}")
            return False

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GlueMetadataEnricher(profile='{self.aws_profile}', region='{self.aws_region}')"


# Utility functions
@beartype
def create_glue_enricher_if_needed(
    connection_string: str,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> GlueMetadataEnricher | None:
    """
    Create Glue enricher only if needed (Athena backend).

    Args:
        connection_string: Database connection string
        aws_profile: AWS profile for Glue access
        aws_region: AWS region for Glue access

    Returns:
        GlueMetadataEnricher instance or None

    """
    if "athena://" not in connection_string:
        logger.debug("Glue enrichment not needed for non-Athena backend")
        return None

    if not aws_profile or not aws_region:
        logger.warning("Glue enrichment requires AWS profile and region")
        return None

    try:
        return GlueMetadataEnricher(aws_profile, aws_region)
    except Exception as e:
        logger.error(f"Failed to create Glue enricher: {e}")
        return None
