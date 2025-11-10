"""
Athena DDL Generation Module.

Handles creation of Athena-compatible DDL statements for table schema documentation.
Extracted from the monolithic AthenaSchemaVectorizer for better maintainability.
"""

from typing import Any

from loguru import logger


class AthenaDDLGenerator:
    """
    Generates Athena DDL CREATE EXTERNAL TABLE statements from table metadata.

    This class encapsulates all logic for building proper Athena DDL syntax
    including column definitions, partitions, storage formats, and table properties.
    """

    def __init__(self):
        """Initialize the DDL generator."""
        logger.debug("AthenaDDLGenerator initialized")

    def generate_ddl(self, table_data: dict[str, Any]) -> str:
        """
        Generate Athena DDL CREATE EXTERNAL TABLE statement.

        Args:
            table_data: Dictionary containing table metadata with keys:
                - table_name: Name of the table
                - database_name: Name of the database
                - columns: List of column definitions
                - partition_keys: Optional partition columns
                - serde_info: Optional serialization info
                - input_format, output_format: Storage format info
                - location: S3 location
                - parameters: Table properties
                - description: Optional table description

        Returns:
            Complete DDL statement as string

        """
        table_name = table_data["table_name"]
        database_name = table_data["database_name"]

        ddl_parts = []
        ddl_parts.append(f"CREATE EXTERNAL TABLE `{database_name}`.`{table_name}` (")

        # Add columns
        column_definitions = self._build_column_definitions(
            table_data.get("columns", [])
        )
        ddl_parts.append(",\n".join(column_definitions))
        ddl_parts.append(")")

        # Add partition columns if present
        if table_data.get("partition_keys"):
            partition_ddl = self._build_partition_ddl(table_data["partition_keys"])
            ddl_parts.extend(partition_ddl)

        # Add storage format information
        storage_ddl = self._build_storage_format_ddl(table_data)
        ddl_parts.extend(storage_ddl)

        # Add location
        if location := table_data.get("location", ""):
            ddl_parts.append(f"LOCATION '{location}'")

        # Add table properties if present
        if parameters := table_data.get("parameters", {}):
            properties_ddl = self._build_table_properties_ddl(parameters)
            ddl_parts.extend(properties_ddl)

        ddl = "\n".join(ddl_parts) + ";"

        # Add description as comment if available
        if description := table_data.get("description", ""):
            ddl = f"-- Table Description: {description}\n{ddl}"

        return ddl

    def _build_column_definitions(self, columns: list) -> list:
        """Build column definition strings."""
        column_definitions = []
        for col in columns:
            col_def = f"  `{col['name']}` {col['type']}"
            if col.get("comment"):
                escaped_comment = col["comment"].replace("'", "''").replace("\n", " ")
                col_def += f" COMMENT '{escaped_comment}'"
            column_definitions.append(col_def)
        return column_definitions

    def _build_partition_ddl(self, partition_keys: list) -> list:
        """Build partition DDL parts."""
        ddl_parts = ["PARTITIONED BY ("]
        partition_definitions = []

        for pk in partition_keys:
            pk_def = f"  `{pk['name']}` {pk['type']}"
            if pk.get("comment"):
                escaped_comment = pk["comment"].replace("'", "''").replace("\n", " ")
                pk_def += f" COMMENT '{escaped_comment}'"
            partition_definitions.append(pk_def)

        ddl_parts.append(",\n".join(partition_definitions))
        ddl_parts.append(")")
        return ddl_parts

    def _build_storage_format_ddl(self, table_data: dict[str, Any]) -> list:
        """Build storage format DDL parts."""
        ddl_parts = []
        serde_info = table_data.get("serde_info", {})
        input_format = table_data.get("input_format", "")
        output_format = table_data.get("output_format", "")

        if input_format and output_format:
            ddl_parts.append(f"STORED AS INPUTFORMAT '{input_format}'")
            ddl_parts.append(f"OUTPUTFORMAT '{output_format}'")
        elif "parquet" in input_format.lower() or "parquet" in output_format.lower():
            ddl_parts.append("STORED AS PARQUET")
        elif serde_info.get("SerializationLibrary"):
            ddl_parts.extend(self._build_serde_ddl(serde_info))

        return ddl_parts

    def _build_serde_ddl(self, serde_info: dict[str, Any]) -> list:
        """Build SerDe DDL parts."""
        ddl_parts = ["ROW FORMAT SERDE"]
        ddl_parts.append(f"  '{serde_info['SerializationLibrary']}'")

        if serde_info.get("Parameters"):
            ddl_parts.append("WITH SERDEPROPERTIES (")
            serde_params = []
            for key, value in serde_info["Parameters"].items():
                serde_params.append(f"  '{key}' = '{value}'")
            ddl_parts.append(",\n".join(serde_params))
            ddl_parts.append(")")

        return ddl_parts

    def _build_table_properties_ddl(self, parameters: dict[str, Any]) -> list:
        """Build table properties DDL parts."""
        ddl_parts = ["TBLPROPERTIES ("]
        prop_definitions = []

        for key, value in parameters.items():
            prop_definitions.append(f"  '{key}' = '{value}'")

        ddl_parts.append(",\n".join(prop_definitions))
        ddl_parts.append(")")
        return ddl_parts
