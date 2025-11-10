"""
Local RAG Database Schema Configuration Module.

This module provides comprehensive database schema management for local RAG (Retrieval-Augmented Generation)
systems. It handles loading, parsing, and generating database schemas from multiple sources without requiring
external API keys.

Key Features:
    - Multi-source schema loading (SQL files, CSV files, custom paths)
    - Automatic schema inference from CSV data
    - Smart column type detection with pandas integration
    - Priority-based schema resolution
    - Support for any SQL database with proper DDL files
    - Fallback mechanisms for robust operation

Supported Data Sources:
    1. Direct SQL files containing CREATE TABLE statements
    2. CSV directories with automatic schema inference
    3. Custom file paths and directories
    4. Environment-based database selection

Schema Loading Priority:
    1. Direct sql_file_path parameter
    2. CSV directory (csv_dir parameter)
    3. SQL file in schemas/ directory (by database name)
    4. CSV files in datasets/{database_name}/ directory
    5. Fallback error handling

Usage Examples:
    Basic usage with default database:
        >>> ddl = load_ddl_from_file()
        >>> schema = load_database_schema()

    Load specific database schema:
        >>> ddl = load_ddl_from_file(database_name="your_db_name")
        >>> ddl = load_ddl_from_file(sql_file_path="/path/to/schema.sql")

    Generate schema from CSV directory:
        >>> ddl = load_ddl_from_csv_directory("./data/csv_files")
        >>> ddl = load_ddl_from_file(csv_dir="./datasets/mydata")

    Get supported databases:
        >>> databases = get_supported_databases()
        >>> print(databases)

Dependencies:
    - pathlib, re, os, csv (standard library)
    - pandas (optional, for enhanced CSV type inference)
    - typing (for type hints)

Environment Variables:
    - DEFAULT_DATABASE: Sets default database name (default: "your_db_name")

File Structure:
    - schemas/: Directory for SQL schema files
    - datasets/: Directory for CSV data files organized by database name

Author: Local RAG Configuration System
Version: 1.0
"""

import csv
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

# Optional pandas import for CSV schema inference
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def get_default_database_name() -> str:
    """Get the default database name from environment or fallback."""
    return os.getenv("POSTGRES_DATABASE", "instantinsight")


def infer_column_type_from_values(values: list[str], column_name: str) -> str:
    """
    Infer SQL column type from sample values (fallback when pandas not available).

    Args:
        values: List of string values from CSV
        column_name: Name of the column for context

    Returns:
        SQL column type string

    """
    # Filter out empty values
    clean_values = [v.strip() for v in values if v.strip()]

    if not clean_values:
        return "TEXT"

    # Check for common ID patterns
    if column_name.lower().endswith("_id") or column_name.lower() == "id":
        # Try to parse as integer
        try:
            # Just check if all values can be converted to int
            for v in clean_values:
                int(v)
            return "INTEGER NOT NULL"
        except ValueError:
            return "VARCHAR(50) NOT NULL"

    # Try to determine type from values
    all_integers = True
    all_floats = True
    all_booleans = True
    max_length = 0

    for value in clean_values:
        # Check integer
        try:
            int(value)
        except ValueError:
            all_integers = False

        # Check float
        try:
            float(value)
        except ValueError:
            all_floats = False

        # Check boolean
        if value.lower() not in ["true", "false", "1", "0", "yes", "no"]:
            all_booleans = False

        # Track max string length
        max_length = max(max_length, len(value))

    # Determine type based on analysis
    if all_integers:
        return "INTEGER"
    elif all_floats:
        return "REAL"
    elif all_booleans:
        return "BOOLEAN"
    else:
        # String type based on length
        if max_length <= 10:
            return f"VARCHAR({max(max_length + 5, 20)})"
        elif max_length <= 50:
            return f"VARCHAR({max_length + 10})"
        elif max_length <= 255:
            return f"VARCHAR({max_length + 20})"
        else:
            return "TEXT"


def infer_column_type(series_or_values, column_name: str) -> str:
    """
    Infer SQL column type from pandas Series data or list of values.

    Args:
        series_or_values: Pandas series with sample data OR list of string values
        column_name: Name of the column for context

    Returns:
        SQL column type string

    """
    if not HAS_PANDAS or not hasattr(series_or_values, "dropna"):
        # Fall back to simple value-based inference
        if isinstance(series_or_values, list):
            return infer_column_type_from_values(series_or_values, column_name)
        else:
            return "TEXT"

    # Use pandas-based inference
    series = series_or_values
    # Remove null values for type inference
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return "TEXT"

    # Check for common ID patterns
    if column_name.lower().endswith("_id") or column_name.lower() == "id":
        if clean_series.dtype in ["int64", "int32"]:
            return "INTEGER NOT NULL"
        return "VARCHAR(50) NOT NULL"

    # Infer based on pandas dtype
    if pd.api.types.is_integer_dtype(clean_series):
        max_val = clean_series.max()
        if max_val < 32767:
            return "SMALLINT"
        elif max_val < 2147483647:
            return "INTEGER"
        else:
            return "BIGINT"

    elif pd.api.types.is_float_dtype(clean_series):
        return "REAL"

    elif pd.api.types.is_bool_dtype(clean_series):
        return "BOOLEAN"

    elif pd.api.types.is_datetime64_any_dtype(clean_series):
        return "TIMESTAMP"

    elif pd.api.types.is_string_dtype(clean_series) or clean_series.dtype == "object":
        # Analyze string lengths
        max_length = clean_series.astype(str).str.len().max()
        if max_length <= 10:
            return f"VARCHAR({max(max_length + 5, 20)})"
        elif max_length <= 50:
            return f"VARCHAR({max_length + 10})"
        elif max_length <= 255:
            return f"VARCHAR({max_length + 20})"
        else:
            return "TEXT"

    else:
        return "TEXT"


def generate_create_table_from_csv(csv_path: str | Path, table_name: str = None) -> str:
    """
    Generate CREATE TABLE statement from CSV file.

    Args:
        csv_path: Path to CSV file
        table_name: Override table name (defaults to filename)

    Returns:
        CREATE TABLE SQL statement

    """
    csv_path = Path(csv_path)

    if table_name is None:
        table_name = csv_path.stem.lower()

    try:
        if HAS_PANDAS:
            # Use pandas for better type inference
            df_sample = pd.read_csv(csv_path, nrows=1000)  # Sample first 1000 rows

            if df_sample.empty:
                return f"-- Could not read data from {csv_path}\n"

            # Generate column definitions using pandas
            columns = []
            for col_name in df_sample.columns:
                # Clean column name for SQL
                clean_col_name = re.sub(r"[^a-zA-Z0-9_]", "_", col_name.lower())
                col_type = infer_column_type(df_sample[col_name], clean_col_name)
                columns.append(f"    {clean_col_name} {col_type}")
        else:
            # Fallback to standard CSV reader
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)

                # Read sample rows for type inference
                sample_rows = []
                for i, row in enumerate(reader):
                    if i >= 100:  # Sample first 100 rows
                        break
                    sample_rows.append(row)

                if not sample_rows:
                    return f"-- No data rows found in {csv_path}\n"

                # Generate column definitions
                columns = []
                for i, col_name in enumerate(headers):
                    # Clean column name for SQL
                    clean_col_name = re.sub(r"[^a-zA-Z0-9_]", "_", col_name.lower())

                    # Get sample values for this column
                    col_values = [row[i] if i < len(row) else "" for row in sample_rows]
                    col_type = infer_column_type(col_values, clean_col_name)
                    columns.append(f"    {clean_col_name} {col_type}")

        # Join columns with commas and newlines
        joined_columns = ",\n".join(columns)

        # Build CREATE TABLE statement
        create_statement = f"""CREATE TABLE {table_name} (
{joined_columns}
);"""

        return create_statement

    except Exception as e:
        return f"-- Error processing {csv_path}: {str(e)}\n"


def load_ddl_from_csv_directory(csv_dir: str | Path, database_name: str = None) -> str:
    """
    Generate DDL from a directory of CSV files.

    Args:
        csv_dir: Directory containing CSV files
        database_name: Name for the database schema

    Returns:
        Complete DDL with CREATE TABLE statements for all CSV files

    """
    csv_dir = Path(csv_dir)
    db_name = database_name or csv_dir.name or "csv_database"

    if not csv_dir.exists() or not csv_dir.is_dir():
        return f"-- Directory not found: {csv_dir}\n"

    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        return f"-- No CSV files found in: {csv_dir}\n"

    ddl_statements = [
        f"-- {db_name.upper()} Database Schema (Generated from CSV files)\n"
    ]

    for csv_file in sorted(csv_files):
        table_name = csv_file.stem.lower()
        create_statement = generate_create_table_from_csv(csv_file, table_name)
        ddl_statements.append(create_statement)

    return "\n\n".join(ddl_statements)


def load_ddl_from_file(
    database_name: str | None = None,
    databases_dir: str | None = None,
    csv_dir: str | None = None,
) -> str:
    """
    Load DDL schema from a SQL file, CSV directory, or fallback.

    This extracts CREATE TABLE statements for schema understanding.

    Priority order:
    1. CSV directory (csv_dir)
    2. SQL file in databases directory
    3. Fallback schemas

    Args:
        database_name: Name of the database (e.g., 'your_db_name', 'chinook', 'sakila')
        databases_dir: Custom databases directory path
        csv_dir: Directory containing CSV files for schema inference

    Returns:
        DDL schema as string with CREATE TABLE statements

    """
    # Priority 1: CSV directory
    if csv_dir is not None:
        print(f"📊 Generating schema from CSV files in: {csv_dir}")
        return load_ddl_from_csv_directory(csv_dir, database_name)

    # Priority 2: SQL file in databases directory
    db_name = database_name or get_default_database_name()

    # Check for CSV directory first (datasets/{db_name}/)
    project_root = Path(__file__).parent.parent.parent
    csv_data_dir = project_root / "datasets" / db_name
    if csv_data_dir.exists() and csv_data_dir.is_dir():
        csv_files = list(csv_data_dir.glob("*.csv"))
        if csv_files:
            print(f"📊 Found CSV files for {db_name}, generating schema from data")
            return load_ddl_from_csv_directory(csv_data_dir, db_name)

    # Fall back to empty DDL - modern system uses dynamic Athena extraction
    logger.warning(
        f"No schema source found for {db_name}. Using dynamic Athena extraction instead."
    )
    return ""


def get_supported_databases() -> dict[str, str]:
    """Get list of supported database schemas."""
    return {
        "any_sql": "Any database with SQL file in schemas/ directory",
        "any_csv": "Any database with CSV files in datasets/ directory",
        "custom": "Custom SQL file path or CSV directory",
    }


def load_database_schema(database_name: str = None) -> dict[str, Any]:
    """
    Load complete database schema information including DDL and metadata.

    Returns:
        Dictionary with DDL, database info, and metadata

    """
    db_name = database_name or get_default_database_name()
    ddl = load_ddl_from_file(database_name=db_name)

    # Determine source type
    source_type = "unknown"
    if "Generated from CSV files" in ddl:
        source_type = "csv"
    elif ddl.startswith("--") and ("not found" in ddl or "Error" in ddl):
        source_type = "error"
    else:
        source_type = "sql"

    return {
        "database_name": db_name,
        "ddl": ddl,
        "table_count": (
            len(ddl.split("CREATE TABLE")) - 1 if "CREATE TABLE" in ddl else 0
        ),
        "supported_databases": get_supported_databases(),
        "source": source_type,
    }


# Load DDL from the source file (defaults to environment or "your_db_name")
DDL = load_ddl_from_file()
