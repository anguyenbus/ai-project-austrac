"""Utility functions for table agent processing and DDL parsing."""

import re


def extract_table_info(ddl_text):
    """
    Extract table name and columns from SQL DDL CREATE TABLE statement.

    Args:
        ddl_text (str): The SQL DDL statement

    Returns:
        dict: JSON object with table_name and columns

    """
    # Extract table name using regex
    table_pattern = r"CREATE\s+EXTERNAL\s+TABLE\s+`([^`]+)`\.`([^`]+)`"
    table_match = re.search(table_pattern, ddl_text, re.IGNORECASE)

    if not table_match:
        # Fallback pattern for tables without database prefix
        table_pattern = r"CREATE\s+EXTERNAL\s+TABLE\s+`([^`]+)`"
        table_match = re.search(table_pattern, ddl_text, re.IGNORECASE)
        if table_match:
            table_name = table_match.group(1)
        else:
            table_name = "unknown"
    else:
        # Use the table name (second group) if database.table format
        table_name = table_match.group(2)

    # Extract columns from the DDL
    # Find the section between the first ( and the first )
    columns_pattern = r"\(\s*\n(.*?)\n\s*\)"
    columns_match = re.search(columns_pattern, ddl_text, re.DOTALL)

    columns = {}

    if columns_match:
        columns_text = columns_match.group(1)

        # Split by lines and process each column definition
        column_lines = columns_text.split("\n")

        for line in column_lines:
            line = line.strip()
            if line and not line.startswith("--"):  # Skip empty lines and comments
                # Remove trailing comma
                line = line.rstrip(",")

                # Extract column name and type using regex
                column_pattern = r"`([^`]+)`\s+([^,\s]+)"
                column_match = re.search(column_pattern, line)

                if column_match:
                    column_name = column_match.group(1)
                    column_type = column_match.group(2)
                    columns[column_name] = column_type

    # Create the result dictionary
    result = {"table_name": table_name, "columns": columns}

    return result


def format_table_info_text(ddl_text, metadata=None):
    """
    Extract table info and format as text with bullet points and join safety info.

    Args:
        ddl_text (str): The SQL DDL statement
        metadata (dict): Optional metadata with analyser and join information

    Returns:
        str: Formatted text with table name, columns, and join safety notes

    """
    # Get the parsed info
    info = extract_table_info(ddl_text)

    # Format as text
    result_text = f"Table Name: {info['table_name']}\n"

    for column_name, column_type in info["columns"].items():
        result_text += f"* {column_name}: {column_type}\n"

    # Add join safety note based on metadata
    if metadata and metadata.get("is_analyser"):
        safe_tables = metadata.get("safe_to_join_with", [])
        if safe_tables:
            result_text += f"NOTE: CAN JOIN WITH {', '.join(safe_tables)}"
        else:
            result_text += "NOTE: DO NOT JOIN"

    return result_text.rstrip()  # Remove trailing newline


def filter_schema_by_selected_tables(schema_text, selected_tables):
    """
    Filter schema_text to keep only tables that exist in selected_tables.

    Args:
        schema_text (str): The original schema text
        selected_tables (list): List of table names to keep

    Returns:
        str: Filtered schema text containing only selected tables and their columns

    """
    lines = schema_text.strip().split("\n")
    filtered_lines = []
    include_current_table = False

    for line in lines:
        line = line.strip()

        # Check if this is a table name line
        if line.startswith("Table Name:"):
            # Extract table name
            table_name = line.split("Table Name:")[1].strip()

            if table_name in selected_tables:
                filtered_lines.append(line)
                include_current_table = True  # This was missing!
            else:
                include_current_table = False

        # Check if this is a column line (starts with *)
        elif line.startswith("*") and include_current_table:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
