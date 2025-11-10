#!/usr/bin/env python3
"""
Initialise RAG cardinality database schema.

Clean, idempotent schema creation without embedded SQL.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import psycopg
from loguru import logger

from src.config.database_config import POSTGRES_CONFIG


def init_schema():
    """Create or update cardinality schema from SQL file."""
    schema_file = Path(__file__).parent.parent / "schemas" / "pgvector-cardinality.sql"

    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return False

    # Read schema SQL
    with open(schema_file) as f:
        sql = f.read()
        # Remove database connection commands (handled via connection string)
        # Handle any database switching commands for current database
        db_name = POSTGRES_CONFIG.get("database", "instantinsight")
        sql = sql.replace(f"\\c {db_name};", "")
        sql = sql.replace("\\c instantinsight;", "")  # Legacy cleanup

    # Connect and execute
    try:
        conn = psycopg.connect(
            host=POSTGRES_CONFIG["host"],
            port=POSTGRES_CONFIG["port"],
            dbname=POSTGRES_CONFIG["database"],
            user=POSTGRES_CONFIG["user"],
            password=POSTGRES_CONFIG["password"],
        )

        with conn.cursor() as cur:
            cur.execute(sql)

        conn.commit()
        conn.close()

        logger.info("Schema initialised successfully")
        return True

    except psycopg.errors.DuplicateTable:
        logger.info("Schema already exists")
        return True

    except Exception as e:
        logger.error(f"Schema initialisation failed: {e}")
        return False


if __name__ == "__main__":
    # Configure minimal logging
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            }
        ]
    )

    success = init_schema()
    sys.exit(0 if success else 1)
