#!/usr/bin/env python3
"""
Create RAG cardinality tables for tracking categorical values.

Creates two tables: rag_cardinality and rag_cardinality_columns.
rag_cardinality_columns tracks which columns have been added/analysed for cardinality.
rag_cardinality stores the categorical values with embeddings.

Usage:
    python scripts/create_rag_cardinality.py --config config/high_cardinality_columns.yaml
"""

import argparse
import os
import sys
import time

import boto3
import psycopg
import yaml
from langchain_aws import BedrockEmbeddings
from loguru import logger
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config.database_config import ATHENA_CONFIG, POSTGRES_CONFIG


class ConfigBasedProcessor:
    """Process specific columns based on YAML configuration."""

    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.athena_client = boto3.client("athena")
        self.database_name = ATHENA_CONFIG.get("database", "text_to_sql")

        # Initialize database connection
        self._connect_database()

        # Initialize embeddings
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", region_name="ap-southeast-2"
        )

        logger.info(f"Loaded configuration with {len(self.config['columns'])} columns")

    def _load_config(self, config_path: str) -> dict:
        """Load and validate YAML configuration."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "columns" not in config:
            raise ValueError("Configuration must contain 'columns' section")

        return config

    def _connect_database(self):
        """Connect to PostgreSQL database."""
        connection_string = (
            f"host={POSTGRES_CONFIG['host']} "
            f"port={POSTGRES_CONFIG['port']} "
            f"dbname={POSTGRES_CONFIG['database']} "
            f"user={POSTGRES_CONFIG['user']} "
            f"password={POSTGRES_CONFIG['password']}"
        )

        self.db_connection = psycopg.connect(connection_string, row_factory=dict_row)
        register_vector(self.db_connection)
        logger.info("Connected to PostgreSQL database")

    def process_all(self):
        """Process all columns in configuration."""
        for column_config in self.config["columns"]:
            table = column_config["table"]
            column = column_config["column"]

            logger.info(f"Processing {table}.{column}")

            # 1. Register column as categorical
            column_id = self._register_column(table, column)

            # 2. Extract values from Athena
            values = self._extract_values(table, column)

            if values:
                # 3. Store values in database
                self._store_values(column_id, table, column, values)

                # 4. Generate embeddings
                self._generate_embeddings(values)

                logger.info(
                    f"Completed processing {table}.{column}: {len(values)} values"
                )
            else:
                logger.warning(f"No values found for {table}.{column}")

    def _register_column(self, table: str, column: str) -> int:
        """Register column as categorical in rag_cardinality_columns."""
        with self.db_connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_cardinality_columns 
                (schema_name, table_name, column_name, column_type, is_categorical,
                 cardinality_tier, detection_method, confidence_score, last_analysed)
                VALUES (%s, %s, %s, 'string', TRUE, 'high', 'manual', 1.0, NOW())
                ON CONFLICT (schema_name, table_name, column_name)
                DO UPDATE SET
                    is_categorical = TRUE,
                    cardinality_tier = 'high',
                    detection_method = 'manual',
                    last_analysed = NOW(),
                    updated_at = NOW()
                RETURNING id
            """,
                (self.database_name, table, column),
            )

            result = cur.fetchone()
            self.db_connection.commit()
            return result["id"]

    def _extract_values(self, table: str, column: str) -> list[dict]:
        """Extract unique values from Athena table."""
        # Handle column names with spaces or special characters
        quoted_column = f'"{column}"' if " " in column or "/" in column else column

        query = f"""
        SELECT {quoted_column} as value, COUNT(*) as frequency
        FROM {self.database_name}.{table}
        WHERE {quoted_column} IS NOT NULL
          AND TRIM({quoted_column}) != ''
        GROUP BY {quoted_column}
        ORDER BY frequency DESC
        """

        try:
            # Execute Athena query
            execution_params = {
                "QueryString": query,
                "WorkGroup": ATHENA_CONFIG.get("work_group", "primary"),
            }

            # Add S3 location if configured
            s3_location = ATHENA_CONFIG.get("s3_staging_dir")
            if s3_location:
                execution_params["ResultConfiguration"] = {
                    "OutputLocation": s3_location
                }

            response = self.athena_client.start_query_execution(**execution_params)
            query_id = response["QueryExecutionId"]

            # Wait for completion
            for _ in range(60):
                response = self.athena_client.get_query_execution(
                    QueryExecutionId=query_id
                )
                status = response["QueryExecution"]["Status"]["State"]

                if status == "SUCCEEDED":
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    logger.error(
                        f"Query failed: {response['QueryExecution']['Status']}"
                    )
                    return []
                time.sleep(1)

            # Get results with pagination to handle large result sets
            results = []
            next_token = None

            while True:
                # Get results (with pagination token if available)
                get_results_params = {"QueryExecutionId": query_id}
                if next_token:
                    get_results_params["NextToken"] = next_token

                response = self.athena_client.get_query_results(**get_results_params)

                # Parse results from this page
                rows = response["ResultSet"]["Rows"]
                start_idx = (
                    1 if next_token is None else 0
                )  # Skip header only on first page

                for row in rows[start_idx:]:
                    value = row["Data"][0].get("VarCharValue", "")
                    frequency = int(row["Data"][1].get("VarCharValue", 0))
                    if value:
                        results.append({"value": value, "frequency": frequency})

                # Check if there are more pages
                next_token = response.get("NextToken")
                if not next_token:
                    break

                # Log progress for large datasets
                if len(results) % 5000 == 0:
                    logger.info(f"Fetched {len(results)} results so far...")

            logger.info(f"Extracted {len(results)} unique values from {table}.{column}")
            return results

        except Exception as e:
            logger.error(f"Failed to extract values: {e}")
            return []

    def _store_values(
        self, column_id: int, table: str, column: str, values: list[dict]
    ):
        """Store extracted values in rag_cardinality table."""
        with self.db_connection.cursor() as cur:
            for item in values:
                value = item["value"]
                frequency = item["frequency"]

                # Normalize value for matching
                value_norm = value.lower().strip()

                cur.execute(
                    """
                    INSERT INTO rag_cardinality
                    (column_id, schema_name, table_name, column_name, 
                     category, category_norm, frequency, embedding_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                    ON CONFLICT (schema_name, table_name, column_name, category)
                    DO UPDATE SET
                        frequency = EXCLUDED.frequency,
                        updated_at = NOW()
                """,
                    (
                        column_id,
                        self.database_name,
                        table,
                        column,
                        value,
                        value_norm,
                        frequency,
                    ),
                )

            self.db_connection.commit()
            logger.info(f"Stored {len(values)} values in database")

    def _generate_embeddings(self, values: list[dict]):
        """Generate and store embeddings for values."""
        # Get values needing embeddings from database
        with self.db_connection.cursor() as cur:
            cur.execute(
                """
                SELECT id, category, category_norm
                FROM rag_cardinality
                WHERE embedding_status = 'pending'
                  AND embedding IS NULL
                ORDER BY frequency DESC
            """
            )
            pending_values = cur.fetchall()

        if not pending_values:
            logger.info("No values need embeddings")
            return

        # Process in batches
        batch_size = 100
        total_embedded = 0

        for i in range(0, len(pending_values), batch_size):
            batch = pending_values[i : i + batch_size]
            texts = [v["category"] for v in batch]

            try:
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)

                # Store embeddings
                with self.db_connection.cursor() as cur:
                    for j, value_info in enumerate(batch):
                        if j < len(embeddings):
                            cur.execute(
                                """
                                UPDATE rag_cardinality
                                SET embedding = %s,
                                    embedding_status = 'completed',
                                    updated_at = NOW()
                                WHERE id = %s
                            """,
                                (embeddings[j], value_info["id"]),
                            )

                    self.db_connection.commit()
                    total_embedded += len(batch)
                    logger.info(
                        f"Generated embeddings for batch {i // batch_size + 1} ({len(batch)} values)"
                    )

                # Rate limiting
                if i + batch_size < len(pending_values):
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")

        logger.info(f"Generated embeddings for {total_embedded} values")

    def close(self):
        """Clean up resources."""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")


def main():
    """Execute main cardinality processing workflow."""
    parser = argparse.ArgumentParser(
        description="Simplified High-Cardinality Processing with YAML Configuration"
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--mode",
        choices=["all", "embed"],
        default="all",
        help="Processing mode (default: all)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "level": log_level,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            }
        ]
    )

    # Check config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Process based on configuration
    processor = None
    try:
        processor = ConfigBasedProcessor(args.config)

        if args.mode == "all":
            processor.process_all()
        elif args.mode == "embed":
            logger.info("Embed-only mode - generating embeddings for existing values")
            processor._generate_embeddings([])  # Will process pending values

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if processor:
            processor.close()


if __name__ == "__main__":
    main()
