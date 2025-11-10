"""
Prerequisite validation for RAG setup.

Validates database connections, AWS credentials (if needed), and RAG system components.
Supports multiple analytics backends through ANALYTICS_DB_URL.
"""

from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
from sqlalchemy import text


class PrerequisiteValidator:
    """Validates prerequisites for RAG setup with generic database support."""

    def __init__(self, athena_config: dict[str, Any], rag_config: dict[str, Any]):
        """
        Initialize the validator.

        Args:
            athena_config: Legacy Athena/AWS configuration dictionary (for backward compatibility)
            rag_config: RAG system configuration dictionary

        Note:
            The athena_config parameter is retained for backward compatibility but
            now primarily provides AWS credentials. The actual database connection
            is configured via ANALYTICS_DB_URL environment variable.

        """
        self.athena_config = athena_config  # Now primarily for AWS config
        self.rag_config = rag_config
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []

    def validate_all(self) -> tuple[bool, list[str], list[str]]:
        """
        Validate all prerequisites.

        Returns:
            Tuple of (success, errors, warnings)

        """
        logger.info("🔍 Validating prerequisites...")

        self.validation_errors = []
        self.validation_warnings = []

        validation_results = {
            "postgresql": self.validate_postgresql(),
            "rag_system": self.validate_rag_system(),
        }

        # Always validate analytics database (could be Athena, Postgres, Snowflake, etc.)
        validation_results["analytics_db"] = self.validate_analytics_database()

        # Only validate AWS credentials if using Athena
        import os

        analytics_url = os.getenv("ANALYTICS_DB_URL", "")
        if "athena://" in analytics_url:
            validation_results["aws_credentials"] = self.validate_aws_credentials()

        # Report validation results
        for component, status in validation_results.items():
            if status:
                logger.info(f"✅ {component.replace('_', ' ').title()}: OK")
            else:
                logger.error(f"❌ {component.replace('_', ' ').title()}: FAILED")

        all_valid = all(validation_results.values())

        if all_valid:
            logger.info("✅ All prerequisites validated successfully")
        else:
            logger.error("❌ Prerequisites validation failed")

        return all_valid, self.validation_errors, self.validation_warnings

    def validate_aws_credentials(self) -> bool:
        """
        Validate AWS credentials and permissions.

        Returns:
            True if valid, False otherwise

        """
        try:
            # Test basic AWS access
            session = boto3.Session(
                profile_name=self.athena_config.get("aws_profile", "default"),
                region_name=self.athena_config.get("region_name", "ap-southeast-2"),
            )

            # Test Glue access
            glue_client = session.client("glue")
            glue_client.get_databases()

            # Test Athena access
            athena_client = session.client("athena")
            athena_client.list_work_groups()

            logger.info("✓ AWS credentials and permissions validated")
            return True

        except NoCredentialsError:
            error = "AWS credentials not found or expired"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False
        except ClientError as e:
            error = f"AWS access error: {e}"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False
        except Exception as e:
            error = f"AWS validation error: {e}"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False

    def validate_analytics_database(self) -> bool:
        """
        Validate analytics database connection using ANALYTICS_DB_URL.

        Returns:
            True if valid, False otherwise

        """
        import os

        analytics_url = os.getenv("ANALYTICS_DB_URL")

        if not analytics_url:
            warning = "ANALYTICS_DB_URL not configured, analytics database unavailable"
            logger.warning(f"⚠️ {warning}")
            self.validation_warnings.append(warning)
            return True  # Not an error, just unavailable

        try:
            # Detect backend type
            backend_type = (
                analytics_url.split("://")[0] if "://" in analytics_url else "unknown"
            )
            logger.info(f"Validating {backend_type} analytics database...")

            # NOTE: s3_staging_dir is optional - workgroup's default will be used if not provided
            if backend_type == "athena":
                s3_staging_dir = self.athena_config.get("s3_staging_dir")
                if not s3_staging_dir:
                    warning = (
                        "ATHENA_S3_STAGING_DIR not configured - will use workgroup's default output location. "
                        "If connection fails, set ATHENA_S3_STAGING_DIR or configure workgroup output location in AWS."
                    )
                    logger.warning(f"⚠️ {warning}")
                    self.validation_warnings.append(warning)

            # Import and test AnalyticsConnector
            from src.connectors.analytics_backend import AnalyticsConnector

            backend = AnalyticsConnector(analytics_url)

            # Test basic connectivity
            databases = backend.list_databases()
            logger.info(f"✓ Analytics database connection validated ({backend_type})")
            logger.debug(f"Found {len(databases)} databases: {databases}")

            backend.close()
            return True

        except Exception as e:
            error = f"Analytics database validation failed: {e}"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False

    def validate_postgresql(self) -> bool:
        """
        Validate PostgreSQL connection and pgvector extension.

        Returns:
            True if valid, False otherwise

        """
        try:
            from src.connectors.database import DatabaseConnectionManager

            db_manager = DatabaseConnectionManager()

            # Test basic connection
            postgres_config = self.rag_config.get("postgres_config", {})
            engine = db_manager.create_postgres_engine(postgres_config)

            with engine.connect() as conn:
                # Test basic connection
                conn.execute(text("SELECT 1 as test"))

                # Check for pgvector extension
                vector_result = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )

                if not vector_result.fetchone():
                    warning = "pgvector extension not detected"
                    logger.warning(
                        "⚠️ pgvector extension not found - may need installation"
                    )
                    self.validation_warnings.append(warning)

            logger.info("✓ PostgreSQL connection validated")
            return True

        except Exception as e:
            error = f"PostgreSQL validation failed: {e}"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False

    def validate_rag_system(self) -> bool:
        """
        Validate RAG system components.

        Returns:
            True if valid, False otherwise

        """
        try:
            # Test embedding model loading
            try:
                from langchain_aws import BedrockEmbeddings
            except ImportError:
                pass

            embedding_provider = self.rag_config.get("embedding_provider", "bedrock")
            if embedding_provider != "bedrock":
                error = "CONSTRAINT VIOLATION: Only 'bedrock' embedding provider is permitted"
                raise ValueError(error)

            # Initialize Bedrock embeddings ONLY
            try:
                bedrock_embedding_model = self.rag_config.get(
                    "bedrock_embedding_model", "amazon.titan-embed-text-v2:0"
                )

                # Use AWS configuration (backward compatible)
                import os

                aws_region = os.getenv("AWS_DEFAULT_REGION") or self.athena_config.get(
                    "region_name", "ap-southeast-2"
                )
                aws_profile = os.getenv("AWS_PROFILE") or self.athena_config.get(
                    "aws_profile"
                )

                embeddings = BedrockEmbeddings(
                    model_id=bedrock_embedding_model,
                    region_name=aws_region,
                    credentials_profile_name=aws_profile,
                )
                logger.info(
                    f"✓ Initialized Bedrock embeddings: {bedrock_embedding_model}"
                )

            except Exception as e:
                error = f"Failed to initialize Bedrock embeddings: {e}"
                logger.error(error)
                raise RuntimeError(
                    f"CONSTRAINT VIOLATION: Bedrock embedding initialization failed: {e}"
                ) from e

            # Test embedding generation
            test_embedding = embeddings.embed_query("test query")

            if len(test_embedding) == 0:
                error = "Embedding generation failed"
                logger.error(f"❌ {error}")
                self.validation_errors.append(error)
                return False

            logger.info("✓ RAG system components validated")
            return True

        except Exception as e:
            error = f"RAG system validation failed: {e}"
            logger.error(f"❌ {error}")
            self.validation_errors.append(error)
            return False
