"""
Factory for creating configured SQL engine and query executor.

Handles dependency injection and configuration wiring.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.agents.strand_agents.query.clarifier import ClarificationAgent
from src.agents.strand_agents.query.normalizer import QueryNormalizer
from src.agents.strand_agents.schema.column_mapper import (
    ColumnAgent,
    ColumnConfig,
)
from src.agents.strand_agents.schema.filter_builder import (
    FilterConfig,
    FilteringAgent,
)
from src.agents.strand_agents.schema.table_selector import TableAgent
from src.agents.strand_agents.schema.validator import SchemaValidatorAgent
from src.config.database_config import (
    ATHENA_CONFIG,
    DATABASE_URL,
    POSTGRES_CONFIG,
    RAG_CONFIG,
)
from src.rag.backends import PgVectorBackend, PgVectorBackendConfig
from src.rag.pgvector_rag import PgvectorRAG

from .query_executor import QueryExecutor, TestExecutor, UniversalExecutor
from .sql_engine import SQLEngine


class EngineFactory:
    """
    Factory for creating configured SQL engine with proper dependencies.

    Responsibilities:
    - Read configuration
    - Wire dependencies
    - Return configured engine and executor
    """

    @staticmethod
    def create_sql_engine(
        config: dict = None,
    ) -> tuple[SQLEngine | None, str | None]:
        """
        Create SQL engine with injected dependencies.

        Args:
            config: Configuration dictionary (uses default if None)

        Returns:
            Tuple of (SQLEngine, error_message)

        """
        try:
            # Load configuration
            if config is None:
                config = EngineFactory._load_default_config()

            # Create RAG instance
            rag_instance = EngineFactory._create_rag_instance(config)
            if not rag_instance:
                return None, "Failed to create RAG instance"

            # Create agents
            agents = EngineFactory._create_agents(rag_instance, config)
            if not agents:
                return None, "Failed to create agents"

            # Create SQL engine with injected dependencies
            engine = SQLEngine(
                rag_instance=rag_instance,
                table_agent=agents["table_agent"],
                schema_validator=agents["schema_validator"],
                clarification_agent=agents["clarification_agent"],
                filtering_agent=agents.get("filtering_agent"),
                column_agent=agents.get("column_agent"),
                query_normalizer=agents.get("query_normalizer"),
            )

            logger.info("✅ SQL engine created successfully")
            return engine, None

        except Exception as e:
            error_msg = f"Failed to create SQL engine: {e}"
            logger.error(error_msg)
            return None, error_msg

    @staticmethod
    def create_query_executor(
        config: dict = None, test_mode: bool = False
    ) -> tuple[QueryExecutor | None, str | None]:
        """
        Create query executor with Ibis backend.

        Args:
            config: Configuration dictionary
            test_mode: If True, returns test executor

        Returns:
            Tuple of (QueryExecutor, error_message)

        """
        try:
            if test_mode:
                return TestExecutor(), None

            if config is None:
                config = EngineFactory._load_default_config()

            # Use universal ANALYTICS_DB_URL
            analytics_db_url = config.get("analytics_db_url")

            if not analytics_db_url:
                return None, "ANALYTICS_DB_URL not configured"

            executor = UniversalExecutor(analytics_db_url)
            logger.info(
                f"✅ UniversalExecutor created ({executor.connector.backend_type})"
            )
            return executor, None

        except Exception as e:
            error_msg = f"Failed to create query executor: {e}"
            logger.error(error_msg)
            return None, error_msg

    @staticmethod
    def _load_default_config() -> dict:
        """Load default configuration."""
        try:
            # Setup imports
            project_root = Path(__file__).parent.parent.parent
            sys.path.append(str(project_root))
            load_dotenv()

            return {**RAG_CONFIG, "athena_config": ATHENA_CONFIG}

        except Exception as e:
            logger.error(f"Failed to load default config: {e}")
            return {}

    @staticmethod
    def _create_rag_instance(config: dict):
        """Create PgvectorRAG instance."""
        try:
            logger.info("Creating PgvectorRAG instance...")
            pool_config = config.get("pgvector_pool", {}) if config else {}
            backend_settings = config.get("pgvector_backend", {}) if config else {}
            sqlalchemy_url = config.get("sqlalchemy_database_url", DATABASE_URL)

            backend = PgVectorBackend(
                PgVectorBackendConfig(
                    db_url=backend_settings.get("db_url", sqlalchemy_url),
                    table_name=backend_settings.get(
                        "table_name", "instantinsight_pgvector_internal"
                    ),
                    schema=backend_settings.get("schema", "ai"),
                    pool_size=pool_config.get("pool_size", 5),
                    max_overflow=pool_config.get("max_overflow", 10),
                )
            )

            rag_instance = PgvectorRAG(
                connection_string=config.get("database_url", DATABASE_URL)
                if config
                else DATABASE_URL,
                bedrock_embedding_model=config.get(
                    "bedrock_embedding_model", "amazon.titan-embed-text-v2:0"
                )
                if config
                else "amazon.titan-embed-text-v2:0",
                backend=backend,
                pool_config=pool_config,
                backend_config={
                    **backend_settings,
                    "db_url": backend_settings.get("db_url", sqlalchemy_url),
                },
            )

            if not rag_instance.connect_to_database():
                logger.warning("Failed to connect PgvectorRAG to database")
                return None

            if not rag_instance.initialize_embeddings():
                logger.warning("Failed to initialize PgvectorRAG embeddings")
                return None

            logger.info("✅ PgvectorRAG instance created")
            return rag_instance

        except Exception as e:
            logger.error(f"Failed to create RAG instance: {e}")
            return None

    @staticmethod
    def _create_agents(rag_instance, config: dict) -> dict | None:
        """Create agent instances."""
        try:
            agents = {}

            # Core agents (required)
            table_agent_config = {
                "max_selected_tables": 1,
                "min_confidence_threshold": 0.5,
                "vector_search_k": 10,
                "include_related_tables": False,
                "allow_unsafe_joins": False,
            }
            agents["table_agent"] = TableAgent(
                pgvector_rag=rag_instance,
                config=table_agent_config,
            )

            agents["schema_validator"] = SchemaValidatorAgent(rag_backend=rag_instance)
            agents["clarification_agent"] = ClarificationAgent()
            agents["query_normalizer"] = QueryNormalizer()

            # Optional agents
            try:
                agents["filtering_agent"] = FilteringAgent(config=FilterConfig())

                agents["column_agent"] = ColumnAgent(
                    config=ColumnConfig(
                        postgres_config=POSTGRES_CONFIG,
                        aws_region=ATHENA_CONFIG.get("region_name", "ap-southeast-2"),
                    ),
                    rag_instance=None,  # Will be set by engine if needed
                )
            except Exception as e:
                logger.warning(f"Failed to create optional agents: {e}")

            logger.info("✅ Agents created successfully")
            return agents

        except Exception as e:
            logger.error(f"Failed to create agents: {e}")
            return None
