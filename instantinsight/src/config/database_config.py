"""Database configuration for local RAG system."""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env in multiple locations
    # 1. Project root (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent / ".env"
    # 2. Current working directory
    cwd_env = Path.cwd() / ".env"
    # 3. Parent directory of this file
    parent_env = Path(__file__).parent / ".env"

    env_loaded = False
    for env_path in [project_root, cwd_env, parent_env]:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment variables from {env_path}")
            env_loaded = True
            break

    if not env_loaded:
        # Try to load from current directory (will search for .env automatically)
        load_dotenv()
        print("✓ Loaded environment variables from default .env search")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Could not load .env file: {e}")


# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "database": os.getenv("POSTGRES_DATABASE", "instantinsight"),
}

# Database URL (standard format for most libraries including SQLAlchemy, etc.)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}",
)


def _build_sqlalchemy_url(database_url: str) -> str:
    """Ensure the SQLAlchemy URL uses the psycopg driver when available."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return database_url


SQLALCHEMY_DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URL", _build_sqlalchemy_url(DATABASE_URL)
)


# AWS Bedrock Configuration
BEDROCK_CONFIG = {
    "aws_profile": os.getenv(
        "AWS_PROFILE"
    ),  # AWS profile to use (optional, uses IAM role if None)
    "aws_region": os.getenv("AWS_REGION", "ap-southeast-2"),  # AWS region
    "model": os.getenv(
        "BEDROCK_MODEL", "apac.anthropic.claude-sonnet-4-20250514-v1:0"
    ),  # Bedrock model ID - using Claude Sonnet 4
    "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "4000")),
    "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.1")),
}

# Knowledge Graph Configuration (Neo4j removed)
KNOWLEDGE_GRAPH_CONFIG = {
    "enabled": False,  # Neo4j support removed
    "schema_extraction": {
        "enabled": False,
        "max_depth": 3,
        "include_indexes": True,
        "include_constraints": True,
    },
    "graph_algorithms": {
        "shortest_path_enabled": False,
        "centrality_enabled": False,
        "community_detection_enabled": False,
    },
}


# RAG System Configuration
RAG_CONFIG = {
    # Updated to use pgvector exclusively
    "vectorstore_type": "pgvector",  # Only pgvector supported
    # Embedding configuration - BEDROCK TITAN ONLY
    "embedding_provider": "bedrock",  # ONLY OPTION: bedrock
    "embedding_model": "amazon.titan-embed-text-v2:0",  # DEPRECATED - use bedrock_embedding_model
    "bedrock_embedding_model": "amazon.titan-embed-text-v2:0",  # ONLY OPTION: Titan v2
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    # Conversation settings
    "conversation_memory": os.getenv("CONVERSATION_MEMORY", "true").lower() == "true",
    "max_conversation_history": int(os.getenv("MAX_CONVERSATION_HISTORY", "10")),
    # Hybrid retrieval settings (Neo4j removed)
    "hybrid_retrieval": {
        "enabled": False,  # Neo4j support removed
        "vector_weight": 1.0,  # Pure vector retrieval only
        "graph_weight": 0.0,
        "max_vector_results": int(os.getenv("MAX_VECTOR_RESULTS", "10")),
        "max_graph_results": 0,
    },
    # Database connection for SQL execution and data extraction
    "database_url": DATABASE_URL,
    "sqlalchemy_database_url": SQLALCHEMY_DATABASE_URL,
    "postgres_config": POSTGRES_CONFIG,
    "pgvector_pool": {
        "pool_size": int(os.getenv("PGVECTOR_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("PGVECTOR_MAX_OVERFLOW", "10")),
    },
    "pgvector_backend": {
        "db_url": os.getenv("PGVECTOR_BACKEND_URL", SQLALCHEMY_DATABASE_URL),
        "table_name": os.getenv(
            "PGVECTOR_BACKEND_TABLE", "instantinsight_pgvector_internal"
        ),
        "schema": os.getenv("PGVECTOR_BACKEND_SCHEMA", "ai"),
    },
    # LLM Configuration - Bedrock only
    "bedrock_config": BEDROCK_CONFIG,
    # Agent system configuration
    "enable_agents": True,  # Enable the multi-agent system for SQL refinement and validation
}

# Whether to automatically extract schema from database
AUTO_EXTRACT_SCHEMA = True

# Whether to force rebuild vector store on startup
FORCE_REBUILD_VECTOR_STORE = (
    os.getenv("FORCE_REBUILD_VECTOR_STORE", "false").lower() == "true"
)


# AWS Athena Configuration
ATHENA_CONFIG = {
    "s3_staging_dir": os.getenv(
        "ATHENA_S3_STAGING_DIR", "s3://instantinsight-staging-dir/"
    ),  # S3 bucket for query results
    "region_name": os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),  # AWS region
    "database": os.getenv(
        "ATHENA_DATABASE", "text_to_sql"
    ),  # Athena database name - updated default
    "aws_profile": os.getenv(
        "AWS_PROFILE"
    ),  # AWS profile for credentials (optional, uses IAM role if None)
    "work_group": os.getenv(
        "ATHENA_WORK_GROUP", "instantinsight-workgroup"
    ),  # Athena work group
    "query_timeout": int(
        os.getenv("ATHENA_QUERY_TIMEOUT", "60")
    ),  # Query timeout in seconds
    "s3_output_location": os.getenv(
        "ATHENA_S3_STAGING_DIR", "s3://instantinsight-staging-dir/"
    ),  # Alias for compatibility
    # Custom workgroup configuration
    "use_managed_workgroup": os.getenv("ATHENA_USE_MANAGED_WORKGROUP", "false").lower()
    == "true",
    "enable_result_cache": os.getenv("ATHENA_ENABLE_RESULT_CACHE", "true").lower()
    == "true",
}

# ============================================================================
# UNIVERSAL ANALYTICS DATABASE CONFIGURATION
# ============================================================================


def _build_analytics_url() -> str:
    """
    Build ANALYTICS_DB_URL from ATHENA_CONFIG for initial deployment.

    This helper exists only for initial migration convenience.
    Once ANALYTICS_DB_URL is set in environment, this is bypassed.

    Returns:
        Connection string for Ibis-based AnalyticsConnector

    Note:
        Ibis 11.0+ uses different parameter names for Athena:
        - schema_name instead of database
        - catalog_name instead of catalog
        - s3_staging_dir is required

    """
    from loguru import logger

    # Extract Athena configuration
    region = ATHENA_CONFIG.get("region_name", "ap-southeast-2")
    schema_name = ATHENA_CONFIG.get("database", "text_to_sql")
    work_group = ATHENA_CONFIG.get("work_group", "primary")
    s3_staging = ATHENA_CONFIG.get("s3_staging_dir")

    # Build Athena connection string for Ibis 11.0+
    # NOTE: Using custom workgroup 'instantinsight-workgroup' with dedicated S3 staging directory

    if not s3_staging:
        # Default to instantinsight staging directory
        s3_staging = "s3://instantinsight-staging-dir/"
        logger.warning(f"No s3_staging_dir configured, using default: {s3_staging}")

    # Build connection URL with all required parameters
    # Format: athena://?schema_name=<db>&work_group=<wg>&region_name=<region>&s3_staging_dir=<s3>
    params = [
        f"schema_name={schema_name}",
        f"work_group={work_group}",
        f"region_name={region}",
        f"s3_staging_dir={s3_staging}",
    ]

    url = f"athena://?{'&'.join(params)}"

    logger.info(f"Built ANALYTICS_DB_URL: {url}")
    logger.info(f"Using workgroup '{work_group}' with S3 staging: {s3_staging}")
    return url


# Universal Analytics Database URL
# NOTE: Set ANALYTICS_DB_URL in environment to override automatic building
ANALYTICS_DB_URL = os.getenv("ANALYTICS_DB_URL") or _build_analytics_url()

if not ANALYTICS_DB_URL:
    from loguru import logger

    logger.error(
        "No analytics database configured. "
        "Set ANALYTICS_DB_URL environment variable. "
        "Example: ANALYTICS_DB_URL=athena://awsdatacatalog?region=ap-southeast-2&database=mydb"
    )


# Database selection configuration
DATABASE_ROUTING_CONFIG = {
    "enable_athena": os.getenv("ENABLE_ATHENA", "true").lower()
    == "true",  # Enable Athena by default
    "athena_config": ATHENA_CONFIG,
    "postgres_config": POSTGRES_CONFIG,
    "vector_db": "postgresql",  # PostgreSQL+pgvector for vector operations ONLY
    "business_data_db": "athena",  # Business data goes to Athena
    "default_vector_db": "postgresql",  # Always use PostgreSQL for vector operations
    "default_analytics_db": "athena",  # Analytics and business data goes to Athena
    "force_athena_for_business_queries": True,  # Enforce clean architecture separation
}

# Update RAG_CONFIG to include Athena support
RAG_CONFIG.update(
    {
        # Universal analytics backend (NEW)
        "analytics_db_url": ANALYTICS_DB_URL,
        # Legacy configuration (KEEP for transition - can remove later)
        "database_routing": DATABASE_ROUTING_CONFIG,
        "athena_config": ATHENA_CONFIG,
        # Athena-only mode configuration
        "use_athena_schema_only": os.getenv("USE_ATHENA_SCHEMA_ONLY", "true").lower()
        == "true",
    }
)


# Utility functions
def test_connections():
    """Test all configured database connections."""
    results = {}

    # Test PostgreSQL connection
    try:
        import psycopg

        conn = psycopg.connect(
            host=POSTGRES_CONFIG["host"],
            port=POSTGRES_CONFIG["port"],
            user=POSTGRES_CONFIG["user"],
            password=POSTGRES_CONFIG["password"],
            dbname=POSTGRES_CONFIG["database"],
        )
        conn.close()
        results["postgresql"] = True
    except Exception as e:
        results["postgresql"] = False
        print(f"PostgreSQL connection failed: {e}")

    # Neo4j support removed
    results["neo4j"] = "removed"

    return results


# Expose commonly used configurations for easy import
__all__ = [
    "DATABASE_URL",
    "POSTGRES_CONFIG",
    "RAG_CONFIG",
    "KNOWLEDGE_GRAPH_CONFIG",
    "BEDROCK_CONFIG",
    "ATHENA_CONFIG",
    "ANALYTICS_DB_URL",
    "DATABASE_ROUTING_CONFIG",
    "test_connections",
]
