"""
Configuration dataclasses for VannaRAGEngine.

This module provides structured configuration objects to replace
the numerous parameters in the original VannaRAGEngine constructor.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""

    provider: str = "bedrock"
    model: str = "amazon.titan-embed-text-v2:0"
    dimension: int = 1024
    batch_size: int = 100
    max_retries: int = 3

    def __post_init__(self):
        """Validate embedding configuration."""
        if self.provider != "bedrock":
            raise ValueError(
                f"Only 'bedrock' embedding provider is supported, got: {self.provider}"
            )

        valid_models = [
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-text-v1",
        ]
        if self.model not in valid_models:
            raise ValueError(f"Invalid embedding model: {self.model}")


@dataclass
class PostgresConfig:
    """PostgreSQL database configuration."""

    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DATABASE", "instantinsight")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    # Connection pool settings
    # Production values for thousands of users
    min_connections: int = int(os.getenv("POSTGRES_MIN_CONNECTIONS", 20))
    max_connections: int = int(os.getenv("POSTGRES_MAX_CONNECTIONS", 100))

    # Timeout settings (in seconds)
    pool_timeout: int = int(os.getenv("POSTGRES_POOL_TIMEOUT", 30))
    statement_timeout_ms: int = int(
        os.getenv("POSTGRES_STATEMENT_TIMEOUT", 60000)
    )  # 60 seconds

    # pgvector specific
    vector_dimension: int = 1024
    similarity_threshold: float = 0.7
    max_results: int = 10

    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass
class AthenaConfig:
    """AWS Athena configuration."""

    database: str = "finanalysers"
    s3_staging_dir: str = ""
    region: str = "ap-southeast-2"
    profile: str = None  # Use IAM role when None
    workgroup: str = "primary"

    # Query settings
    query_timeout_seconds: int = 300
    max_results: int = 10000

    def __post_init__(self):
        """Set defaults from environment if not provided."""
        if not self.s3_staging_dir:
            self.s3_staging_dir = os.getenv(
                "ATHENA_S3_STAGING_DIR", "s3://your-staging-bucket/athena-results/"
            )
        if not self.database:
            self.database = os.getenv("ATHENA_DATABASE", "finanalysers")
        if not self.region:
            self.region = os.getenv("AWS_REGION", "ap-southeast-2")
        if not self.profile:
            self.profile = os.getenv(
                "AWS_PROFILE"
            )  # Use IAM role if no profile specified


@dataclass
class DatabaseConfig:
    """Combined database configuration."""

    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    athena: AthenaConfig | None = field(default_factory=AthenaConfig)

    # Routing configuration
    use_athena_for_business_data: bool = True
    athena_table_patterns: list[str] = field(
        default_factory=lambda: [
            "ar*",
            "ap*",
            "gl*",
            "asset*",
            "expense*",
            "agreement*",
        ]
    )

    def should_use_athena(self, sql: str) -> bool:
        """Determine if query should use Athena based on table patterns."""
        if not self.use_athena_for_business_data or not self.athena:
            return False

        sql_lower = sql.lower()
        for pattern in self.athena_table_patterns:
            pattern_regex = pattern.replace("*", ".*")
            if any(pattern_regex in table for table in sql_lower.split()):
                return True
        return False


@dataclass
class LLMConfig:
    """LLM configuration."""

    provider: str = "bedrock"
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120

    # Bedrock specific
    region: str = "ap-southeast-2"
    profile: str = None  # Use IAM role when None

    def __post_init__(self):
        """Validate LLM configuration."""
        if self.provider != "bedrock":
            raise ValueError(
                f"Only 'bedrock' LLM provider is supported, got: {self.provider}"
            )

        # Set from environment if available
        self.region = os.getenv("AWS_REGION", self.region)
        self.profile = os.getenv(
            "AWS_PROFILE", self.profile
        )  # Use IAM role if no profile specified


@dataclass
class AgentConfig:
    """Agent system configuration."""

    enabled: bool = True
    timeout: int = 60
    max_retries: int = 3

    # Individual agent toggles
    agents: dict[str, bool] = field(
        default_factory=lambda: {
            "query_intent_detector": True,
            "query_selector": True,
            "sql_refiner": True,
            "schema_validator": True,
        }
    )

    # Agent-specific settings
    sql_refiner_max_iterations: int = 3
    schema_validator_strict_mode: bool = False

    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if a specific agent is enabled."""
        return self.enabled and self.agents.get(agent_name, False)


@dataclass
class HybridRAGConfig:
    """Hybrid RAG configuration."""

    enabled: bool = True

    # Retrieval weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # Context settings
    max_context_length: int = 8000
    context_overlap: int = 200

    # Merging strategy
    merge_strategy: str = "weighted"  # weighted, concatenate, best
    deduplication: bool = True


@dataclass
class ConversationConfig:
    """Conversation management configuration."""

    max_history: int = 10
    context_window_size: int = 6

    # Memory settings
    enable_summary: bool = True
    summary_after_messages: int = 20

    # Context formatting
    include_timestamps: bool = True
    include_metadata: bool = True


@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration."""

    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Connection pooling
    enable_connection_pooling: bool = True

    # Monitoring
    enable_metrics: bool = True
    metrics_export_interval: int = 60

    # Timeouts
    query_timeout: int = 300
    generation_timeout: int = 120


@dataclass
class EngineConfig:
    """Complete configuration for VannaRAGEngine."""

    # Core components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Optional components
    agents: AgentConfig = field(default_factory=AgentConfig)
    hybrid_rag: HybridRAGConfig | None = field(default_factory=HybridRAGConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    # pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Feature flags
    use_v2_components: bool = False
    enable_parallel_validation: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EngineConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Recursively create nested configs
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        """Create config from dictionary with nested dataclass support."""
        # Handle nested configs
        if "embedding" in data and isinstance(data["embedding"], dict):
            data["embedding"] = EmbeddingConfig(**data["embedding"])

        if "database" in data and isinstance(data["database"], dict):
            db_data = data["database"]
            if "postgres" in db_data and isinstance(db_data["postgres"], dict):
                db_data["postgres"] = PostgresConfig(**db_data["postgres"])
            if "athena" in db_data and isinstance(db_data["athena"], dict):
                db_data["athena"] = AthenaConfig(**db_data["athena"])
            data["database"] = DatabaseConfig(**db_data)

        if "llm" in data and isinstance(data["llm"], dict):
            data["llm"] = LLMConfig(**data["llm"])

        if "agents" in data and isinstance(data["agents"], dict):
            data["agents"] = AgentConfig(**data["agents"])

        if "hybrid_rag" in data and isinstance(data["hybrid_rag"], dict):
            data["hybrid_rag"] = HybridRAGConfig(**data["hybrid_rag"])

        if "conversation" in data and isinstance(data["conversation"], dict):
            data["conversation"] = ConversationConfig(**data["conversation"])

        if "performance" in data and isinstance(data["performance"], dict):
            data["performance"] = PerformanceConfig(**data["performance"])

        return cls(**data)

    def to_yaml(self, path: str | Path):
        """Save configuration to YAML file."""
        path = Path(path)
        data = self._to_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict[str, Any]:
        """Convert to dictionary recursively."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                continue
            elif hasattr(field_value, "__dict__"):
                # Recursively convert nested dataclasses
                result[field_name] = self._dataclass_to_dict(field_value)
            else:
                result[field_name] = field_value
        return result

    def _dataclass_to_dict(self, obj) -> dict[str, Any]:
        """Convert a dataclass to dictionary."""
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: self._dataclass_to_dict(v) if hasattr(v, "__dict__") else v
                for k, v in obj.__dict__.items()
            }
        return obj

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate required fields
        if not self.database.postgres:
            issues.append("PostgreSQL configuration is required")

        # Validate feature compatibility
        # Note: Hybrid RAG with Neo4j has been removed

        # Validate performance settings
        if self.performance.cache_ttl_seconds < 0:
            issues.append("Cache TTL must be non-negative")

        return issues

    @classmethod
    def create_default(cls) -> "EngineConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def create_minimal(cls) -> "EngineConfig":
        """Create minimal configuration (no optional features)."""
        return cls(
            agents=AgentConfig(enabled=False),
            hybrid_rag=None,
            performance=PerformanceConfig(enable_caching=False, enable_metrics=False),
        )
