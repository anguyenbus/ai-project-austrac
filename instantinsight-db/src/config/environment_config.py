"""
Environment Configuration Manager.

This module handles loading and managing environment-specific configurations
for development, staging, and production deployments.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Import V2 config classes
from ..rag.v2.components.engine_config import (
    AgentConfig,
    AthenaConfig,
    ConversationConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EngineConfig,
    HybridRAGConfig,
    LLMConfig,
    PerformanceConfig,
    PipelineConfig,
    PostgresConfig,
)


@dataclass
class SecurityConfig:
    """Security configuration."""

    enable_sql_validation: bool = True
    max_query_length: int = 10000
    allowed_sql_operations: list = None
    blocked_keywords: list = None
    enable_code_validation: bool = True
    enable_input_sanitization: bool = False
    max_file_upload_size: int = 10485760
    session_timeout: int = 3600

    def __post_init__(self):
        """Initialize default allowed SQL operations if not specified."""
        if self.allowed_sql_operations is None:
            self.allowed_sql_operations = ["SELECT", "WITH"]
        if self.blocked_keywords is None:
            self.blocked_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
            ]


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    enable_health_checks: bool = True
    health_check_interval: int = 60
    enable_metrics_export: bool = True
    metrics_endpoint: str = "/metrics"
    enable_tracing: bool = False
    enable_alerting: bool = False
    alert_thresholds: dict[str, float] = None

    def __post_init__(self):
        """Initialize default alert thresholds if not specified."""
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "error_rate": 0.05,
                "response_time_p95": 10000,
                "memory_usage": 0.85,
                "cpu_usage": 0.80,
            }


@dataclass
class RateLimitingConfig:
    """Rate limiting configuration."""

    enabled: bool = False
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "{time} | {level} | {name}:{function}:{line} - {message}"
    rotation: str = "100 MB"
    retention: str = "7 days"
    structured: bool = False
    include_trace_id: bool = False


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""

    environment: str
    engine: EngineConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    rate_limiting: RateLimitingConfig
    logging: LoggingConfig
    feature_flags: dict[str, bool]

    @classmethod
    def load_from_file(cls, config_path: Path) -> "EnvironmentConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Substitute environment variables
        data = cls._substitute_env_vars(data)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        """Create configuration from dictionary."""
        # Create EngineConfig
        engine_config = cls._create_engine_config(data)

        # Create other configs
        security_config = SecurityConfig(**data.get("security", {}))
        monitoring_config = MonitoringConfig(**data.get("monitoring", {}))
        rate_limiting_config = RateLimitingConfig(**data.get("rate_limiting", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))

        return cls(
            environment=data.get("environment", "development"),
            engine=engine_config,
            security=security_config,
            monitoring=monitoring_config,
            rate_limiting=rate_limiting_config,
            logging=logging_config,
            feature_flags=data.get("feature_flags", {}),
        )

    @classmethod
    def _create_engine_config(cls, data: dict[str, Any]) -> EngineConfig:
        """Create EngineConfig from configuration data."""
        # Database config
        db_data = data.get("database", {})
        postgres_config = PostgresConfig(**db_data.get("postgres", {}))

        athena_data = db_data.get("athena", {})
        athena_config = AthenaConfig(**athena_data) if athena_data else None

        # Neo4j support removed
        database_config = DatabaseConfig(postgres=postgres_config, athena=athena_config)

        # Other configs
        llm_config = LLMConfig(**data.get("llm", {}))
        embedding_config = EmbeddingConfig(**data.get("embedding", {}))
        agent_config = AgentConfig(**data.get("agents", {}))
        conversation_config = ConversationConfig(**data.get("conversation", {}))
        performance_config = PerformanceConfig(**data.get("performance", {}))

        # Hybrid RAG config
        hybrid_rag_data = data.get("hybrid_rag", {})
        hybrid_rag_config = (
            HybridRAGConfig(**hybrid_rag_data) if hybrid_rag_data else None
        )

        # Pipeline config
        pipeline_data = data.get("pipeline", {})
        pipeline_config = (
            PipelineConfig(**pipeline_data) if pipeline_data else PipelineConfig()
        )

        return EngineConfig(
            embedding=embedding_config,
            database=database_config,
            llm=llm_config,
            agents=agent_config,
            hybrid_rag=hybrid_rag_config,
            conversation=conversation_config,
            performance=performance_config,
            pipeline=pipeline_config,
            use_v2_components=data.get("feature_flags", {}).get(
                "use_engine_config", False
            ),
            enable_parallel_validation=data.get("feature_flags", {}).get(
                "enable_parallel_validation", False
            ),
        )

    @classmethod
    def _substitute_env_vars(cls, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {key: cls._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [cls._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            if data.startswith("${") and data.endswith("}"):
                # Format: ${VAR_NAME} or ${VAR_NAME:-default_value}
                var_expr = data[2:-1]  # Remove ${ and }

                if ":-" in var_expr:
                    var_name, default_value = var_expr.split(":-", 1)
                    return os.getenv(var_name, default_value)
                else:
                    var_value = os.getenv(var_expr)
                    if var_value is None:
                        raise ValueError(
                            f"Required environment variable not set: {var_expr}"
                        )
                    return var_value
            return data
        else:
            return data

    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate engine config
        engine_issues = self.engine.validate()
        issues.extend(engine_issues)

        # Environment-specific validation
        if self.environment == "production":
            # Production-specific validation
            if not self.security.enable_sql_validation:
                issues.append("SQL validation must be enabled in production")

            if not self.monitoring.enable_health_checks:
                issues.append("Health checks must be enabled in production")

            if (
                self.engine.database.athena
                and not self.engine.database.athena.s3_staging_dir
            ):
                issues.append("Athena S3 staging directory is required in production")

            if self.logging.level == "DEBUG":
                issues.append("DEBUG logging level not recommended for production")

        return issues


class EnvironmentConfigManager:
    """Manages environment-specific configurations."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize environment configuration manager.

        Args:
            config_dir: Directory containing environment configuration files

        """
        self.config_dir = (
            config_dir
            or Path(__file__).parent.parent.parent / "config" / "environments"
        )
        self._current_config: EnvironmentConfig | None = None
        self._environment: str | None = None

    def load_environment(self, environment: str) -> EnvironmentConfig:
        """Load configuration for specified environment."""
        config_file = self.config_dir / f"{environment}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        logger.info(f"Loading {environment} environment configuration")

        config = EnvironmentConfig.load_from_file(config_file)

        # Validate configuration
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration issues found: {issues}")
            if environment == "production" and issues:
                raise ValueError(
                    f"Production configuration has critical issues: {issues}"
                )

        self._current_config = config
        self._environment = environment

        logger.info(f"✓ {environment} environment configuration loaded successfully")
        return config

    def get_current_config(self) -> EnvironmentConfig | None:
        """Get currently loaded configuration."""
        return self._current_config

    def get_current_environment(self) -> str | None:
        """Get current environment name."""
        return self._environment

    def auto_detect_environment(self) -> str:
        """Auto-detect environment from environment variables."""
        # Check common environment variables
        env_indicators = [
            (
                "NODE_ENV",
                {
                    "production": "production",
                    "staging": "staging",
                    "development": "development",
                },
            ),
            (
                "ENVIRONMENT",
                {"prod": "production", "stage": "staging", "dev": "development"},
            ),
            (
                "DEPLOY_ENV",
                {
                    "production": "production",
                    "staging": "staging",
                    "development": "development",
                },
            ),
        ]

        for env_var, mappings in env_indicators:
            env_value = os.getenv(env_var, "").lower()
            if env_value in mappings:
                detected_env = mappings[env_value]
                logger.info(
                    f"Auto-detected environment: {detected_env} (from {env_var}={env_value})"
                )
                return detected_env

        # Default to development
        logger.info("No environment detected, defaulting to development")
        return "development"

    def apply_feature_flags(self):
        """Apply feature flags from current configuration."""
        if not self._current_config:
            logger.warning("No configuration loaded, cannot apply feature flags")
            return

        from ..config.feature_flags import feature_flags

        # Apply feature flags
        for flag_name, flag_value in self._current_config.feature_flags.items():
            if flag_value:
                feature_flags.enable(flag_name)
            else:
                feature_flags.disable(flag_name)

        logger.info(f"Applied feature flags for {self._environment} environment")

    def setup_logging(self):
        """Configure logging based on current configuration."""
        if not self._current_config:
            logger.warning("No configuration loaded, cannot setup logging")
            return

        from loguru import logger as loguru_logger

        # Remove default handler
        loguru_logger.remove()

        # Add new handler with environment-specific configuration
        log_config = self._current_config.logging
        loguru_logger.add(
            sink=f"logs/nl2vis_{self._environment}.log",
            level=log_config.level,
            format=log_config.format,
            rotation=log_config.rotation,
            retention=log_config.retention,
            enqueue=True,
        )

        # Also add console output
        loguru_logger.add(
            sink=lambda msg: print(msg, end=""),
            level=log_config.level,
            format=log_config.format,
        )

        logger.info(f"Logging configured for {self._environment} environment")


# Global instance
config_manager = EnvironmentConfigManager()
