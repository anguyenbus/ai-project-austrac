"""
SQL Generation Service for V2 Architecture.

This component handles all SQL generation functionality that was previously
embedded in the monolithic VannaRAGEngine.
"""

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class SQLGenerationConfig:
    """Configuration for SQL generation service."""

    use_agents: bool = True
    max_context_messages: int = 6
    validate_tables: bool = True
    use_conversation_context: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""

    sql: str
    confidence: float = 0.0
    validation_passed: bool = False
    agent_analysis: dict[str, Any] | None = None
    generation_time: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize default metadata if not specified."""
        if self.metadata is None:
            self.metadata = {}


class SQLGenerationService:
    """
    Service for generating SQL from natural language questions.

    Features:
    - Context-aware SQL generation
    - Agent-based routing and validation
    - Schema validation and hallucination prevention
    - Performance monitoring and caching
    """

    def __init__(self, rag_engine, config: SQLGenerationConfig | None = None):
        """
        Initialize SQL generation service.

        Args:
            rag_engine: The RAG engine instance for database access
            config: Configuration for SQL generation

        """
        self.rag_engine = rag_engine
        self.config = config or SQLGenerationConfig()

        # Initialize agents if available and enabled
        self.agents = {}
        if self.config.use_agents:
            self._init_agents()

        # Performance tracking
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "validation_pass_rate": 0.0,
        }

        logger.info("SQLGenerationService initialized")

    def _init_agents(self):
        """Initialize agent system for enhanced SQL generation."""
        try:
            # Check if agents are available in the RAG engine
            if hasattr(self.rag_engine, "sql_refiner"):
                self.agents["sql_refiner"] = self.rag_engine.sql_refiner

            if hasattr(self.rag_engine, "schema_validator"):
                self.agents["schema_validator"] = self.rag_engine.schema_validator

            logger.info(
                f"SQLGenerationService agents initialized: {list(self.agents.keys())}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize some agents: {e}")

    def generate_sql(
        self,
        question: str,
        conversation_context: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> SQLGenerationResult:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            conversation_context: Previous conversation messages
            **kwargs: Additional generation parameters

        Returns:
            SQLGenerationResult with generated SQL and metadata

        """
        start_time = time.time()
        self.generation_stats["total_requests"] += 1

        try:
            # Check if RAG system is initialized
            if not getattr(self.rag_engine, "is_initialized", False):
                return SQLGenerationResult(
                    sql="",
                    error="RAG system not initialized",
                    generation_time=time.time() - start_time,
                )

            # Get conversation context if not provided
            if conversation_context is None and self.config.use_conversation_context:
                conversation_context = self._get_conversation_context()

            # Step 1: Intent analysis (if agents enabled)
            agent_analysis = {}
            if self.config.use_agents and "intent_detector" in self.agents:
                intent_result = self._analyze_intent(question, conversation_context)
                agent_analysis["intent"] = intent_result

            # Step 2: Generate base SQL
            sql = self._generate_base_sql(question, conversation_context, **kwargs)

            if not sql:
                return SQLGenerationResult(
                    sql="",
                    error="Failed to generate SQL",
                    agent_analysis=agent_analysis,
                    generation_time=time.time() - start_time,
                )

            # Step 3: Validation and refinement
            validation_result = None
            if self.config.validate_tables:
                validation_result = self._validate_and_refine_sql(sql, question)
                if validation_result.get("refined_sql"):
                    sql = validation_result["refined_sql"]

            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(
                sql, agent_analysis, validation_result
            )

            # Track success
            self.generation_stats["successful_generations"] += 1
            generation_time = time.time() - start_time

            # Update average generation time
            self._update_avg_generation_time(generation_time)

            return SQLGenerationResult(
                sql=sql,
                confidence=confidence,
                validation_passed=(
                    validation_result.get("validation_passed", False)
                    if validation_result
                    else False
                ),
                agent_analysis=agent_analysis,
                generation_time=generation_time,
                metadata={
                    "question": question,
                    "context_used": conversation_context is not None,
                    "agents_used": (
                        list(self.agents.keys()) if self.config.use_agents else []
                    ),
                    "validation_result": validation_result,
                },
            )

        except Exception as e:
            self.generation_stats["failed_generations"] += 1
            logger.error(f"SQL generation failed: {e}")

            return SQLGenerationResult(
                sql="", error=str(e), generation_time=time.time() - start_time
            )

    def generate_sql_with_context(
        self,
        question: str,
        use_conversation_context: bool = True,
        max_context_messages: int = 4,
        **kwargs,
    ) -> SQLGenerationResult:
        """
        Generate SQL with explicit conversation context control.

        Args:
            question: Natural language question
            use_conversation_context: Whether to use conversation history
            max_context_messages: Maximum context messages to include
            **kwargs: Additional arguments

        Returns:
            SQLGenerationResult

        """
        conversation_context = None
        if use_conversation_context:
            conversation_context = self._get_conversation_context(max_context_messages)

        return self.generate_sql(
            question=question, conversation_context=conversation_context, **kwargs
        )

    def generate_sql_with_validation(
        self,
        question: str,
        validate_tables: bool = True,
        use_conversation_context: bool = True,
        max_context_messages: int = 6,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate SQL with mandatory validation (legacy interface).

        Args:
            question: Natural language question
            validate_tables: Whether to validate table existence
            use_conversation_context: Whether to use conversation history
            max_context_messages: Maximum context messages to include
            **kwargs: Additional arguments

        Returns:
            Dictionary with sql, validation results, and errors

        """
        # Update config for this request
        original_validate = self.config.validate_tables
        original_context = self.config.use_conversation_context
        original_max_msgs = self.config.max_context_messages

        self.config.validate_tables = validate_tables
        self.config.use_conversation_context = use_conversation_context
        self.config.max_context_messages = max_context_messages

        try:
            result = self.generate_sql(question, **kwargs)

            # Convert to legacy format
            return {
                "sql": result.sql,
                "error": result.error,
                "validation_passed": result.validation_passed,
                "confidence": result.confidence,
                "agent_analysis": result.agent_analysis,
                "generation_time": result.generation_time,
                "metadata": result.metadata,
            }

        finally:
            # Restore original config
            self.config.validate_tables = original_validate
            self.config.use_conversation_context = original_context
            self.config.max_context_messages = original_max_msgs

    def _get_conversation_context(
        self, max_messages: int | None = None
    ) -> list[dict[str, str]]:
        """Get conversation context from the RAG engine."""
        max_messages = max_messages or self.config.max_context_messages

        try:
            if hasattr(self.rag_engine, "get_conversation_context"):
                return self.rag_engine.get_conversation_context(max_messages)
        except Exception as e:
            logger.warning(f"Failed to get conversation context: {e}")

        return []

    def _analyze_intent(
        self, question: str, conversation_context: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Analyze query intent using agent."""
        try:
            intent_detector = self.agents.get("intent_detector")
            if intent_detector:
                context_strings = []
                if conversation_context:
                    context_strings = [
                        msg.get("content", "") for msg in conversation_context
                    ]

                return intent_detector.analyze_intent(
                    question, conversation_context=context_strings
                )
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")

        return {}

    def _generate_base_sql(
        self,
        question: str,
        conversation_context: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> str:
        """Generate base SQL using the RAG engine."""
        try:
            # Build context information
            context_info = ""
            if conversation_context:
                context_parts = []
                for _i, msg in enumerate(
                    conversation_context[-self.config.max_context_messages :]
                ):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    context_parts.append(f"{role.title()}: {content}")

                if context_parts:
                    context_info = "\n\nConversation Context:\n" + "\n".join(
                        context_parts
                    )

            # Enhanced prompt with context
            enhanced_question = f"{question}{context_info}"

            # Use the RAG engine to generate SQL directly
            # Note: RAG persistence is handled by VannaRAGEngine directly
            # to avoid circular calls
            if hasattr(self.rag_engine, "generate_sql"):
                return self.rag_engine.generate_sql(enhanced_question)
            else:
                logger.warning("RAG SQL generation method not available")
                return ""

        except Exception as e:
            logger.error(f"Base SQL generation failed: {e}")
            return ""

    def _validate_and_refine_sql(self, sql: str, question: str) -> dict[str, Any]:
        """Validate and refine SQL using agents."""
        validation_result = {
            "validation_passed": False,
            "refined_sql": sql,
            "validation_errors": [],
            "refinement_applied": False,
        }

        try:
            # Step 1: Schema validation
            if "schema_validator" in self.agents:
                schema_validator = self.agents["schema_validator"]
                schema_result = schema_validator.validate_sql_components(sql)

                validation_result["schema_validation"] = schema_result
                validation_result["validation_passed"] = schema_result.get(
                    "all_tables_valid", False
                )

                if not validation_result["validation_passed"]:
                    validation_result["validation_errors"].extend(
                        schema_result.get("validation_errors", [])
                    )

            # Step 2: SQL refinement if validation failed or refinement is requested
            if "sql_refiner" in self.agents and (
                not validation_result["validation_passed"] or self.config.use_agents
            ):
                sql_refiner = self.agents["sql_refiner"]
                # Use SQLFixer directly
                fix_result = sql_refiner.refine_sql(
                    sql, "Schema validation failed", None
                )
                refinement_result = {
                    "refined_sql": (
                        fix_result.corrected_sql if fix_result.success else sql
                    ),
                    "success": fix_result.success,
                    "error": fix_result.error_message,
                }

                if (
                    refinement_result.get("refined_sql")
                    and refinement_result["refined_sql"] != sql
                ):
                    validation_result["refined_sql"] = refinement_result["refined_sql"]
                    validation_result["refinement_applied"] = True
                    validation_result["refinement_details"] = refinement_result

                    # Re-validate refined SQL
                    if "schema_validator" in self.agents:
                        refined_validation = schema_validator.validate_sql_components(
                            validation_result["refined_sql"]
                        )
                        validation_result["validation_passed"] = refined_validation.get(
                            "all_tables_valid", False
                        )

        except Exception as e:
            logger.error(f"SQL validation/refinement failed: {e}")
            validation_result["validation_errors"].append(str(e))

        return validation_result

    def _calculate_confidence(
        self,
        sql: str,
        agent_analysis: dict[str, Any],
        validation_result: dict[str, Any] | None,
    ) -> float:
        """Calculate confidence score for generated SQL."""
        confidence = 0.5  # Base confidence

        try:
            # Factor 1: SQL complexity and structure
            if sql and len(sql.strip()) > 10:
                confidence += 0.1

            if "SELECT" in sql.upper() and "FROM" in sql.upper():
                confidence += 0.1

            # Factor 2: Agent analysis confidence
            if agent_analysis and "intent" in agent_analysis:
                intent_confidence = agent_analysis["intent"].get("confidence", 0.0)
                confidence += intent_confidence * 0.2

            # Factor 3: Validation results
            if validation_result:
                if validation_result.get("validation_passed", False):
                    confidence += 0.2
                else:
                    confidence -= 0.1

                if validation_result.get("refinement_applied", False):
                    confidence += 0.1

            # Clamp between 0 and 1
            confidence = max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            confidence = 0.5

        return confidence

    def _update_avg_generation_time(self, generation_time: float):
        """Update average generation time statistics."""
        total_successful = self.generation_stats["successful_generations"]
        current_avg = self.generation_stats["avg_generation_time"]

        # Calculate new average
        if total_successful == 1:
            self.generation_stats["avg_generation_time"] = generation_time
        else:
            self.generation_stats["avg_generation_time"] = (
                current_avg * (total_successful - 1) + generation_time
            ) / total_successful

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the service."""
        total_requests = self.generation_stats["total_requests"]
        successful = self.generation_stats["successful_generations"]

        return {
            "total_requests": total_requests,
            "successful_generations": successful,
            "failed_generations": self.generation_stats["failed_generations"],
            "success_rate": successful / total_requests if total_requests > 0 else 0.0,
            "avg_generation_time": self.generation_stats["avg_generation_time"],
            "agents_available": list(self.agents.keys()),
            "config": {
                "use_agents": self.config.use_agents,
                "validate_tables": self.config.validate_tables,
                "max_context_messages": self.config.max_context_messages,
            },
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "validation_pass_rate": 0.0,
        }
        logger.info("SQLGenerationService stats reset")
