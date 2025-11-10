"""
Query Intent Guardrail Agent - First line of defense for natural language queries using Strand framework.

This agent validates user queries at the natural language level to ensure they have
legitimate data querying intent and blocks malicious attempts before SQL generation.
Migrated to use Strand framework for enhanced clarification generation.
"""

import re
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, Field
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)

from ..model_config import get_agent_config


@dataclass
class IntentViolation:
    """Details of an intent violation."""

    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    suggestion: str


@dataclass
class IntentValidationResult:
    """Result of intent validation."""

    is_valid: bool
    violations: list[IntentViolation]
    needs_clarification: bool
    clarification_context: dict | None = None

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        return any(v.severity == "CRITICAL" for v in self.violations)


class ClarificationResponse(BaseModel):
    """Structured response for clarification generation."""

    message: str = Field(
        description="Clear, helpful clarification message for the user"
    )
    examples: list[str] = Field(
        description="2-3 example queries the user could try instead",
        default_factory=list,
    )
    reasoning: str = Field(
        description="Internal reasoning for why clarification was needed"
    )


class QueryIntentGuardrail:
    """Strand-based query intent validation implementation."""

    # Patterns that indicate SQL injection or hacking attempts in natural language
    INJECTION_PATTERNS = [
        # Direct SQL injection attempts
        (
            r"(?:drop|delete|truncate|alter|create)\s+(?:table|database)",
            "Direct SQL command in query",
            "CRITICAL",
        ),
        (r";\s*--", "SQL comment injection pattern", "CRITICAL"),
        (r"union\s+select", "UNION SELECT pattern", "CRITICAL"),
        (r"exec(?:ute)?\s*\(", "Execute command pattern", "CRITICAL"),
        # Common injection phrases
        (r"(?:^|\s)or\s+1\s*=\s*1", "OR 1=1 injection pattern", "CRITICAL"),
        (r"(?:^|\s)and\s+1\s*=\s*1", "AND 1=1 injection pattern", "CRITICAL"),
        (r"'\s*or\s*'", "Quote OR quote pattern", "CRITICAL"),
        # System probing
        (r"show\s+(?:tables|databases|schemas)", "System enumeration attempt", "HIGH"),
        (r"information_schema", "System table reference", "HIGH"),
        (r"(?:select|from)\s+(?:mysql|pg_|sys)\.", "System table access", "HIGH"),
    ]

    # Patterns that indicate debugging or testing
    DEBUG_PATTERNS = [
        (r"\b(?:test|debug|demo)\b", "Testing keyword detected", "MEDIUM"),
        (r"(?:^|\s)1\s*=\s*1", "Always-true condition", "HIGH"),
        (r"(?:^|\s)2\s*=\s*2", "Always-true condition", "HIGH"),
        (r"where\s+(?:true|1)(?:\s|$)", "WHERE TRUE/1 pattern", "HIGH"),
        (r"select\s+\*\s+from\s+\w+\s*;", "Select all pattern", "LOW"),
    ]

    # Keywords that suggest non-querying intent
    FORBIDDEN_INTENTS = [
        "hack",
        "exploit",
        "injection",
        "vulnerability",
        "password",
        "admin",
        "root",
        "sudo",
        "bypass",
        "override",
        "disable",
        "turn off",
    ]

    # Patterns for vague/ambiguous queries that need clarification
    VAGUE_QUERY_PATTERNS = [
        # Too broad - just "show X" without specificity
        (
            r"^(?:show|display|get|list|find)\s+(?:me\s+)?(?:the\s+)?(\w+)(?:\s+data)?$",
            "Query too broad - needs more specificity",
            "MEDIUM",
            "Please specify what information about {0} you need",
        ),
        # Vague temporal references
        (
            r"\b(?:recent|recently|lately|soon|later|earlier)\b",
            "Vague temporal reference",
            "MEDIUM",
            "Please specify an exact time period (e.g., 'last 30 days', 'since January 2024')",
        ),
        (
            r"^(?:show|give|get)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?data$",
            "Request too generic",
            "MEDIUM",
            "Please specify which data you need and for what purpose",
        ),
    ]

    def __init__(
        self, *, model_id: str = None, aws_region: str = None, debug_mode: bool = False
    ):
        """
        Initialize QueryIntentGuardrail with Strand Agent.

        Args:
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            debug_mode: Enable debug logging

        """
        # Get configuration from centralized config
        agent_config = get_agent_config("QueryIntentValidator", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]
        self.debug_mode = debug_mode

        # Compile regex patterns
        self.injection_regex = [
            (re.compile(p, re.IGNORECASE), d, s) for p, d, s in self.INJECTION_PATTERNS
        ]
        self.debug_regex = [
            (re.compile(p, re.IGNORECASE), d, s) for p, d, s in self.DEBUG_PATTERNS
        ]
        # Compile vague query patterns with suggestions
        self.vague_regex = [
            (re.compile(p, re.IGNORECASE), d, s, sug)
            for p, d, s, sug in self.VAGUE_QUERY_PATTERNS
        ]

        # Create Bedrock model
        # Note: Bedrock region is set via AWS_REGION env var or AWS credentials
        self._cache_prompt_type = agent_config.get("cache_prompt")

        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            streaming=False,
            cache_prompt=self._cache_prompt_type,
        )

        # Base instructions for clarification generation
        base_instructions = """You are a helpful query clarification specialist for a business data analysis system.
        Generate clear, friendly clarification messages when users provide vague, problematic, or insecure queries.
        Provide helpful examples of proper data queries they could use instead."""

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        # Initialize Strand Agent for clarification generation
        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

        logger.info("✓ QueryIntentGuardrail (strand) initialized")

    @observe(as_type="generation")
    def validate(self, query: str) -> IntentValidationResult:
        """
        Validate a natural language query for legitimate data querying intent.

        Args:
            query: Natural language query from user

        Returns:
            IntentValidationResult with validation details

        """
        if not query or not query.strip():
            return IntentValidationResult(
                is_valid=False,
                violations=[
                    IntentViolation(
                        type="EMPTY_QUERY",
                        severity="LOW",
                        description="Empty or whitespace-only query",
                        suggestion="Please provide a specific data query",
                    )
                ],
                needs_clarification=True,
            )

        violations = []
        needs_clarification = False
        clarification_context = {}

        # Check for injection patterns
        injection_violations = self._check_injection_patterns(query)
        violations.extend(injection_violations)

        # Check for debugging patterns
        debug_violations = self._check_debug_patterns(query)
        violations.extend(debug_violations)

        # Check for forbidden intents
        forbidden_violations = self._check_forbidden_intents(query)
        violations.extend(forbidden_violations)

        # Check for vague/ambiguous patterns
        vague_violations = self._check_vague_patterns(query)
        violations.extend(vague_violations)

        # Determine if query is valid
        # CRITICAL and HIGH violations block the query
        # MEDIUM violations (vague queries) need clarification but aren't blocking
        is_valid = not any(v.severity in ["HIGH", "CRITICAL"] for v in violations)

        # Check if we need clarification (for security violations or vague queries)
        if not is_valid and violations:
            needs_clarification = True
            clarification_context = {
                "type": "security_violation",
                "reasoning": self._summarize_violations(violations),
                "original_query": query,
                "violations": [
                    {"type": v.type, "description": v.description}
                    for v in violations
                    if v.severity in ["HIGH", "CRITICAL"]
                ],
            }
        elif vague_violations:  # Handle vague queries that need clarification
            needs_clarification = True
            clarification_context = {
                "type": "vague_query",
                "reasoning": "Query needs more specificity",
                "original_query": query,
                "violations": [
                    {
                        "type": v.type,
                        "description": v.description,
                        "suggestion": v.suggestion,
                    }
                    for v in vague_violations
                ],
            }

        return IntentValidationResult(
            is_valid=is_valid,
            violations=violations,
            needs_clarification=needs_clarification,
            clarification_context=clarification_context,
        )

    def generate_clarification_message(self, result: IntentValidationResult) -> str:
        """
        Generate a clarification message using Strand Agent for enhanced responses.

        Args:
            result: Validation result with context

        Returns:
            Clarification message for the user

        """
        if not result.clarification_context:
            return "Please provide a specific data query."

        context_type = result.clarification_context.get("type")

        if context_type == "security_violation":
            return self._generate_security_clarification_with_strand(result)
        elif context_type == "vague_query":
            return self._generate_vague_query_clarification_with_strand(result)
        else:
            return "Please clarify your data query."

    @observe(as_type="generation")
    def _generate_security_clarification_with_strand(
        self, result: IntentValidationResult
    ) -> str:
        """Generate enhanced clarification for security violations using Strand."""
        try:
            violations_summary = self._summarize_violations(result.violations)
            original_query = result.clarification_context.get("original_query", "")

            prompt = f"""
            A user submitted a query that triggered security violations: "{original_query}"

            Issue: {violations_summary}

            Generate a helpful, friendly clarification message that:
            1. Explains this is a business data system accepting natural language queries only
            2. Provides 2-3 example queries they could use instead
            3. Is educational rather than accusatory
            4. Guides them toward proper data queries

            Focus on being helpful and constructive.
            """

            # Reset usage
            self._usage_container["last_usage"] = None

            # Run agent
            llm_result = self.agent.structured_output(ClarificationResponse, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "QueryIntentValidator",
                langfuse_context,
            )
            self._log_prompt_cache_metrics()

            # Format the response nicely
            message = llm_result.message
            if llm_result.examples:
                message += "\n\nHere are some example queries you could try:\n"
                for i, example in enumerate(llm_result.examples[:3], 1):
                    message += f"{i}. {example}\n"

            return message

        except Exception as e:
            logger.error(f"Strand security clarification failed: {e}")
            return self._get_fallback_security_message(result)

    def _get_fallback_security_message(self, result: IntentValidationResult) -> str:
        """Get fallback security message when Strand fails."""
        if result.has_critical_violations:
            return (
                "Your query appears to contain SQL syntax or patterns that are not allowed.\n\n"
                "This system accepts natural language queries only. For example:\n"
                "- 'Show me sales data for last month'\n"
                "- 'What are the top 10 products by revenue?'\n"
                "- 'List customers from Sydney with orders over $1000'\n\n"
                "Please reformulate your query in plain English without SQL syntax."
            )
        else:
            return (
                "Your query contains patterns that suggest non-data querying intent.\n\n"
                "This system is designed for business data queries only. Examples:\n"
                "- 'Show total revenue by product category'\n"
                "- 'Find customers with overdue payments'\n"
                "- 'What were the sales trends last quarter?'\n\n"
                "Please provide a legitimate business data query."
            )

    @observe(as_type="generation")
    def _generate_vague_query_clarification_with_strand(
        self, result: IntentValidationResult
    ) -> str:
        """Generate enhanced clarification for vague queries using Strand."""
        try:
            original_query = result.clarification_context.get("original_query", "")
            violations = result.clarification_context.get("violations", [])

            violations_text = "\n".join(
                [f"- {v['description']}: {v.get('suggestion', '')}" for v in violations]
            )

            prompt = f"""
            A user submitted a vague query that needs clarification: "{original_query}"

            Issues identified:
            {violations_text}

            Generate a helpful clarification message that:
            1. Explains what additional information is needed
            2. Provides 2-3 specific example queries they could use instead
            3. Is encouraging and constructive
            4. Helps them understand how to be more specific

            Focus on being helpful and guiding them to success.
            """

            # Reset usage
            self._usage_container["last_usage"] = None

            # Run agent
            llm_result = self.agent.structured_output(ClarificationResponse, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "QueryIntentValidator",
                langfuse_context,
            )
            self._log_prompt_cache_metrics()

            # Format the response nicely
            message = llm_result.message
            if llm_result.examples:
                message += "\n\nHere are some specific examples:\n"
                for i, example in enumerate(llm_result.examples[:3], 1):
                    message += f"{i}. {example}\n"

            return message

        except Exception as e:
            logger.error(f"Strand vague query clarification failed: {e}")
            return self._get_fallback_vague_message(result)

    def _get_fallback_vague_message(self, result: IntentValidationResult) -> str:
        """Get fallback vague query message when Strand fails."""
        violations = result.clarification_context.get("violations", [])

        if not violations:
            return (
                "Your query needs more specificity. Please provide additional details."
            )

        # Build clarification message from violations
        message = "Your query needs clarification:\n\n"

        for violation in violations:
            message += f"• {violation['description']}\n"
            if "suggestion" in violation:
                message += f"  → {violation['suggestion']}\n"

        return message

    def _check_injection_patterns(self, query: str) -> list[IntentViolation]:
        """Check for SQL injection patterns in natural language."""
        violations = []

        for pattern, description, severity in self.injection_regex:
            if pattern.search(query):
                violations.append(
                    IntentViolation(
                        type="INJECTION_ATTEMPT",
                        severity=severity,
                        description=description,
                        suggestion="Please formulate a legitimate data query without SQL syntax",
                    )
                )

        return violations

    def _check_debug_patterns(self, query: str) -> list[IntentViolation]:
        """Check for debugging/testing patterns."""
        violations = []

        for pattern, description, severity in self.debug_regex:
            if pattern.search(query):
                violations.append(
                    IntentViolation(
                        type="DEBUG_PATTERN",
                        severity=severity,
                        description=description,
                        suggestion="Please provide a real business query, not a test",
                    )
                )

        return violations

    def _check_forbidden_intents(self, query: str) -> list[IntentViolation]:
        """Check for keywords suggesting non-querying intent."""
        violations = []
        query_lower = query.lower()

        for keyword in self.FORBIDDEN_INTENTS:
            if keyword in query_lower:
                violations.append(
                    IntentViolation(
                        type="FORBIDDEN_INTENT",
                        severity="HIGH",
                        description=f"Query contains forbidden keyword: {keyword}",
                        suggestion="This system is for data queries only",
                    )
                )

        return violations

    def _check_vague_patterns(self, query: str) -> list[IntentViolation]:
        """Check for vague or ambiguous query patterns that need clarification."""
        violations = []

        for pattern, description, severity, suggestion in self.vague_regex:
            match = pattern.search(query)
            if match:
                # Format suggestion with captured groups if any
                formatted_suggestion = suggestion
                if match.groups():
                    try:
                        formatted_suggestion = suggestion.format(*match.groups())
                    except Exception as e:
                        logger.error(f"Error formatting suggestion: {e}")
                        pass  # Use original suggestion if formatting fails

                violations.append(
                    IntentViolation(
                        type="VAGUE_QUERY",
                        severity=severity,
                        description=description,
                        suggestion=formatted_suggestion,
                    )
                )

        return violations

    def _summarize_violations(self, violations: list[IntentViolation]) -> str:
        """Create a summary of violations for clarification."""
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        high_violations = [v for v in violations if v.severity == "HIGH"]

        if critical_violations:
            return f"Security violation detected: {critical_violations[0].description}"
        elif high_violations:
            return f"Invalid query pattern: {high_violations[0].description}"
        elif violations:
            return f"Query issue: {violations[0].description}"
        else:
            return "Query needs clarification"

    def _log_prompt_cache_metrics(self) -> None:
        """Log prompt caching metrics for observability when enabled."""
        if not self._cache_prompt_type:
            logger.debug(
                "QueryIntentValidator prompt cache metrics skipped (cache_prompt disabled)"
            )
            return None

        return log_prompt_cache_status("QueryIntentValidator", self._usage_container)
