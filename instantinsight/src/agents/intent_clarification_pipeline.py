"""
Intent Clarification Pipeline - Integration layer for query validation and clarification.

This module demonstrates how QueryIntentGuardrail integrates with ClarificationAgent
to provide a complete intent validation and user clarification workflow.

Flow:
1. User query → QueryIntentGuardrail (validate intent, detect security issues)
2. If invalid/needs clarification → ClarificationAgent (generate helpful response)
3. If valid → Continue to SQL generation pipeline
"""

from loguru import logger

from .strand_agents.query.clarifier import ClarificationAgent
from .strand_agents.query.intent_validator import QueryIntentGuardrail


class GuardrailPipeline:
    """
    Pipeline that integrates intent validation with clarification.

    Flow:
    1. User query -> QueryIntentGuardrail (validate intent)
    2. If invalid/needs clarification -> ClarificationAgent (generate response)
    3. If valid -> Continue to SQL generation
    """

    def __init__(self):
        """Initialize the pipeline components."""
        self.intent_guardrail = QueryIntentGuardrail()
        self.clarification_agent = ClarificationAgent()
        logger.info("GuardrailPipeline initialized")

    def process_query(self, query: str) -> tuple[bool, str | None, str | None]:
        """
        Process a user query through the guardrail pipeline.

        Args:
            query: Natural language query from user

        Returns:
            Tuple of (is_valid, clarification_message, sanitized_query)
            - is_valid: Whether query can proceed to SQL generation
            - clarification_message: Message to show user if clarification needed
            - sanitized_query: The original query if valid, None otherwise

        """
        # Step 1: Validate intent
        logger.info(f"Validating query intent: {query[:100]}...")
        intent_result = self.intent_guardrail.validate(query)

        # Log validation result
        if intent_result.violations:
            logger.warning(f"Intent violations found: {len(intent_result.violations)}")
            for violation in intent_result.violations:
                logger.debug(f"  - {violation.type}: {violation.description}")

        # Step 2: Handle invalid or unclear queries
        if not intent_result.is_valid or intent_result.needs_clarification:
            # Generate clarification message
            clarification_message = self._generate_clarification(query, intent_result)

            logger.info("Query needs clarification or is invalid")
            return False, clarification_message, None

        # Step 3: Query is valid and clear
        logger.info("Query passed all guardrails")
        return True, None, query

    def _generate_clarification(self, query: str, intent_result) -> str:
        """
        Generate appropriate clarification message.

        Args:
            query: Original user query
            intent_result: Result from intent validation

        Returns:
            Clarification message for the user

        """
        # For security violations, use guardrail's message
        if intent_result.has_critical_violations:
            return self.intent_guardrail.generate_clarification_message(intent_result)

        # For other clarifications, use the ClarificationAgent
        if intent_result.clarification_context:
            context = intent_result.clarification_context
            # Pass context to ClarificationAgent
            return self.clarification_agent.generate_clarification_response(
                context, query
            )

        # Default clarification
        return (
            "Your query needs clarification. Please provide:\n"
            "- What specific data you're looking for\n"
            "- Any filters or conditions\n"
            "- How you want the data organized"
        )


def validate_user_query(query: str) -> tuple[bool, str | None]:
    """
    Validate a user query.

    Args:
        query: Natural language query from user

    Returns:
        Tuple of (is_valid, clarification_message)

    """
    pipeline = GuardrailPipeline()
    is_valid, clarification, _ = pipeline.process_query(query)
    return is_valid, clarification
