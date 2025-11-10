"""Prompt builder for query normalization agent."""


class QueryNormalizationPrompts:
    """Build prompts that drive LLM-only normalization."""

    @staticmethod
    def build_prompt(query: str, prior_turns: list[dict] | None = None) -> str:
        """
        Build normalization prompt with conversation context.

        Args:
            query: Natural language query to normalize
            prior_turns: Optional list of prior conversation turns

        Returns:
            Formatted prompt for query normalization

        """
        clean_query = query.strip()

        # Build conversation context if provided
        context_section = ""
        if prior_turns and len(prior_turns) > 0:
            context_lines = ["\nCONVERSATION CONTEXT:"]

            # Use last 2-3 turns for context
            recent_turns = prior_turns[-3:]
            for i, turn in enumerate(recent_turns, 1):
                content = turn.get("content", "")
                sql = turn.get("sql", "")
                if content:
                    context_lines.append(f"\nTurn {i}: {content}")
                    if sql:
                        # Show abbreviated SQL for context
                        sql_preview = sql[:100] + ("..." if len(sql) > 100 else "")
                        context_lines.append(f"  SQL: {sql_preview}")

            context_lines.append("\nCURRENT REQUEST:")
            context_lines.append("The user is continuing the conversation above.")
            context_lines.append(
                "Use the context to understand ambiguous references like 'show more', 'change to', 'go back to first', etc."
            )
            context_lines.append(
                "If the current query seems unrelated to context, treat it as a fresh request.\n"
            )

            context_section = "\n".join(context_lines)

        return (
            "Normalize the business question into JSON with fields"
            " main_clause, details_for_filterings, required_visuals, tables."
            "\nThink through the transformation before responding:"
            "\n1. Understand the analytical intent and rewrite it as a concise directive"
            "   free of display verbs or schema/table/column names (main_clause)."
            "\n2. Identify every concrete filter (dates, IDs, numeric limits, segments)."
            "   Convert them into abstract phrases like 'for some time range' or"
            "   'with generalized filters' and list them in details_for_filterings."
            "\n3. Detect references to visuals; summarize them in required_visuals"
            "   (e.g., 'line chart') or set to null when absent."
            "\n4. Consider whether the user points to a business domain; if so, provide"
            "   a generalized descriptor in tables (e.g., ['sales dataset']); otherwise null."
            "\nWhen context is provided above, use it to resolve ambiguous references:"
            "\n- 'show me more' → refer to the same subject with higher limits"
            "\n- 'change to pie chart' → same data, different visualization"
            "\n- 'go back to first' → reference the initial query subject"
            "\nExamples of correct normalization:"
            "\n- 'Show me a list of sales transactions for Q4, including product details'"
            "   → main_clause: 'list of sales transactions'; details_for_filterings: ['for some time range']; tables: ['generalized dataset']"
            "\n- 'Display the number of active users by country over the past month'"
            "   → main_clause: 'number of active users'; details_for_filterings: ['for some time range']; tables: null"
            "\n- 'Get the total revenue for each department last year'"
            "   → main_clause: 'total revenue for each department'; details_for_filterings: ['for some time range']; tables: null"
            "\nKeep your reasoning internal and output only valid JSON adhering to"
            " the NormalizedQuery schema—no explanations."
            f"{context_section}"
            f"\n\nORIGINAL QUERY:\n{clean_query}"
        )
