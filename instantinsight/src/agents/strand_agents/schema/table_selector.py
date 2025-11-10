"""
TableAgent for Text2SQL Table Selection using Strand framework.

Intelligently selects and prunes relevant tables from large database schemas based on
natural language queries using LLM with structured outputs.
"""

from typing import Any

try:  # pragma: no cover - optional dependency
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover
    dict_row = None  # type: ignore

from loguru import logger
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.prompts import Prompts
from src.agents.prompt_builders.schema.table_selector import SchemaTableSelectorPrompts
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    log_prompt_cache_status,
    update_langfuse_with_usage,
)
from src.utils.table_agent_utils import (
    filter_schema_by_selected_tables,
    format_table_info_text,
)

from ..model_config import get_agent_config
from .table_hint_collector import SchemaHintCollector
from .table_selector_models import (
    LLMTableSelectionResult,
    TableSelectionResult,
    clean_schema,
)


class TableAgent:
    """
    Table selection agent using Strand framework.

    Intelligently selects and prunes relevant tables from large database schemas based on
    natural language queries using LLM with structured outputs.
    """

    def __init__(
        self,
        pgvector_rag=None,
        config: dict[str, Any] | None = None,
        aws_region: str = None,
        model_id: str = None,
    ):
        """
        Initialize TableAgent.

        Args:
            pgvector_rag: PgvectorRAG instance for vector similarity search
            config: Configuration options for table selection
            aws_region: AWS region for Bedrock (used for config only)
            model_id: Bedrock model ID (optional, uses config default)

        """
        self.rag = pgvector_rag
        self.config = config or {}

        # Configuration parameters - safe defaults
        self.max_tables = self.config.get("max_selected_tables", 1)
        self.min_confidence = self.config.get("min_confidence_threshold", 0.5)
        self.vector_search_k = self.config.get("vector_search_k", 10)
        self.include_related_tables = self.config.get("include_related_tables", False)
        self.allow_unsafe_joins = self.config.get("allow_unsafe_joins", False)
        self.keyword_weight = self.config.get("keyword_weight", 0.3)
        self.semantic_weight = self.config.get("semantic_weight", 0.5)
        self.usage_weight = self.config.get("usage_weight", 0.2)
        self.similarity_threshold = self.config.get(
            "schema_similarity_threshold",
            self.config.get("similarity_threshold", 0.2),
        )
        self.example_search_k = max(1, min(self.config.get("example_search_k", 20), 50))
        self.example_similarity_threshold = self.config.get(
            "example_similarity_threshold", 0.2
        )
        default_chunk_types = ["example_overview", "example_ctes"]
        configured_chunk_types = self.config.get("example_chunk_types")
        if isinstance(configured_chunk_types, list):
            valid_chunk_types = [
                str(chunk_type)
                for chunk_type in configured_chunk_types
                if isinstance(chunk_type, str)
            ]
            self.example_chunk_types = valid_chunk_types or default_chunk_types
        else:
            self.example_chunk_types = default_chunk_types
        self.term_match_enabled = self.config.get("term_match_enabled", True)
        self.term_match_top_k = max(1, self.config.get("term_match_top_k", 5))
        self.term_match_min_score = max(
            0.0, self.config.get("term_match_min_score", 0.5)
        )
        self.term_match_fuzzy_threshold = min(
            1.0, max(0.1, self.config.get("term_match_fuzzy_threshold", 0.78))
        )

        # Initialize Strand agent for LLM-based table selection
        self.debug_mode = self.config.get("debug_mode", True)
        agent_config = get_agent_config("SchemaTableSelector", aws_region)

        self.aws_region = aws_region or agent_config["aws_region"]
        self.model_id = model_id or agent_config["model_id"]

        # Create Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
            streaming=False,
            cache_prompt=agent_config.get("cache_prompt"),
        )

        self._cache_prompt_type = agent_config.get("cache_prompt")
        self._initialise_agent()

        self.hint_collector = SchemaHintCollector(self)

        logger.info(f"✓ TableAgent (strand) initialized with config: {self.config}")

    def _initialise_agent(self):
        """Initialise the Strand agent."""
        # Load cached system prompt optimized for prompt caching
        base_instructions = Prompts.SCHEMA_TABLE_SELECTOR

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        self.agent = Agent(
            model=self.model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

    @observe(as_type="generation")
    def _run_selection(
        self,
        query: str,
        schema_documents: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the Strand agent to analyse schema tables and return structured output.

        Args:
            query: Natural language query
            schema_documents: List of schema documents with content and metadata
            context: Optional context from previous pipeline stages

        Returns:
            Dictionary with selected tables and analysis metadata

        """
        if not schema_documents:
            return {
                "selected_tables": [],
                "reasoning": "No schema documents available for analysis",
                "confidence_scores": {},
                "schema_context": "",
            }

        schema_context = self._prepare_schema_context(schema_documents)
        context_data = context or {}
        term_match_tables = context_data.get("term_match_tables") or []
        term_match_scores = context_data.get("term_match_scores") or {}
        example_hints = context_data.get("example_table_hints") or {}

        hint_sections = []
        term_hint_block = self.hint_collector.format_term_match_hints(
            term_match_tables, term_match_scores
        )
        if term_hint_block:
            hint_sections.append(term_hint_block)

        vector_hint_block = self.hint_collector.format_vector_search_hints(context)
        if vector_hint_block:
            hint_sections.append(vector_hint_block)

        example_hint_block = self.hint_collector.format_example_hints(context)
        if example_hint_block:
            hint_sections.append(example_hint_block)

        normalized_table_block = self.hint_collector.format_normalized_table_hints(
            context
        )
        if normalized_table_block:
            hint_sections.append(normalized_table_block)

        if hint_sections:
            combined_hint_text = "\n\n".join(hint_sections)
            schema_context = (
                f"{combined_hint_text}\n\n## ALL TABLES:\n{schema_context}"
                if schema_context
                else combined_hint_text
            )

        # Build prompt
        prompt = SchemaTableSelectorPrompts.build_table_selection_prompt(
            query, schema_context
        )

        try:
            # Reset usage
            self._usage_container["last_usage"] = None

            # Run agent with structured output
            llm_result = self.agent.structured_output(LLMTableSelectionResult, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "TableAgent",
                langfuse_context,
            )

            if self._cache_prompt_type:
                log_prompt_cache_status("TableAgent", self._usage_container)

            if llm_result and isinstance(llm_result, LLMTableSelectionResult):
                selected_tables = [
                    analysis.table_name for analysis in llm_result.selected_tables
                ]
                confidence_scores = {
                    analysis.table_name: analysis.confidence_score
                    for analysis in llm_result.selected_tables
                }

                filtered_schema = filter_schema_by_selected_tables(
                    schema_context, selected_tables
                )

                join_paths = []
                for join in llm_result.suggested_joins or []:
                    if (
                        join.is_necessary
                        and hasattr(join, "relationship_verified")
                        and join.relationship_verified
                        and hasattr(join, "data_integrity_risk")
                        and join.data_integrity_risk == "LOW"
                    ):
                        join_paths.append(
                            {
                                "table1": join.table1,
                                "table2": join.table2,
                                "join_type": join.join_type,
                                "join_condition": join.join_condition,
                                "confidence": join.confidence,
                                "is_necessary": join.is_necessary,
                                "relationship_verified": join.relationship_verified,
                                "data_integrity_risk": join.data_integrity_risk,
                                "alternative_single_table": getattr(
                                    join, "alternative_single_table", None
                                ),
                            }
                        )
                    elif join.is_necessary:
                        risk = getattr(join, "data_integrity_risk", "UNKNOWN")
                        verified = getattr(join, "relationship_verified", False)
                        logger.warning(
                            f"🚨 REJECTED DANGEROUS JOIN: {join.table1} ↔ {join.table2} "
                            f"(Risk: {risk}, Verified: {verified})"
                        )

                duplicate_info = None
                if (
                    llm_result.duplicate_analysis
                    and llm_result.duplicate_analysis.confidence >= 0.7
                ):
                    duplicate_info = {
                        "similar_tables": llm_result.duplicate_analysis.similar_tables,
                        "recommended_table": llm_result.duplicate_analysis.recommended_table,
                        "similarity_reason": llm_result.duplicate_analysis.similarity_reason,
                        "confidence": llm_result.duplicate_analysis.confidence,
                    }

                    if (
                        llm_result.duplicate_analysis.recommended_table
                        in selected_tables
                    ):
                        logger.info(
                            f"Duplicates detected, reducing to single table: {llm_result.duplicate_analysis.recommended_table}"
                        )
                        selected_tables = [
                            llm_result.duplicate_analysis.recommended_table
                        ]
                        join_paths = []
                        recommended_table = (
                            llm_result.duplicate_analysis.recommended_table
                        )
                        confidence_scores = {
                            recommended_table: confidence_scores.get(
                                recommended_table, 0.9
                            )
                        }

                reasoning = f"""LLM Analysis Results:

Query Understanding: {llm_result.query_analysis}

Selected Tables ({len(selected_tables)}):
"""
                for analysis in llm_result.selected_tables:
                    reasoning += f"• {analysis.table_name} (confidence: {analysis.confidence_score:.2f})\n"
                    reasoning += f"  - Purpose: {analysis.table_purpose}\n"
                    reasoning += f"  - Key columns: {', '.join(analysis.key_columns)}\n"
                    reasoning += f"  - Reasoning: {analysis.relevance_reasoning}\n\n"

                if join_paths:
                    reasoning += f"Necessary Joins ({len(join_paths)}):\n"
                    for join in join_paths:
                        necessity = " [NECESSARY]" if join.get("is_necessary") else ""
                        reasoning += f"• {join['table1']} → {join['table2']} ({join['join_type']}){necessity}\n"
                        reasoning += f"  - Condition: {join['join_condition']}\n\n"

                if duplicate_info:
                    reasoning += "Duplicate Analysis:\n"
                    reasoning += f"• Similar tables detected: {', '.join(duplicate_info['similar_tables'])}\n"
                    reasoning += (
                        f"• Recommended table: {duplicate_info['recommended_table']}\n"
                    )
                    reasoning += f"• Reason: {duplicate_info['similarity_reason']}\n\n"

                reasoning += f"Selection Strategy: {llm_result.selection_reasoning}\n"
                reasoning += f"Query Complexity: {llm_result.complexity_assessment}"

                if example_hints.get("example_tables"):
                    tables_summary = ", ".join(example_hints["example_tables"][:10])
                    reasoning += (
                        "\nHistorical SQL examples highlighted tables: "
                        f"{tables_summary}"
                    )
                if term_match_tables:
                    summary_items = []
                    for table in term_match_tables[:10]:
                        score_value = term_match_scores.get(table)
                        if isinstance(score_value, int | float):
                            summary_items.append(f"{table} ({score_value:.2f})")
                        else:
                            summary_items.append(table)
                    reasoning += (
                        f"\nInitial keyword matches: {', '.join(summary_items)}"
                    )

                result_dict = {
                    "selected_tables": selected_tables,
                    "confidence_scores": confidence_scores,
                    "related_tables": llm_result.related_tables or [],
                    "join_paths": join_paths,
                    "duplicate_analysis": duplicate_info,
                    "requires_multiple_tables": llm_result.requires_multiple_tables,
                    "reasoning": reasoning,
                    "llm_response": llm_result,
                    "schema_context": filtered_schema,
                }

                if example_hints:
                    result_dict["historical_example_tables"] = example_hints.get(
                        "example_tables", []
                    )
                    result_dict["historical_example_questions"] = example_hints.get(
                        "example_questions", []
                    )
                if term_match_tables:
                    result_dict["term_match_tables"] = term_match_tables
                    result_dict["term_match_scores"] = term_match_scores

                result_dict["llm_response"] = llm_result
                return result_dict
            else:
                return {
                    "selected_tables": [],
                    "reasoning": "Response not in expected format",
                    "confidence_scores": {},
                    "schema_context": "",
                }

        except Exception as e:
            logger.error(f"Table selection failed: {e}")
            return {
                "selected_tables": [],
                "reasoning": f"Table selection failed: {str(e)}",
                "confidence_scores": {},
                "schema_context": "",
            }

    def _prepare_schema_context(self, schema_documents: list[dict[str, Any]]) -> str:
        """Prepare schema context for LLM, limiting size to prevent overflow."""
        formatted_docs = []
        for doc in schema_documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if "CREATE EXTERNAL TABLE" in content:
                formatted_docs.append(format_table_info_text(content, metadata))

        return "\n\n".join(formatted_docs)

    def select_tables(
        self, query: str, context: dict[str, Any] | None = None
    ) -> TableSelectionResult:
        """
        Orchestrate the entire table selection process.

        Args:
            query: Natural language query
            context: Optional context from previous pipeline stages

        Returns:
            TableSelectionResult: Selected tables with metadata and reasoning

        """
        logger.info(f"Starting table selection for query: {query[:100]}...")

        context = dict(context or {})
        normalized_hint = context.get("normalized_query")
        # Step 1: collecting hints from term matching, historical examples, and vector store
        hint_data = self.hint_collector.collect(
            query=query, normalized_hint=normalized_hint, context=context
        )
        schema_documents = hint_data["schema_documents"]
        all_schema_documents = hint_data["all_schema_documents"]
        term_match_info = hint_data["term_match_info"]
        example_hints = hint_data["example_hints"]

        if not schema_documents:
            return TableSelectionResult(
                selection_reasoning="No schema documents available for analysis",
                total_tables_analyzed=0,
            )

        try:
            # Step 2: Use agent to intelligently analyze and select relevant tables
            selected_tables_result = self._run_selection(
                query, schema_documents, context
            )

            selected_tables = selected_tables_result.get("selected_tables", [])

            # Step 2b: Enforce join safety based on metadata
            original_tables = selected_tables.copy()
            selected_tables = self._enforce_join_safety(
                selected_tables, schema_documents
            )

            confidence_scores = selected_tables_result.get("confidence_scores", {})
            reasoning = selected_tables_result.get(
                "reasoning", "LLM analysis completed"
            )

            # Update reasoning if tables were reduced
            if len(selected_tables) < len(original_tables):
                reasoning += f" [Safety: Reduced from {original_tables} to {selected_tables} due to join restrictions]"
            related_tables = selected_tables_result.get("related_tables", [])
            schema_context = selected_tables_result.get("schema_context", "")

            # Clean and format the schema context
            if schema_context:
                schema_context = clean_schema(schema_context)

            # Step 3: Analyze join paths from LLM results
            join_paths = selected_tables_result.get("join_paths", [])
            duplicate_analysis = selected_tables_result.get("duplicate_analysis", None)
            requires_multiple_tables = selected_tables_result.get(
                "requires_multiple_tables", False
            )

            result = TableSelectionResult(
                selected_tables=selected_tables,
                related_tables=related_tables,
                join_paths=join_paths,
                duplicate_analysis=duplicate_analysis,
                requires_multiple_tables=requires_multiple_tables,
                confidence_scores=confidence_scores,
                selection_reasoning=reasoning,
                metadata={
                    "query": query,
                    "total_schemas": len(all_schema_documents),
                    "strategy": "llm_intelligent_selection",
                    "example_questions": example_hints.get("example_questions", []),
                    "example_tables": example_hints.get("example_tables", []),
                    "term_match_tables": term_match_info.get("matched_tables", []),
                    "term_match_scores": term_match_info.get("scores", {}),
                    "term_match_original_count": term_match_info.get("original_count"),
                    "term_match_filtered_count": term_match_info.get("filtered_count"),
                    "vector_search_tables": context.get("vector_search_tables", []),
                },
                strategy_used="llm_intelligent_selection",
                total_tables_analyzed=len(schema_documents),
                schema_context=schema_context,
            )

            logger.info(f"✓ Selected {len(selected_tables)} tables: {selected_tables}")
            return result

        except Exception as e:
            logger.error(f"Error in table selection: {e}")
            return TableSelectionResult(
                selection_reasoning=f"Error during table selection: {e}",
                total_tables_analyzed=0,
            )

    def _enforce_join_safety(
        self, selected_tables: list[str], schema_documents: list[dict[str, Any]]
    ) -> list[str]:
        """
        Enforce join safety rules based on metadata.

        Supports transitive joins through intermediate tables.
        If multiple tables with "DO NOT JOIN" are selected, keep only the best one.

        Returns:
            List[str]: Safe table selection (may be reduced from original)

        """
        if len(selected_tables) <= 1:
            return selected_tables

        # Build metadata lookup
        table_metadata = {}
        for doc in schema_documents:
            metadata = doc.get("metadata", {})
            table_name = metadata.get("table_name")
            if table_name and table_name in selected_tables:
                table_metadata[table_name] = metadata

        # Build adjacency map for join relationships
        join_graph = {}
        for table in selected_tables:
            metadata = table_metadata.get(table, {})
            if metadata.get("is_analyser"):
                safe_tables = metadata.get("safe_to_join_with", [])
                if not safe_tables:  # Empty means DO NOT JOIN
                    # This table cannot join with anything
                    logger.warning(f"Table {table} has no safe join partners")
                    # Keep only the first/best table if this is included
                    if table == selected_tables[0]:
                        return [table]
                    else:
                        # Remove this table from selection
                        return [t for t in selected_tables if t != table]
                join_graph[table] = set(safe_tables) & set(selected_tables)
            else:
                # Non-analyser tables can join with anything (backwards compatibility)
                join_graph[table] = set(selected_tables) - {table}

        # Check if all tables are reachable through direct or transitive joins
        def can_reach(from_table, to_table, visited=None):
            """Check if from_table can reach to_table through the join graph."""
            if visited is None:
                visited = set()

            if from_table == to_table:
                return True

            if from_table in visited:
                return False

            visited.add(from_table)

            # Direct connection
            if to_table in join_graph.get(from_table, set()):
                return True

            # Transitive connection through intermediate tables
            for intermediate in join_graph.get(from_table, set()):
                if can_reach(intermediate, to_table, visited.copy()):
                    return True

            return False

        # Verify all tables can be connected
        can_join = True
        for i, table1 in enumerate(selected_tables):
            for j, table2 in enumerate(selected_tables):
                if i != j:
                    # Check bidirectional reachability (either direction is fine)
                    if not (can_reach(table1, table2) or can_reach(table2, table1)):
                        logger.warning(
                            f"No join path found between {table1} and {table2}"
                        )
                        can_join = False
                        break
            if not can_join:
                break

        if not can_join:
            # Multiple tables cannot join - keep only the first/best one
            logger.warning(
                f"Tables {selected_tables} cannot all be joined. Keeping only {selected_tables[0]}"
            )
            return [selected_tables[0]]

        logger.info(f"✓ Join safety validated for tables: {selected_tables}")
        return selected_tables


# Global agent instance
_table_agent = None


def get_table_agent(
    pgvector_rag=None,
    config: dict[str, Any] | None = None,
) -> TableAgent:
    """Get the global table agent instance."""
    global _table_agent
    if _table_agent is None:
        _table_agent = TableAgent(pgvector_rag, config)
    return _table_agent
