"""
Pure SQL generation engine with injected dependencies.

Focused solely on SQL generation without execution or configuration concerns.
"""

from loguru import logger

from src.utils.langfuse_client import observe

from .types import (
    SQLGenerationResult,
    SQLGenerationStatus,
    ValidationResult,
    ValidationStatus,
)


class SQLEngine:
    """
    Pure SQL generation engine with dependency injection.

    Responsibilities:
    - Orchestrate agents for SQL generation
    - Return structured results
    - No configuration, connection, or execution logic
    """

    def __init__(
        self,
        rag_instance,
        table_agent,
        schema_validator,
        clarification_agent,
        filtering_agent=None,
        column_agent=None,
        query_normalizer=None,
    ):
        """
        Initialize with injected dependencies.

        Args:
            rag_instance: PgvectorRAG for context retrieval
            table_agent: TableAgent for table selection
            schema_validator: SchemaValidatorAgent for validation
            clarification_agent: ClarificationAgent for user guidance
            filtering_agent: Optional FilteringAgent
            column_agent: Optional ColumnAgent
            query_normalizer: Optional QueryNormalizer for query preprocessing

        """
        self.rag_instance = rag_instance
        self.table_agent = table_agent
        self.schema_validator = schema_validator
        self.clarification_agent = clarification_agent
        self.filtering_agent = filtering_agent
        self.column_agent = column_agent
        self.query_normalizer = query_normalizer

    @observe(name="sql_engine_generate")
    def generate_sql(
        self, question: str, prior_turns: list[dict] | None = None
    ) -> SQLGenerationResult:
        """
        Generate SQL using agent pipeline.

        Args:
            question: Natural language question
            prior_turns: Optional list of prior conversation turns

        Returns:
            SQLGenerationResult with status and data

        """
        try:
            logger.info(f"🔄 Generating SQL for: {question[:50]}...")

            normalized_payload = None
            normalized_question = question
            if self.query_normalizer:
                try:
                    normalized = self.query_normalizer.normalize(question, prior_turns)
                    normalized_payload = normalized.model_dump(exclude_none=True)
                    if normalized_payload.get("main_clause"):
                        normalized_question = self._compose_normalized_text(normalized)
                        if not normalized_question.strip():
                            normalized_question = question
                    logger.debug(
                        "Normalized query main clause: {}",
                        normalized_payload.get("main_clause", ""),
                    )
                except Exception as error:
                    logger.warning(f"Query normalization failed: {error}")

            # Step 1: Table selection
            try:
                table_selection_context = {}
                if normalized_payload:
                    table_selection_context["normalized_query"] = normalized_payload
                    table_selection_context["original_question"] = question

                table_selection = self.table_agent.select_tables(
                    normalized_question, table_selection_context
                )
            except Exception as e:
                logger.warning(f"Table selection failed: {e}")
                table_selection = None
            if not table_selection:
                return SQLGenerationResult(
                    status=SQLGenerationStatus.CLARIFICATION_NEEDED,
                    clarification_message="Could not identify relevant tables for your query. Please be more specific about what data you need.",
                    normalized_query=normalized_payload,
                )

            # Check if clarification is needed
            if self._needs_clarification(table_selection):
                clarification_msg = (
                    self.clarification_agent.generate_clarification_response(
                        table_selection, question
                    )
                )
                return SQLGenerationResult(
                    status=SQLGenerationStatus.CLARIFICATION_NEEDED,
                    clarification_message=clarification_msg,
                    normalized_query=normalized_payload,
                )

            # Step 2: Extract filters (optional)
            filter_context = self._extract_filters(
                question, table_selection, normalized_payload
            )

            # Step 3: Generate initial SQL
            initial_sql = self._generate_initial_sql(
                question, table_selection, filter_context, prior_turns
            )
            if not initial_sql:
                return SQLGenerationResult(
                    status=SQLGenerationStatus.ERROR,
                    error="Failed to generate SQL from context",
                    normalized_query=normalized_payload,
                )

            # Check if this is a clarification message
            is_clarification = initial_sql.startswith("CANNOT FIND TABLES:")

            # Step 4: Validate schema (optional - log warnings but don't block)
            # Skip validation for clarification messages
            if is_clarification:
                validation_result = ValidationResult(
                    status=ValidationStatus.VALID,  # Skip validation for clarification
                    error="Clarification message - validation skipped",
                )
                logger.info("Skipping schema validation for clarification message")
            else:
                validation_result = self._validate_schema(initial_sql, table_selection)

            validation_warnings = []
            if not validation_result.valid:
                # Build structured warning messages
                if validation_result.invalid_tables:
                    validation_warnings.append(
                        f"Invalid tables: {', '.join(validation_result.invalid_tables)}"
                    )
                if validation_result.invalid_columns:
                    validation_warnings.append(
                        f"Invalid columns: {', '.join(validation_result.invalid_columns)}"
                    )
                if validation_result.error:
                    validation_warnings.append(
                        f"Validation error: {validation_result.error}"
                    )

                logger.warning(f"Schema validation warnings: {validation_warnings}")
                logger.warning(
                    "SQL will be executed anyway - refinement will handle any issues"
                )

            # Always return the generated SQL - let execution and refinement handle issues
            return SQLGenerationResult(
                status=SQLGenerationStatus.SUCCESS,
                sql=initial_sql,
                schema_context=getattr(table_selection, "schema_context", ""),
                selected_tables=getattr(table_selection, "selected_tables", []),
                validation_warnings=validation_warnings
                if validation_warnings
                else None,
                normalized_query=normalized_payload,
            )

        except Exception as e:
            logger.error(f"Error in SQL generation: {e}")
            return SQLGenerationResult(
                status=SQLGenerationStatus.ERROR,
                error=str(e),
                normalized_query=locals().get("normalized_payload"),
            )

    def _compose_normalized_text(self, normalized) -> str:
        main = normalized.main_clause.strip() or "General business request"
        details = " ".join(normalized.details_for_filterings).strip()
        visuals = (normalized.required_visuals or "").strip()
        parts = [part for part in [main, details, visuals] if part]
        return "; ".join(parts)

    def _needs_clarification(self, table_selection) -> bool:
        """Check if clarification is needed."""
        if not self.clarification_agent:
            return False
        return self.clarification_agent.needs_clarification(table_selection)

    def _extract_filters(
        self,
        question: str,
        table_selection,
        normalized_hint: dict | None,
    ) -> dict | None:
        """Extract filters using filtering agents, leveraging normalized hints when available."""
        if not self.filtering_agent or not table_selection:
            return None

        try:
            filter_result = self.filtering_agent.extract_filters(
                question, normalized_hint
            )
            if filter_result and filter_result.get("filterings"):
                logger.info(f"✅ Extracted {len(filter_result['filterings'])} filters")

                # Apply column mapping if available
                if self.column_agent:
                    column_mapping_result = self.column_agent.map_columns(
                        filtering_result=filter_result,
                        selected_tables=table_selection.selected_tables,
                        schema_context=table_selection.schema_context,
                    )
                    if column_mapping_result and column_mapping_result.get(
                        "filterings"
                    ):
                        return {
                            "mapped_filters": column_mapping_result["filterings"],
                            "confidence": column_mapping_result.get(
                                "mapping_confidence", 0.5
                            ),
                        }

                return {
                    "semantic_filters": filter_result["filterings"],
                    "confidence": filter_result.get("confidence", 0.5),
                }
        except Exception as e:
            logger.warning(f"Filter extraction failed: {e}")

        return None

    def _generate_initial_sql(
        self,
        question: str,
        table_selection,
        filter_context: dict | None,
        prior_turns: list[dict] | None = None,
    ) -> str | None:
        """Generate initial SQL using RAG."""
        try:
            # Prepare context for RAG
            kwargs = {
                "selected_tables": table_selection.selected_tables,
                "schema_context": table_selection.schema_context,
            }

            if filter_context:
                kwargs["filter_context"] = filter_context

            # Get context from RAG
            context_with_examples = self.rag_instance.find_relevant_examples(
                question, **kwargs
            )
            if not isinstance(context_with_examples, dict):
                logger.error("Expected dict from find_relevant_examples")
                return None

            # Generate SQL using SQLWriterAgent
            from src.agents.strand_agents.sql.generator import SQLWriterAgent

            sql_writer = SQLWriterAgent()
            result = sql_writer.generate_sql(
                question=question,
                schema_context=context_with_examples.get("schema_context", []),
                example_context=context_with_examples.get("example_context", []),
                selected_tables=context_with_examples.get("selected_tables", []),
                search_results=context_with_examples.get("search_results", []),
                best_similarity=context_with_examples.get("best_similarity", 0.0),
                filter_context=filter_context
                or context_with_examples.get("filter_context"),
                prior_turns=prior_turns,
            )

            sql = result.get("sql", "") if isinstance(result, dict) else str(result)
            logger.info(f"✅ Initial SQL generated: {sql[:100]}...")
            return sql

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return None

    def _validate_schema(self, sql: str, table_selection) -> ValidationResult:
        """Validate SQL schema."""
        try:
            validation_result = self.schema_validator.validate_sql_tables(
                sql, schema_context=table_selection.schema_context
            )

            from .types import ValidationStatus

            if validation_result.validation_passed:
                return ValidationResult(status=ValidationStatus.VALID)
            else:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    valid_tables=validation_result.valid_tables,
                    invalid_tables=validation_result.invalid_tables,
                    invalid_columns=getattr(validation_result, "invalid_columns", None),
                )

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return ValidationResult(status=ValidationStatus.ERROR, error=str(e))
