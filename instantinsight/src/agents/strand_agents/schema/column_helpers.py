"""Helper classes for ColumnAgent to improve code organization and maintainability."""

import difflib
import json

from loguru import logger

from src.utils.langfuse_client import observe


class EmbeddingSearcher:
    """Handles embedding-based searches in the RAG cardinality table."""

    def __init__(self, pg_connection, bedrock_client, config):
        """
        Initialize the EmbeddingSearcher with database and model connections.

        Args:
            pg_connection: PostgreSQL database connection for cardinality queries
            bedrock_client: AWS Bedrock client for embedding generation
            config: Configuration object with embedding and search settings

        """
        self.pg_connection = pg_connection
        self.bedrock_client = bedrock_client
        self.config = config

    def search_with_columns(
        self,
        search_value: str,
        columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
    ) -> dict[str, list[str]]:
        """
        Search using embedding similarity, preserving column information.

        Returns:
            Dictionary mapping column names to their matching values

        """
        if not self.pg_connection:
            return {}

        values = self.search(search_value, columns_info, tables)

        # Group values under the first matching column
        if values:
            for table in tables:
                for col_dict in columns_info.get(table, []):
                    return {col_dict["name"]: values}
        return {}

    @observe(as_type="generation")
    def search(
        self,
        search_value: str,
        columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
    ) -> list[str]:
        """
        Search rag_cardinality table using embedding similarity.

        Returns:
            List of top matching category values

        """
        if not self.pg_connection:
            return []

        # Prepare search value
        search_value = self._prepare_search_value(search_value)
        if not search_value:
            return []

        # Generate embedding
        search_embedding = self._generate_embedding(search_value)
        if not search_embedding:
            logger.debug(f"Could not generate embedding for '{search_value}'")
            return []

        try:
            # Build table-column pairs and check availability
            target_pairs = self._build_target_pairs(columns_info, tables)
            if not target_pairs:
                return []

            logger.debug(
                f"Checking availability for {len(target_pairs)} table-column pairs"
            )
            available_pairs = self._check_pairs_availability(target_pairs)
            if not available_pairs:
                logger.debug("No available pairs found in rag_cardinality_columns")
                return []

            logger.debug(
                f"Executing embedding search for {len(available_pairs)} available pairs"
            )
            # Execute search and process results
            results = self._execute_embedding_search(search_embedding, available_pairs)
            return self._process_search_results(results)

        except Exception as e:
            import traceback

            logger.error(f"Error searching rag_cardinality by embedding: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # NOTE: Rollback transaction on error to prevent "aborted transaction" state
            if self.pg_connection:
                try:
                    self.pg_connection.rollback()
                    logger.warning("Rolled back PostgreSQL transaction after error")
                except Exception as rollback_error:
                    logger.warning(f"Failed to rollback transaction: {rollback_error}")
            return []

    def _prepare_search_value(self, search_value: str) -> str:
        """Sanitize and normalize search value."""
        search_value = search_value.strip().lower()
        if not search_value:
            logger.debug("Empty search value after sanitization")
        return search_value

    def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using Bedrock Titan model."""
        if not self.bedrock_client:
            return None

        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.config.embedding_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text}),
            )

            response_body = json.loads(response["body"].read())
            return response_body.get("embedding")

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _build_target_pairs(
        self, columns_info: dict[str, list[dict[str, str]]], tables: list[str]
    ) -> list[tuple[str, str]]:
        """Build list of table-column pairs to search."""
        target_pairs = []
        for table in tables:
            if table in columns_info:
                for column_info in columns_info[table]:
                    target_pairs.append((table, column_info["name"]))

        if not target_pairs:
            logger.debug(f"No columns found in columns_info for tables: {tables}")

        return target_pairs

    def _check_pairs_availability(
        self, target_pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Check which table-column pairs exist in rag_cardinality_columns."""
        try:
            with self.pg_connection.cursor() as cur:
                # Build query conditions
                pair_conditions = []
                pair_params = []
                for table_name, column_name in target_pairs:
                    pair_conditions.append("(table_name = %s AND column_name = %s)")
                    pair_params.extend([table_name, column_name])

                pair_check_condition = " OR ".join(pair_conditions)

                cur.execute(
                    f"""
                    SELECT DISTINCT table_name, column_name
                    FROM rag_cardinality_columns
                    WHERE {pair_check_condition}
                    """,
                    pair_params,
                )

                available_pairs = cur.fetchall()

                if not available_pairs:
                    logger.debug(
                        f"No table-column pairs from columns_info found in rag_cardinality_columns. "
                        f"Checked {len(target_pairs)} pairs: {target_pairs}"
                    )
                else:
                    logger.debug(
                        f"Found {len(available_pairs)} table-column pairs in rag_cardinality_columns "
                        f"out of {len(target_pairs)} checked"
                    )

                return available_pairs
        except Exception as e:
            import traceback

            logger.error(f"Error checking pair availability: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return []

    def _execute_embedding_search(
        self, search_embedding: list[float], available_pairs: list[tuple[str, str]]
    ) -> list:
        """Execute the embedding similarity search query."""
        try:
            with self.pg_connection.cursor() as cur:
                # Build condition for available pairs
                final_pair_conditions = []
                final_pair_params = []
                for table_name, column_name in available_pairs:
                    final_pair_conditions.append(
                        "(table_name = %s AND column_name = %s)"
                    )
                    final_pair_params.extend([table_name, column_name])

                pair_condition = " OR ".join(final_pair_conditions)

                # Execute pgvector cosine similarity search
                cur.execute(
                    f"""
                    SELECT category, category_norm, schema_name, table_name, column_name,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM rag_cardinality
                    WHERE embedding IS NOT NULL
                    AND ({pair_condition})
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [search_embedding]
                    + final_pair_params
                    + [search_embedding, self.config.top_k_similar],
                )

                return cur.fetchall()
        except Exception as e:
            import traceback

            logger.error(f"Error executing embedding search: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # NOTE: Rollback transaction to prevent "aborted transaction" state
            if self.pg_connection:
                try:
                    self.pg_connection.rollback()
                    logger.warning(
                        "Rolled back PostgreSQL transaction after search error"
                    )
                except Exception as rollback_error:
                    logger.warning(f"Failed to rollback transaction: {rollback_error}")
            return []

    def _process_search_results(self, results: list) -> list[str]:
        """Process search results and extract category values."""
        category_values = []

        for row in results:
            category, category_norm, schema, table, column, similarity = row

            # Only include results above similarity threshold
            if similarity >= self.config.similarity_threshold:
                value = category
                if value and value not in category_values:
                    category_values.append(value)
                    logger.debug(
                        f"Found similar value: '{value}' (similarity: {similarity:.3f}) "
                        f"from {schema}.{table}.{column}"
                    )

        logger.info(
            f"Embedding search found {len(category_values)} similar values "
            f"above threshold {self.config.similarity_threshold}"
        )

        return category_values


class FuzzyMatcher:
    """Handles fuzzy matching of categorical values."""

    @staticmethod
    def match_values(
        search_term: str,
        candidate_values: list[str],
        similarity_threshold: float = 0.2,
        max_results: int = 10,
    ) -> list[str]:
        """
        Use fuzzy matching to find values most similar to search term.

        Returns:
            List of fuzzy matched values sorted by similarity

        """
        if not search_term or not candidate_values:
            return []

        search_term_lower = search_term.lower().strip()
        scored_matches = []

        for value in candidate_values:
            if not value:
                continue

            value_lower = value.lower().strip()

            # Calculate similarity scores
            similarity = FuzzyMatcher._calculate_similarity(
                search_term_lower, value_lower
            )

            if similarity >= similarity_threshold:
                scored_matches.append((value, similarity))

        # Sort by similarity score and return top matches
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in scored_matches[:max_results]]

    @staticmethod
    def _calculate_similarity(search_term: str, value: str) -> float:
        """Calculate similarity between search term and value."""
        # Sequence matcher for overall similarity
        seq_similarity = difflib.SequenceMatcher(None, search_term, value).ratio()

        # Check for substring matches (higher weight)
        substring_bonus = 0.0
        if search_term in value or value in search_term:
            substring_bonus = 0.2

        # Check for word matches
        search_words = set(search_term.split())
        value_words = set(value.split())
        if search_words and value_words:
            word_overlap = len(search_words.intersection(value_words)) / len(
                search_words.union(value_words)
            )
        else:
            word_overlap = 0.0

        # Combined similarity score
        final_similarity = max(seq_similarity, word_overlap) + substring_bonus
        return min(final_similarity, 1.0)  # Cap at 1.0


class NumericalColumnDetector:
    """Detects whether columns are likely numerical based on name and type."""

    NUMERICAL_TYPES = [
        "int",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "decimal",
        "numeric",
        "float",
        "double",
        "real",
        "money",
        "smallmoney",
        "number",
    ]

    NUMERICAL_INDICATORS = [
        "amount",
        "price",
        "cost",
        "total",
        "sum",
        "count",
        "quantity",
        "qty",
        "balance",
        "rate",
        "percentage",
        "percent",
        "pct",
        "score",
        "value",
        "fee",
        "charge",
        "payment",
        "revenue",
        "profit",
        "loss",
        "tax",
        "discount",
        "salary",
        "wage",
        "income",
        "expense",
        "budget",
    ]

    @classmethod
    def is_numerical(cls, column_name: str, column_type: str = None) -> bool:
        """
        Determine if a column is likely numerical.

        Returns:
            True if column is likely numerical, False if categorical

        """
        column_lower = column_name.lower()

        # Check column type if provided
        if column_type:
            type_lower = column_type.lower()
            for num_type in cls.NUMERICAL_TYPES:
                if num_type in type_lower:
                    logger.debug(
                        f"Column '{column_name}' has numerical type '{column_type}'"
                    )
                    return True

        # Check for numerical indicators in column name
        for indicator in cls.NUMERICAL_INDICATORS:
            if indicator in column_lower:
                logger.debug(
                    f"Column '{column_name}' contains '{indicator}' - treating as numerical"
                )
                return True

        # Default to categorical
        logger.debug(
            f"Column '{column_name}' with type '{column_type}' - treating as categorical"
        )
        return False


class CategoricalValueRetriever:
    """Handles the 3-step process for retrieving categorical values."""

    def __init__(self, agent):
        """Initialize with reference to parent ColumnAgent."""
        self.agent = agent
        self.config = agent.config
        self.rag_engine = agent.rag_engine
        self.embedding_searcher = EmbeddingSearcher(
            agent.pg_connection, agent.bedrock_client, agent.config
        )
        self.fuzzy_matcher = FuzzyMatcher()

    def retrieve_values(
        self,
        filter_key: str,
        filter_value: str,
        columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
        schema_context: str | None = None,
        question: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Execute the 3-step process to get categorical values.

        Returns:
            Dictionary mapping column names to their categorical values

        """
        # Step 0: Identify and filter candidate columns
        filtered_columns_info = self._prepare_columns(
            filter_key, columns_info, schema_context, question
        )

        if not filtered_columns_info:
            logger.info(f"No categorical columns found for filter key '{filter_key}'")
            return {}

        column_values_map = {}

        # Step 1: Try embedding search
        column_values_map = self._try_embedding_search(
            filter_value, filtered_columns_info, tables, column_values_map
        )
        if column_values_map:
            return column_values_map

        # Step 2: Try database cardinality table
        column_values_map = self._try_db_cardinality(
            filter_key, filter_value, filtered_columns_info, tables, column_values_map
        )
        if column_values_map:
            return column_values_map

        # Step 3: Try direct SELECT DISTINCT
        column_values_map = self._try_direct_query(
            filter_key, filter_value, filtered_columns_info, tables, column_values_map
        )

        if not column_values_map:
            logger.info(
                f"All 3 steps failed for filter '{filter_key}' = '{filter_value}'"
            )

        return column_values_map

    def _prepare_columns(
        self,
        filter_key: str,
        columns_info: dict[str, list[dict[str, str]]],
        schema_context: str | None,
        question: str | None,
    ) -> dict[str, list[dict[str, str]]]:
        """Step 0: Identify and filter candidate columns."""
        logger.debug(f"Step 0: Finding candidate columns for '{filter_key}'")

        # Get or cache candidate columns
        if filter_key not in self.agent.candidate_columns:
            result = self.agent._identify_candidate_columns(
                filter_key, columns_info, schema_context, question
            )
            # Handle both dict with 'candidates' key and direct dict response
            if isinstance(result, dict) and "candidates" in result:
                candidate_columns = result["candidates"]
            else:
                candidate_columns = result
        else:
            candidate_columns = self.agent.candidate_columns[filter_key]
            logger.debug(
                f"Using cached candidate columns for '{filter_key}': {candidate_columns}"
            )

        # Filter to only categorical columns
        filtered_columns_info = self.agent._filter_columns_info_by_candidates(
            columns_info, candidate_columns
        )

        if filtered_columns_info:
            logger.info(
                f"Step 0 SUCCESS: Filtered to categorical candidate columns for '{filter_key}'"
            )

        return filtered_columns_info

    def _try_embedding_search(
        self,
        filter_value: str,
        filtered_columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
        column_values_map: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Step 1: Try embedding search."""
        logger.debug(f"Step 1: Trying embedding search for '{filter_value}'")

        embedding_results = self.embedding_searcher.search_with_columns(
            filter_value, filtered_columns_info, tables
        )

        if embedding_results:
            for column_name, values in embedding_results.items():
                if values:
                    column_values_map[column_name] = values
                    logger.info(
                        f"Step 1 SUCCESS: Found {len(values)} values for column '{column_name}'"
                    )

        return column_values_map

    def _try_db_cardinality(
        self,
        filter_key: str,
        filter_value: str,
        filtered_columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
        column_values_map: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Step 2: Try database cardinality table."""
        import os

        MAX_FUZZY_MATCH_RESULTS = int(os.getenv("MAX_FUZZY_MATCH_RESULTS", 20))

        logger.debug(f"Step 2: Trying database cardinality table for '{filter_key}'")

        for table in tables:
            for col_dict in filtered_columns_info.get(table, []):
                column_name = col_dict["name"]
                db_values = self.agent._check_column_cardinality(table, column_name)

                if db_values:
                    # Apply fuzzy matching
                    fuzzy_matches = self.fuzzy_matcher.match_values(
                        filter_value, db_values, max_results=MAX_FUZZY_MATCH_RESULTS
                    )

                    if fuzzy_matches:
                        column_values_map[column_name] = fuzzy_matches
                        logger.info(
                            f"Step 2 SUCCESS: Found {len(fuzzy_matches)} fuzzy matched values for '{column_name}'"
                        )
                    else:
                        # Fallback to top N values
                        column_values_map[column_name] = db_values[
                            :MAX_FUZZY_MATCH_RESULTS
                        ]
                        logger.info(
                            f"Step 2 PARTIAL: Found {len(column_values_map[column_name])} values for '{column_name}'"
                        )

        return column_values_map

    def _try_direct_query(
        self,
        filter_key: str,
        filter_value: str,
        filtered_columns_info: dict[str, list[dict[str, str]]],
        tables: list[str],
        column_values_map: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Step 3: Try direct SELECT DISTINCT query."""
        import os

        MAX_FUZZY_MATCH_RESULTS = int(os.getenv("MAX_FUZZY_MATCH_RESULTS", 20))

        logger.debug(f"Step 3: Trying direct SELECT DISTINCT for '{filter_key}'")

        for table in tables:
            for col_dict in filtered_columns_info.get(table, []):
                column_name = col_dict["name"]
                distinct_values = self.agent._query_distinct_values(
                    table, column_name, filter_value
                )

                if distinct_values:
                    # Apply fuzzy matching
                    fuzzy_matches = self.fuzzy_matcher.match_values(
                        filter_value,
                        distinct_values,
                        max_results=MAX_FUZZY_MATCH_RESULTS,
                    )

                    if fuzzy_matches:
                        column_values_map[column_name] = fuzzy_matches
                        logger.info(
                            f"Step 3 SUCCESS: Found {len(fuzzy_matches)} fuzzy matches for '{column_name}'"
                        )
                    else:
                        # Return top values if no good matches
                        column_values_map[column_name] = distinct_values[
                            :MAX_FUZZY_MATCH_RESULTS
                        ]
                        logger.info(
                            f"Step 3 PARTIAL: Found values for '{column_name}' but no good fuzzy matches"
                        )

        return column_values_map
