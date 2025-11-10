"""Helper module for collecting table selection hints from various sources."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

try:
    from psycopg.rows import dict_row
except ImportError:
    dict_row = None  # type: ignore

from loguru import logger

from .table_selector_utils import (
    extract_columns_from_doc,
    extract_example_question,
    extract_example_sql,
    extract_tables_from_sql,
    score_match,
    split_identifier,
    tokenize_query,
)

if TYPE_CHECKING:
    from .table_selector import TableAgent


GENERIC_DETAIL_PHRASES = {
    "for some time range",
    "with generalized filters",
    "across relevant datasets",
}


class SchemaHintCollector:
    """Encapsulates hint gathering for TableAgent."""

    def __init__(self, agent: TableAgent):
        """Initialize hint collector with parent TableAgent."""
        self.agent = agent

    def collect(
        self,
        query: str,
        normalized_hint: Any | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather schema documents plus auxiliary hint metadata."""
        term_match_query = self._prepare_term_match_query(query, normalized_hint)
        schema_documents = self._get_all_schema_documents()
        all_schema_documents = list(schema_documents)

        schema_documents, term_match_info = self._apply_term_matching(
            term_match_query, schema_documents, context
        )

        example_hints = self._apply_example_hints(query, context)

        schema_documents = self._apply_vector_search(
            query, schema_documents, all_schema_documents, context
        )

        return {
            "schema_documents": schema_documents,
            "all_schema_documents": all_schema_documents,
            "term_match_info": term_match_info,
            "example_hints": example_hints,
        }

    def _apply_term_matching(
        self,
        term_match_query: str,
        schema_documents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        context["term_match_query"] = term_match_query
        term_match_info: dict[str, Any] = {}

        if not schema_documents or not self.agent.term_match_enabled:
            return schema_documents, term_match_info

        filtered_docs, match_info = self._filter_schema_by_term_matching(
            term_match_query, schema_documents
        )
        if match_info:
            term_match_info = match_info

        if filtered_docs:
            original_count = term_match_info.get(
                "original_count", len(schema_documents)
            )
            logger.info(
                "✓ Term matching reduced schema set from {} to {}",
                original_count,
                len(filtered_docs),
            )
            schema_documents = filtered_docs

        if term_match_info:
            context["term_match_tables"] = term_match_info.get("matched_tables", [])
            context["term_match_scores"] = term_match_info.get("scores", {})

        return schema_documents, term_match_info

    def _apply_example_hints(
        self, query: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        example_hints = self._collect_example_hints(query)
        if example_hints:
            context["example_table_hints"] = example_hints
            logger.info(
                "✓ Retrieved {}/{} historical SQL questions/tables",
                len(example_hints.get("example_questions", [])),
                len(example_hints.get("example_tables", [])),
            )
        return example_hints

    def _apply_vector_search(
        self,
        query: str,
        schema_documents: list[dict[str, Any]],
        all_schema_documents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if (
            not all_schema_documents
            or not self.agent.rag
            or not hasattr(self.agent.rag, "similarity_search")
        ):
            return schema_documents

        schema_lookup = self._build_schema_lookup(all_schema_documents)

        table_search_results = self.agent.rag.similarity_search(
            query=query,
            k=self.agent.vector_search_k,
            similarity_threshold=self.agent.similarity_threshold,
            doc_types=["schema"],
            chunk_types=None,
        )

        vector_tables = self._extract_tables_from_schema_results(
            table_search_results or []
        )
        if not vector_tables:
            return schema_documents

        vector_docs = self._collect_schema_documents(schema_lookup, vector_tables)
        schema_documents = self._merge_prioritized_documents(
            vector_docs, schema_documents
        )
        context["vector_search_tables"] = vector_tables
        logger.info("✓ Vector search prioritized {} schema tables", len(vector_tables))
        return schema_documents

    def _extract_tables_from_schema_results(
        self, results: list[dict[str, Any]]
    ) -> list[str]:
        ordered_tables: list[str] = []

        for result in results or []:
            table_name: str | None = None
            for key in ("doc_metadata", "chunk_metadata", "metadata"):
                metadata = result.get(key)
                if isinstance(metadata, dict):
                    table_name = self._resolve_table_name(metadata)
                    if table_name:
                        break

            if table_name and table_name not in ordered_tables:
                ordered_tables.append(table_name)

        return ordered_tables

    def _build_schema_lookup(
        self, schema_documents: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for doc in schema_documents or []:
            metadata = doc.get("metadata", {}) or {}
            content = doc.get("content", "")
            table_name = self._resolve_table_name(metadata, content)
            if table_name:
                key = table_name.lower()
                lookup.setdefault(key, doc)
        return lookup

    def _collect_schema_documents(
        self, lookup: dict[str, dict[str, Any]], table_names: list[str]
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for table_name in table_names or []:
            key = str(table_name).lower()
            doc = lookup.get(key)
            if doc and key not in seen_keys:
                seen_keys.add(key)
                collected.append(doc)
        return collected

    def _merge_prioritized_documents(
        self,
        prioritized_docs: list[dict[str, Any]],
        existing_docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for doc_list in (prioritized_docs or [], existing_docs or []):
            for doc in doc_list:
                key = self._schema_doc_key(doc)
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        return merged

    def _schema_doc_key(self, doc: dict[str, Any]) -> str:
        metadata = doc.get("metadata", {}) or {}
        content = doc.get("content", "")
        table_name = self._resolve_table_name(metadata, content)
        return table_name.lower() if table_name else f"id:{id(doc)}"

    def _filter_schema_by_term_matching(
        self, query: str, schema_documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tokens = tokenize_query(query)
        if not tokens:
            return [], {}

        table_scores: list[tuple[str, float]] = []
        table_docs: dict[str, dict[str, Any]] = {}

        for doc in schema_documents:
            metadata = doc.get("metadata", {}) or {}
            content = doc.get("content", "")
            table_name = metadata.get("table_name") or metadata.get("name")
            if not table_name and content:
                match = re.search(
                    r"CREATE\s+EXTERNAL\s+TABLE\s+(?:`?[\w]+`?\.)?`?([\w]+)`?",
                    content,
                    re.IGNORECASE,
                )
                if match:
                    table_name = match.group(1)

            if not table_name:
                continue

            candidates: list[str] = [table_name]
            candidates.extend(split_identifier(table_name))

            aliases = metadata.get("aliases") or metadata.get("alternate_names")
            if isinstance(aliases, list):
                candidates.extend(str(alias) for alias in aliases)
            elif isinstance(aliases, str):
                candidates.append(aliases)

            keywords = metadata.get("keywords")
            if isinstance(keywords, list):
                candidates.extend(str(keyword) for keyword in keywords)

            columns = extract_columns_from_doc(content, metadata)
            candidates.extend(columns)
            for column in columns:
                candidates.extend(split_identifier(column))

            score_val = score_match(
                tokens, candidates, self.agent.term_match_fuzzy_threshold
            )
            if score_val > 0:
                table_docs[table_name] = doc
                table_scores.append((table_name, score_val))

        if not table_scores:
            return [], {}

        table_scores.sort(key=lambda item: item[1], reverse=True)

        selected: list[tuple[str, float]] = []
        for table_name, score_val in table_scores:
            if len(selected) >= self.agent.term_match_top_k:
                break
            if score_val < self.agent.term_match_min_score and selected:
                continue
            selected.append((table_name, score_val))

        if not selected:
            selected = table_scores[: self.agent.term_match_top_k]

        matched_docs = [table_docs[name] for name, _ in selected]
        info = {
            "matched_tables": [name for name, _ in selected],
            "scores": {name: score_val for name, score_val in table_scores},
            "original_count": len(schema_documents),
            "filtered_count": len(matched_docs),
        }

        if info["filtered_count"] >= info["original_count"]:
            return [], info

        return matched_docs, info

    def _collect_example_hints(self, query: str) -> dict[str, Any]:
        results = self._search_sql_examples(query)
        if not results:
            return {}

        questions: list[str] = []
        tables = set()
        question_limit = min(self.agent.example_search_k, 20)

        for result in results[: self.agent.example_search_k]:
            question = extract_example_question(result)
            if (
                question
                and question not in questions
                and len(questions) < question_limit
            ):
                questions.append(question)

            sql_text = extract_example_sql(result)
            for table in extract_tables_from_sql(sql_text):
                tables.add(table)

        if not questions and not tables:
            return {}
        return {
            "example_questions": questions,
            "example_tables": sorted(tables),
        }

    def _search_sql_examples(self, query: str) -> list[dict[str, Any]]:
        if not self.agent.rag or not hasattr(self.agent.rag, "similarity_search"):
            return []

        return (
            self.agent.rag.similarity_search(
                query=query,
                k=self.agent.example_search_k,
                similarity_threshold=self.agent.example_similarity_threshold,
                doc_types=["sql_example"],
                chunk_types=self.agent.example_chunk_types,
            )
            or []
        )

    def _prepare_term_match_query(self, query: str, normalized_hint: Any | None) -> str:
        """Combine normalized hints with original query for term matching."""
        data = None
        if hasattr(normalized_hint, "model_dump"):
            try:
                data = normalized_hint.model_dump(exclude_none=True)
            except Exception:
                data = None
        elif isinstance(normalized_hint, dict):
            data = normalized_hint

        parts: list[str] = []
        terms = self._extract_normalized_terms(data)
        if terms:
            parts.append(" ".join(terms))

        parts.append(query)
        combined = " ".join(part for part in parts if part)
        return combined.strip()

    def _extract_normalized_terms(self, data: dict[str, Any] | None) -> list[str]:
        if not isinstance(data, dict):
            return []
        tokens: list[str] = []
        tokens.extend(tokenize_query(data.get("main_clause", "")))
        for detail in data.get("details_for_filterings") or []:
            if (
                isinstance(detail, str)
                and detail.strip()
                and detail.strip().lower() not in GENERIC_DETAIL_PHRASES
            ):
                tokens.extend(tokenize_query(detail))
        tokens.extend(tokenize_query(data.get("required_visuals", "")))
        for table in data.get("tables") or []:
            if isinstance(table, str):
                tokens.extend(tokenize_query(table))
        seen: list[str] = []
        for token in tokens:
            if token and token not in seen:
                seen.append(token)
        return seen

    def _get_all_schema_documents(self) -> list[dict[str, Any]]:
        """Retrieve all schema documents with metadata from RAG system."""
        try:
            cursor_callable = getattr(self.agent.rag, "cursor", None)
            if not callable(cursor_callable):
                return []

            row_factory = dict_row if dict_row else None
            with self.agent.rag.cursor(row_factory=row_factory) as cur:
                cur.execute(
                    """
                    SELECT full_content, metadata
                    FROM rag_documents
                    WHERE doc_type = 'schema'
                    ORDER BY created_at DESC
                    """
                )
                results: list[Any] = cur.fetchall()

            schema_docs: list[dict[str, Any]] = []
            for row in results:
                if isinstance(row, Mapping):
                    content = row.get("full_content", "")
                    metadata = row.get("metadata", {})
                else:
                    content = row[0] if len(row) > 0 else ""
                    metadata = row[1] if len(row) > 1 else {}

                if content:
                    schema_docs.append({"content": content, "metadata": metadata})

            return schema_docs

        except Exception as e:
            logger.error(f"Failed to get schema documents: {e}")
            return []

    def _resolve_table_name(
        self, metadata: dict[str, Any] | None, content: str = ""
    ) -> str | None:
        table_name: str | None = None
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        if isinstance(metadata, dict):
            raw_name = metadata.get("table_name") or metadata.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                table_name = raw_name.strip()

        if not table_name and content:
            match = re.search(
                r"CREATE\s+EXTERNAL\s+TABLE\s+(?:`?[\w]+`?\.)?`?([\w]+)`?",
                content,
                re.IGNORECASE,
            )
            if match:
                table_name = match.group(1)

        return table_name

    def format_example_hints(self, context: dict[str, Any] | None) -> str:
        """Format SQL example hints for inclusion in schema context."""
        if not context:
            return ""

        hints = context.get("example_table_hints") or {}
        tables = hints.get("example_tables") or []
        questions = hints.get("example_questions") or []
        if not tables and not questions:
            return ""

        question_lines = "\n".join(f"- {q}" for q in questions[:5])
        table_lines = "\n".join(f"- {t}" for t in tables[:10])

        parts = ["## HISTORICAL SQL EXAMPLES"]
        if question_lines:
            parts.append("Similar Questions:")
            parts.append(question_lines)
        if table_lines:
            parts.append("Tables Recommended:")
            parts.append(table_lines)
        return "\n".join(parts)

    def format_normalized_table_hints(self, context: dict[str, Any] | None) -> str:
        """Format normalized table hints as domain keywords."""
        if not context:
            return ""

        normalized = context.get("normalized_query")
        if not normalized:
            return ""

        tables = None
        if hasattr(normalized, "tables"):
            tables = normalized.tables
        elif isinstance(normalized, dict):
            tables = normalized.get("tables")

        if not tables:
            return ""

        hints = [
            str(table).strip()
            for table in tables
            if isinstance(table, str) and table.strip()
        ]
        if not hints:
            return ""

        table_hints = ", ".join(hints[:10])
        return (
            "## USER MENTIONED DATASETS\n"
            f"User referenced: {table_hints}\n"
            "(Treat as domain keywords, not literal table names)"
        )

    def format_term_match_hints(
        self, matched_tables: list[str], score_map: dict[str, Any]
    ) -> str:
        """Format keyword matching hints for inclusion in schema context."""
        if not matched_tables:
            return ""

        lines = []
        for table in matched_tables[:10]:
            score = score_map.get(table)
            if isinstance(score, int | float):
                lines.append(f"- {table} (score: {score:.2f})")
            else:
                lines.append(f"- {table}")

        parts = ["## TERM MATCH CANDIDATES", "Keyword matched tables:"]
        parts.append("\n".join(lines))
        return "\n".join(parts)

    def format_vector_search_hints(self, context: dict[str, Any] | None) -> str:
        """Format vector search candidates for inclusion in schema context."""
        if not context:
            return ""

        tables = context.get("vector_search_tables") or []
        if not tables:
            return ""

        lines = []
        for table in tables[:10]:
            lines.append(f"- {table}")

        if not lines:
            return ""

        parts = ["## VECTOR SEARCH CANDIDATES", "Embedding matched tables:"]
        parts.append("\n".join(lines))
        return "\n".join(parts)
