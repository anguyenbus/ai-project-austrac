"""
Utility functions for schema and table analysis.

Provides SQL parsing, text tokenisation, and scoring functions for table selection.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any

_SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join|into|update|delete\s+from)\s+[`\"\[]?([a-zA-Z_][\w$.]*)[`\"\]]?",
    re.IGNORECASE,
)
_EXAMPLE_SQL_KEYS = ("sql", "query", "sql_text", "statement")
_COLUMN_PATTERN = re.compile(
    r"^\s*`?([a-zA-Z_][\w]*)`?\s+[a-zA-Z]",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "what",
    "which",
    "show",
    "list",
    "find",
    "get",
    "give",
    "current",
    "latest",
    "all",
    "for",
    "with",
    "the",
    "a",
    "an",
    "by",
    "from",
    "in",
    "of",
    "to",
    "on",
    "and",
    "or",
    "compare",
    "between",
}


def extract_example_question(result: dict[str, Any]) -> str | None:
    """Best-effort extraction of the question associated with a SQL example."""
    metadata = result.get("chunk_metadata") or {}
    question = metadata.get("question")
    if question:
        return question.strip()

    doc_meta = result.get("doc_metadata") or {}
    question = doc_meta.get("question") or doc_meta.get("title")
    if question:
        return question.strip()

    content = (result.get("content") or "").strip()
    return content.splitlines()[0].strip() if content else None


def extract_example_sql(result: dict[str, Any]) -> str:
    """Return the SQL text from a RAG example search result."""
    metadata = result.get("chunk_metadata") or {}
    for key in _EXAMPLE_SQL_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value

    doc_meta = result.get("doc_metadata") or {}
    for key in _EXAMPLE_SQL_KEYS:
        value = doc_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value

    content = result.get("content")
    return content if isinstance(content, str) else ""


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table references from SQL or structured payloads."""
    if sql is None:
        return []

    tables: list[str] = []
    seen = set()

    def add_table(name: str) -> None:
        if not isinstance(name, str):
            return
        candidate = name.strip('`"[] ')
        if not candidate:
            return
        base = candidate.split(".")[-1]
        lowered = base.lower()
        if lowered in {"select", "where", "from"}:
            return
        if base not in seen:
            seen.add(base)
            tables.append(base)

    raw_sql: str = ""

    if isinstance(sql, dict):
        table_list = sql.get("tables")
        if isinstance(table_list, list):
            for entry in table_list:
                add_table(entry)
        raw_sql = sql.get("sql") or sql.get("query") or ""
    else:
        raw_sql = str(sql or "").strip()
        trimmed = raw_sql
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                payload = json.loads(trimmed)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                table_list = payload.get("tables")
                if isinstance(table_list, list):
                    for entry in table_list:
                        add_table(entry)
                raw_sql = payload.get("sql") or payload.get("query") or ""
                if not raw_sql:
                    raw_sql = trimmed
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        table_list = item.get("tables")
                        if isinstance(table_list, list):
                            for entry in table_list:
                                add_table(entry)
                        if not raw_sql:
                            raw_sql = item.get("sql") or item.get("query") or ""
                if not raw_sql:
                    raw_sql = trimmed
        else:
            raw_sql = trimmed

    raw_sql = str(raw_sql or "").strip()
    if not raw_sql:
        return tables

    cleaned = re.sub(
        r"--.*?$|/\*.*?\*/",
        " ",
        raw_sql,
        flags=re.MULTILINE | re.DOTALL,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)

    for raw in _SQL_TABLE_PATTERN.findall(cleaned):
        add_table(raw)

    return tables


def tokenize_query(query: str) -> list[str]:
    """Tokenize the query into normalised terms for matching."""
    if not query:
        return []

    tokens = re.split(r"[^a-z0-9]+", query.lower())
    normalised = []
    for token in tokens:
        if not token or token in _STOP_WORDS:
            continue
        normalised.append(simple_stem(token))
    return normalised


def simple_stem(token: str) -> str:
    """Apply lightweight stemming without external dependencies."""
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def split_identifier(identifier: str) -> list[str]:
    """Split identifiers like customer_orders into component tokens."""
    if not identifier:
        return []
    parts = identifier.replace(".", "_").split("_")
    return [part.lower() for part in parts if part]


def score_match(
    tokens: list[str], candidates: list[str], fuzzy_threshold: float
) -> float:
    """Compute aggregate match score between query tokens and candidate identifiers."""
    if not tokens or not candidates:
        return 0.0

    unique_candidates = {candidate.lower() for candidate in candidates if candidate}
    if not unique_candidates:
        return 0.0

    score = 0.0
    for token in tokens:
        token_score = 0.0
        for candidate in unique_candidates:
            if token == candidate:
                token_score = max(token_score, 3.0)
                continue
            if token in candidate:
                token_score = max(token_score, 2.0)
                continue
            if simple_stem(token) == simple_stem(candidate):
                token_score = max(token_score, 2.0)
                continue
            ratio = SequenceMatcher(None, token, candidate).ratio()
            if ratio >= fuzzy_threshold:
                token_score = max(token_score, ratio)
        score += token_score
    return score


def extract_columns_from_doc(content: str, metadata: dict[str, Any]) -> list[str]:
    """Pull column names from metadata or CREATE TABLE definition."""
    columns: list[str] = []

    meta_columns = metadata.get("columns") or metadata.get("column_names")
    if isinstance(meta_columns, list):
        for column in meta_columns:
            if isinstance(column, str):
                columns.append(column)
            elif isinstance(column, dict):
                name = column.get("name")
                if name:
                    columns.append(name)

    if not columns and content:
        for line in content.splitlines():
            match = _COLUMN_PATTERN.match(line)
            if match:
                column_name = match.group(1)
                if column_name.upper() not in {"PRIMARY", "KEY", "CONSTRAINT"}:
                    columns.append(column_name)

    return columns
