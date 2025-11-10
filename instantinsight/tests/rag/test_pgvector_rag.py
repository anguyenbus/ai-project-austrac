"""Tests for PgvectorRAG utility methods."""

from unittest.mock import Mock

from src.rag.pgvector_rag import PgvectorRAG


def test_standard_similarity_search_rolls_back_on_error():
    """Ensure failed legacy searches roll back the transaction."""
    rag = PgvectorRAG("postgresql://fake")

    cur = Mock()
    cur.execute.side_effect = Exception("invalid input value for enum")
    cur.connection = Mock()

    result = rag._standard_similarity_search(
        cur=cur,
        query_embedding=[0.1],
        k=5,
        similarity_threshold=0.2,
        doc_types=["sql_example"],
        chunk_types=["example_overview"],
        table_filter=None,
        namespace="semantic",
    )

    assert result == []
    cur.connection.rollback.assert_called_once()
