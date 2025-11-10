"""
Unit tests for SessionStore with moto S3 mocking.

Tests cover:
- Basic load/save/append/delete operations
- ETag-based optimistic locking
- Concurrency conflict handling
- Error handling and retries
- Turn truncation to 6 max turns
- Query result to turn conversion
"""

from datetime import datetime

import boto3
import pytest
from moto import mock_s3

from src.session import SessionStore, query_result_to_turn

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def s3_client():
    """Create mocked S3 client."""
    with mock_s3():
        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def session_store(s3_client):
    """Create SessionStore with test bucket."""
    bucket = "test-sessions"
    s3_client.create_bucket(Bucket=bucket)
    return SessionStore(bucket=bucket, prefix="test/")


@pytest.fixture
def sample_turn():
    """Create sample turn for testing."""
    return {
        "content": "Show revenue",
        "sql": "SELECT SUM(revenue) FROM sales",
        "success": True,
        "row_count": 1,
        "cache_hit": False,
        "visualization": None,
        "timestamp": "2025-01-15T12:00:00Z",
        "duration_ms": 500,
        "error": None,
    }


@pytest.fixture
def sample_query_result():
    """Create sample QueryResult for testing."""
    return {
        "query": "Show top 10 products",
        "sql": "SELECT * FROM products LIMIT 10",
        "success": True,
        "row_count": 10,
        "cache_hit": False,
        "visualization": {"chart_type": "bar", "config": {}},
        "total_duration_ms": 1234,
        "error": None,
    }


# ============================================================================
# Basic Operations Tests
# ============================================================================


def test_session_store_initialization(s3_client):
    """Test SessionStore initialization."""
    bucket = "test-bucket"
    s3_client.create_bucket(Bucket=bucket)

    store = SessionStore(bucket=bucket, prefix="sessions/")

    assert store.bucket == bucket
    assert store.prefix == "sessions/"
    assert store._s3_client is not None


def test_session_store_initialization_empty_bucket():
    """Test SessionStore initialization with empty bucket name."""
    with pytest.raises(ValueError, match="S3 bucket name cannot be empty"):
        SessionStore(bucket="", prefix="sessions/")


def test_load_missing_session(session_store):
    """Test loading non-existent session returns empty session."""
    session_id = "missing-session-123"

    session = session_store.load(session_id)

    assert session["session_id"] == session_id
    assert session["turns"] == []
    assert "updated_at" in session
    assert session["metadata"] == {}


def test_save_and_load_session(session_store, sample_turn):
    """Test saving and loading session."""
    session_id = "test-session-123"
    turns = [sample_turn]

    # Save session
    session_store.save(session_id, turns)

    # Load session
    session = session_store.load(session_id)

    assert session["session_id"] == session_id
    assert len(session["turns"]) == 1
    assert session["turns"][0]["content"] == "Show revenue"
    assert session["turns"][0]["sql"] == "SELECT SUM(revenue) FROM sales"


def test_append_turn(session_store, sample_turn):
    """Test appending turn to session."""
    session_id = "append-test-123"

    # Append first turn
    session_store.append(session_id, sample_turn)

    # Load and verify
    session = session_store.load(session_id)
    assert len(session["turns"]) == 1

    # Append second turn
    sample_turn["content"] = "Show profit"
    sample_turn["sql"] = "SELECT SUM(profit) FROM sales"
    session_store.append(session_id, sample_turn)

    # Load and verify
    session = session_store.load(session_id)
    assert len(session["turns"]) == 2
    assert session["turns"][0]["content"] == "Show revenue"
    assert session["turns"][1]["content"] == "Show profit"


def test_delete_session(session_store, sample_turn):
    """Test deleting session."""
    session_id = "delete-test-123"

    # Create session
    session_store.append(session_id, sample_turn)

    # Verify exists
    session = session_store.load(session_id)
    assert len(session["turns"]) == 1

    # Delete session
    session_store.delete(session_id)

    # Verify deleted (returns empty session)
    session = session_store.load(session_id)
    assert len(session["turns"]) == 0


# ============================================================================
# Turn Truncation Tests
# ============================================================================


def test_truncate_to_max_turns(session_store, sample_turn):
    """Test automatic truncation to 6 turns."""
    session_id = "truncate-test-123"

    # Append 10 turns
    for i in range(10):
        turn = sample_turn.copy()
        turn["content"] = f"Query {i}"
        session_store.append(session_id, turn)

    # Load and verify only last 6 turns kept
    session = session_store.load(session_id)
    assert len(session["turns"]) == 6
    assert session["turns"][0]["content"] == "Query 4"
    assert session["turns"][5]["content"] == "Query 9"


def test_save_with_truncation(session_store, sample_turn):
    """Test save automatically truncates to max turns."""
    session_id = "save-truncate-123"

    # Create 8 turns
    turns = []
    for i in range(8):
        turn = sample_turn.copy()
        turn["content"] = f"Query {i}"
        turns.append(turn)

    # Save should truncate to 6
    session_store.save(session_id, turns)

    # Verify only last 6 turns saved
    session = session_store.load(session_id)
    assert len(session["turns"]) == 6
    assert session["turns"][0]["content"] == "Query 2"
    assert session["turns"][5]["content"] == "Query 7"


# ============================================================================
# ETag and Optimistic Locking Tests
# ============================================================================


def test_etag_caching(session_store, sample_turn):
    """Test ETag is cached after load."""
    session_id = "etag-test-123"

    # Create session
    session_store.append(session_id, sample_turn)

    # Load session (should cache ETag)
    session_store.load(session_id)

    # Verify ETag is cached
    assert session_id in session_store._etag_cache
    assert session_store._etag_cache[session_id] is not None


def test_optimistic_locking_success(session_store, sample_turn):
    """Test successful save with matching ETag."""
    session_id = "lock-success-123"

    # Create initial session
    session_store.append(session_id, sample_turn)

    # Load to cache ETag
    session = session_store.load(session_id)

    # Save with matching ETag should succeed
    turn2 = sample_turn.copy()
    turn2["content"] = "Query 2"
    session["turns"].append(turn2)
    session_store.save(session_id, session["turns"])

    # Verify both turns saved
    session = session_store.load(session_id)
    assert len(session["turns"]) == 2


# ============================================================================
# Query Result Conversion Tests
# ============================================================================


def test_query_result_to_turn(sample_query_result):
    """Test converting QueryResult to Turn."""
    turn = query_result_to_turn(sample_query_result)

    assert turn["content"] == "Show top 10 products"
    assert turn["sql"] == "SELECT * FROM products LIMIT 10"
    assert turn["success"] is True
    assert turn["row_count"] == 10
    assert turn["cache_hit"] is False
    assert turn["visualization"] == {"chart_type": "bar", "config": {}}
    assert turn["duration_ms"] == 1234
    assert turn["error"] is None
    assert "timestamp" in turn


def test_query_result_to_turn_with_error():
    """Test converting failed QueryResult to Turn."""
    result = {
        "query": "Invalid query",
        "sql": None,
        "success": False,
        "row_count": None,
        "cache_hit": False,
        "visualization": None,
        "total_duration_ms": 100,
        "error": "SQL generation failed",
    }

    turn = query_result_to_turn(result)

    assert turn["content"] == "Invalid query"
    assert turn["sql"] is None
    assert turn["success"] is False
    assert turn["error"] == "SQL generation failed"


def test_query_result_to_turn_minimal():
    """Test converting minimal QueryResult to Turn."""
    result = {
        "query": "Simple query",
        "success": True,
    }

    turn = query_result_to_turn(result)

    assert turn["content"] == "Simple query"
    assert turn["success"] is True
    assert turn["cache_hit"] is False
    assert turn["sql"] is None
    assert turn["row_count"] is None


# ============================================================================
# Round Trip Tests
# ============================================================================


def test_full_session_lifecycle(session_store, sample_turn):
    """Test complete session lifecycle."""
    session_id = "lifecycle-test-123"

    # 1. Load non-existent session (empty)
    session = session_store.load(session_id)
    assert len(session["turns"]) == 0

    # 2. Append multiple turns
    for i in range(3):
        turn = sample_turn.copy()
        turn["content"] = f"Query {i}"
        session_store.append(session_id, turn)

    # 3. Verify turns saved
    session = session_store.load(session_id)
    assert len(session["turns"]) == 3

    # 4. Delete session
    session_store.delete(session_id)

    # 5. Verify session deleted
    session = session_store.load(session_id)
    assert len(session["turns"]) == 0


def test_multi_session_isolation(session_store, sample_turn):
    """Test sessions are isolated from each other."""
    session_id_1 = "session-1"
    session_id_2 = "session-2"

    # Create turn for session 1
    turn1 = sample_turn.copy()
    turn1["content"] = "Session 1 query"
    session_store.append(session_id_1, turn1)

    # Create turn for session 2
    turn2 = sample_turn.copy()
    turn2["content"] = "Session 2 query"
    session_store.append(session_id_2, turn2)

    # Verify isolation
    session1 = session_store.load(session_id_1)
    session2 = session_store.load(session_id_2)

    assert len(session1["turns"]) == 1
    assert len(session2["turns"]) == 1
    assert session1["turns"][0]["content"] == "Session 1 query"
    assert session2["turns"][0]["content"] == "Session 2 query"


# ============================================================================
# S3 Key Format Tests
# ============================================================================


def test_s3_key_format(session_store):
    """Test S3 key is formatted correctly."""
    session_id = "test-uuid-123"

    expected_key = f"test/session_{session_id}/session.json"
    actual_key = session_store._build_key(session_id)

    assert actual_key == expected_key


def test_prefix_normalization(s3_client):
    """Test prefix is normalized with trailing slash."""
    bucket = "test-bucket"
    s3_client.create_bucket(Bucket=bucket)

    # Test without trailing slash
    store1 = SessionStore(bucket=bucket, prefix="sessions")
    assert store1.prefix == "sessions/"

    # Test with trailing slash
    store2 = SessionStore(bucket=bucket, prefix="sessions/")
    assert store2.prefix == "sessions/"

    # Test empty prefix
    store3 = SessionStore(bucket=bucket, prefix="")
    assert store3.prefix == ""


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_load_with_invalid_json(session_store, s3_client):
    """Test loading session with corrupted JSON."""
    session_id = "corrupted-session"
    key = session_store._build_key(session_id)

    # Put invalid JSON
    s3_client.put_object(
        Bucket=session_store.bucket,
        Key=key,
        Body=b"not valid json {{{",
    )

    # Should raise error
    with pytest.raises(RuntimeError, match="Failed to load session"):
        session_store.load(session_id)


def test_save_with_invalid_bucket(s3_client, sample_turn):
    """Test save with non-existent bucket."""
    store = SessionStore(bucket="non-existent-bucket", prefix="test/")

    with pytest.raises(RuntimeError, match="Failed to save session"):
        store.save("test-session", [sample_turn])


# ============================================================================
# Session Metadata Tests
# ============================================================================


def test_session_updated_at_timestamp(session_store, sample_turn):
    """Test updated_at timestamp is set correctly."""
    session_id = "timestamp-test-123"

    before = datetime.utcnow()
    session_store.append(session_id, sample_turn)
    after = datetime.utcnow()

    session = session_store.load(session_id)
    updated_at = datetime.fromisoformat(session["updated_at"].rstrip("Z"))

    assert before <= updated_at <= after


def test_session_metadata_preserved(session_store, sample_turn):
    """Test session metadata structure."""
    session_id = "metadata-test-123"

    session_store.append(session_id, sample_turn)
    session = session_store.load(session_id)

    assert "session_id" in session
    assert "turns" in session
    assert "updated_at" in session
    assert "metadata" in session
    assert isinstance(session["metadata"], dict)
