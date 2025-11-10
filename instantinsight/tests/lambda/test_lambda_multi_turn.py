"""Tests for Lambda handler with S3-backed session support."""

import json
import os
import sys
from unittest.mock import Mock, patch

import boto3
import pytest
from moto import mock_s3

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lambda"))

from lambda_handler import lambda_handler


class TestLambdaMultiTurnWithS3:
    """Test Lambda handler with S3-backed sessions."""

    @pytest.fixture
    def s3_setup(self):
        """Set up mocked S3 bucket for testing."""
        with mock_s3():
            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_client.create_bucket(Bucket="test-session-bucket")
            yield s3_client

    @pytest.fixture
    def mock_context(self):
        """Create mock Lambda context."""
        context = Mock()
        context.aws_request_id = "test-request-id"
        context.get_remaining_time_in_millis = Mock(return_value=30000)
        return context

    @pytest.fixture
    def mock_processor(self):
        """Create mock QueryProcessor."""
        processor = Mock()

        async def mock_async(**kwargs):
            return {
                "query": kwargs.get("query", "Test query"),
                "success": True,
                "sql": "SELECT * FROM test",
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": None,
            }

        processor.process_query_async = mock_async
        return processor

    def test_lambda_rejects_legacy_prior_turns_parameter(
        self, mock_context, mock_processor
    ):
        """Test Lambda handler rejects legacy prior_turns parameter."""
        event = {
            "query": "Show me more",
            "prior_turns": [
                {"content": "Show top 10 products", "sql": "SELECT * FROM products"}
            ],
        }

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

        # Should return 400 error
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "no longer supported" in body["error"]

    def test_lambda_generates_session_id_if_not_provided(
        self, s3_setup, mock_context, mock_processor
    ):
        """Test Lambda handler generates session_id if not provided."""
        event = {"query": "Show top 10 products"}

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=None),
        ):
            response = lambda_handler(event, mock_context)

        # Should succeed and include session_id
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["success"] is True
        assert "session_id" in body
        assert "turn_count" in body
        assert body["turn_count"] == 1

    def test_lambda_multi_turn_conversation_with_s3(
        self, s3_setup, mock_context, mock_processor
    ):
        """Test multi-turn conversation with S3 session persistence."""
        from src.session import SessionStore

        # Initialize SessionStore with test bucket
        session_store = SessionStore(bucket="test-session-bucket", prefix="test/")

        # Turn 1: Initial query
        event1 = {"query": "Show top 10 products"}

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=session_store),
        ):
            response1 = lambda_handler(event1, mock_context)

        assert response1["statusCode"] == 200
        body1 = json.loads(response1["body"])
        session_id = body1["session_id"]
        assert body1["turn_count"] == 1

        # Turn 2: Follow-up query with session_id
        event2 = {"query": "Show bottom 5", "session_id": session_id}

        # Update mock to return different query
        async def mock_async_turn2(**kwargs):
            return {
                "query": "Show bottom 5",
                "success": True,
                "sql": "SELECT * FROM test ORDER BY value ASC LIMIT 5",
                "row_count": 5,
                "cache_hit": False,
                "total_duration_ms": 600,
                "visualization": None,
            }

        mock_processor.process_query_async = mock_async_turn2

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=session_store),
        ):
            response2 = lambda_handler(event2, mock_context)

        assert response2["statusCode"] == 200
        body2 = json.loads(response2["body"])
        assert body2["session_id"] == session_id
        assert body2["turn_count"] == 2

        # Verify session stored in S3
        session = session_store.load(session_id)
        assert len(session["turns"]) == 2
        assert session["turns"][0]["content"] == "Show top 10 products"
        assert session["turns"][1]["content"] == "Show bottom 5"

    def test_lambda_session_truncates_to_6_turns(
        self, s3_setup, mock_context, mock_processor
    ):
        """Test session automatically truncates to 6 turns."""
        from src.session import SessionStore

        session_store = SessionStore(bucket="test-session-bucket", prefix="test/")
        session_id = None

        # Add 8 turns
        for i in range(8):

            async def mock_async_turn(idx=i, **kwargs):
                return {
                    "query": f"Query {idx}",
                    "success": True,
                    "sql": f"SELECT {idx}",
                    "row_count": 10,
                    "cache_hit": False,
                    "total_duration_ms": 500,
                    "visualization": None,
                }

            mock_processor.process_query_async = mock_async_turn

            event = {"query": f"Query {i}"}
            if session_id:
                event["session_id"] = session_id

            with (
                patch("lambda_handler._init_processor", return_value=mock_processor),
                patch("lambda_handler._init_session_store", return_value=session_store),
            ):
                response = lambda_handler(event, mock_context)

            assert response["statusCode"] == 200
            body = json.loads(response["body"])

            if session_id is None:
                session_id = body["session_id"]

        # Verify only last 6 turns retained
        session = session_store.load(session_id)
        assert len(session["turns"]) == 6
        assert session["turns"][0]["content"] == "Query 2"
        assert session["turns"][5]["content"] == "Query 7"

    def test_lambda_session_save_failure_returns_500(
        self, s3_setup, mock_context, mock_processor
    ):
        """Test Lambda returns 500 if session save fails."""
        from src.session import SessionStore

        # Create session store with non-existent bucket (will fail on save)
        session_store = SessionStore(bucket="non-existent-bucket", prefix="test/")

        event = {"query": "Test query"}

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=session_store),
        ):
            response = lambda_handler(event, mock_context)

        # Should return 500 error
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "session" in body["error"].lower()

    def test_lambda_with_disabled_session_store(self, mock_context, mock_processor):
        """Test Lambda works when session store is disabled."""
        event = {"query": "Test query"}

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=None),
        ):  # Session store disabled
            response = lambda_handler(event, mock_context)

        # Should succeed without session tracking
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["success"] is True
        # Should still include session_id and turn_count
        assert "session_id" in body
        assert body["turn_count"] == 1

    def test_lambda_session_load_failure_returns_500(
        self, s3_setup, mock_context, mock_processor
    ):
        """Test Lambda returns 500 if session load fails."""
        from src.session import SessionStore

        session_store = SessionStore(bucket="test-session-bucket", prefix="test/")

        # Create event with session_id pointing to corrupted session
        session_id = "corrupted-session"
        key = session_store._build_key(session_id)

        # Put invalid JSON in S3
        s3_setup.put_object(
            Bucket="test-session-bucket", Key=key, Body=b"not valid json {{{"
        )

        event = {"query": "Test query", "session_id": session_id}

        with (
            patch("lambda_handler._init_processor", return_value=mock_processor),
            patch("lambda_handler._init_session_store", return_value=session_store),
        ):
            response = lambda_handler(event, mock_context)

        # Should return 500 error
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["success"] is False
        assert "session" in body["error"].lower()
