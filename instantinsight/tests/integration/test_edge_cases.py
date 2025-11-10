"""
Edge case tests for multi-turn conversations.

Tests various edge cases and error conditions to ensure
robustness of the multi-turn implementation.
"""

import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lambda"))

from lambda_handler import lambda_handler


class TestMultiTurnEdgeCases:
    """Test edge cases for multi-turn conversations."""

    def test_empty_prior_turns_list(self):
        """Test with empty prior_turns list."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with empty prior_turns
        event = {"query": "Show me data", "prior_turns": []}

        # Mock the processor
        mock_processor = Mock()

        async def mock_async(**kwargs):
            # Verify prior_turns is empty list
            assert "prior_turns" in kwargs
            assert kwargs["prior_turns"] == []

            return {
                "success": True,
                "sql": "SELECT * FROM data",
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": None,
            }

        mock_processor.process_query_async = mock_async

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

            # Should process normally
            assert response["statusCode"] == 200
            body = response["body"]
            assert '"success": true' in body

    def test_maximum_prior_turns_truncation(self):
        """Test with more than 6 prior turns (should truncate)."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Create 10 prior turns
        prior_turns = [
            {"content": f"Query {i}", "sql": f"SELECT {i}"} for i in range(10)
        ]

        # Event with 10 prior turns
        event = {"query": "Current query", "prior_turns": prior_turns}

        # Mock the processor
        mock_processor = Mock()

        async def mock_async(**kwargs):
            # Verify only last 6 turns are included
            assert "prior_turns" in kwargs
            assert len(kwargs["prior_turns"]) == 6
            # Should keep the last 6 turns (indices 4-9)
            assert kwargs["prior_turns"][0]["content"] == "Query 4"
            assert kwargs["prior_turns"][-1]["content"] == "Query 9"

            return {
                "success": True,
                "sql": "SELECT * FROM test",
                "row_count": 5,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": None,
            }

        mock_processor.process_query_async = mock_async

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

            # Should process normally with truncated context
            assert response["statusCode"] == 200

    def test_prior_turns_with_missing_sql(self):
        """Test prior_turns with missing SQL field."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with prior_turns missing SQL
        event = {
            "query": "Show me more",
            "prior_turns": [
                {"content": "Show products"}  # Missing SQL
            ],
        }

        # Mock the processor
        mock_processor = Mock()

        async def mock_async(**kwargs):
            # Should still include the turn even without SQL
            assert "prior_turns" in kwargs
            assert len(kwargs["prior_turns"]) == 1
            assert kwargs["prior_turns"][0]["content"] == "Show products"
            assert "sql" not in kwargs["prior_turns"][0]

            return {
                "success": True,
                "sql": "SELECT * FROM products",
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": None,
            }

        mock_processor.process_query_async = mock_async

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

            # Should process normally
            assert response["statusCode"] == 200

    def test_prior_turns_with_missing_visualization(self):
        """Test prior_turns with missing visualization field."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with prior_turns missing visualization
        event = {
            "query": "Change to line chart",
            "prior_turns": [
                {
                    "content": "Show sales",
                    "sql": "SELECT * FROM sales",
                }  # Missing visualization
            ],
        }

        # Mock the processor
        mock_processor = Mock()

        async def mock_async(**kwargs):
            # Should still include the turn even without visualization
            assert "prior_turns" in kwargs
            assert len(kwargs["prior_turns"]) == 1
            assert kwargs["prior_turns"][0]["content"] == "Show sales"
            assert "visualization" not in kwargs["prior_turns"][0]

            return {
                "success": True,
                "sql": "SELECT * FROM sales",
                "row_count": 20,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": {
                    "chart": {"type": "line", "config": {"x": "date", "y": "amount"}}
                },
            }

        mock_processor.process_query_async = mock_async

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

            # Should process normally
            assert response["statusCode"] == 200

    def test_prior_turns_with_empty_content(self):
        """Test prior_turns with empty content field."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with prior_turns having empty content
        event = {
            "query": "Show me data",
            "prior_turns": [
                {"content": "", "sql": "SELECT * FROM test"}  # Empty content
            ],
        }

        # Should return 400 error for invalid prior_turns
        response = lambda_handler(event, mock_context)

        assert response["statusCode"] == 400
        body = response["body"]
        assert '"success": false' in body
        assert '"error": "Invalid prior_turns format"' in body

    def test_prior_turns_with_non_string_content(self):
        """Test prior_turns with non-string content field."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with prior_turns having non-string content
        event = {
            "query": "Show me data",
            "prior_turns": [
                {"content": 123, "sql": "SELECT * FROM test"}  # Non-string content
            ],
        }

        # Should return 400 error for invalid prior_turns
        response = lambda_handler(event, mock_context)

        assert response["statusCode"] == 400
        body = response["body"]
        assert '"success": false' in body
        assert '"error": "Invalid prior_turns format"' in body

    def test_backward_compatibility_without_prior_turns(self):
        """Test backward compatibility when prior_turns is not provided."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event without prior_turns (old API)
        event = {"query": "Show me data"}

        # Mock the processor
        mock_processor = Mock()

        async def mock_async(**kwargs):
            # Should pass empty list for prior_turns
            assert "prior_turns" in kwargs
            assert kwargs["prior_turns"] == []

            return {
                "success": True,
                "sql": "SELECT * FROM data",
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": None,
            }

        mock_processor.process_query_async = mock_async

        with patch("lambda_handler._init_processor", return_value=mock_processor):
            response = lambda_handler(event, mock_context)

            # Should process normally
            assert response["statusCode"] == 200

    def test_malformed_prior_turns_structure(self):
        """Test with malformed prior_turns structure."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with malformed prior_turns (not a list)
        event = {"query": "Show me data", "prior_turns": "not a list"}

        # Should return 400 error for invalid prior_turns
        response = lambda_handler(event, mock_context)

        assert response["statusCode"] == 400
        body = response["body"]
        assert '"success": false' in body
        assert '"error": "Invalid prior_turns format"' in body

    def test_mixed_valid_invalid_prior_turns(self):
        """Test with mix of valid and invalid prior turns."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with mix of valid and invalid turns
        event = {
            "query": "Show me data",
            "prior_turns": [
                {"content": "Valid query", "sql": "SELECT * FROM test"},
                "invalid turn",  # Not a dict
                {"content": "Another valid query", "sql": "SELECT * FROM test2"},
            ],
        }

        # Should return 400 error for invalid prior_turns
        response = lambda_handler(event, mock_context)

        assert response["statusCode"] == 400
        body = response["body"]
        assert '"success": false' in body
        assert '"error": "Invalid prior_turns format"' in body
