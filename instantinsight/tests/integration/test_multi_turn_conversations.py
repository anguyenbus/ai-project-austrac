"""
Integration tests for multi-turn conversations.

Tests the full flow from Lambda handler through the pipeline
with conversation context.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lambda"))

from lambda_handler import lambda_handler


class TestMultiTurnConversations:
    """Test multi-turn conversation integration."""

    @pytest.mark.integration
    def test_simple_two_turn_conversation(self):
        """Test simple two-turn conversation with context."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Turn 1 - Establish context
        event1 = {
            "query": "Show me the top 5 products by revenue",
            "export_to_s3": False,
            "generate_visualization": True,
        }

        # Mock the processor for Turn 1
        mock_processor1 = Mock()

        async def mock_async1(**kwargs):
            return {
                "success": True,
                "sql": "SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue DESC LIMIT 5",
                "row_count": 5,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": {
                    "chart": {
                        "type": "bar",
                        "config": {"x": "product_name", "y": "revenue"},
                    }
                },
            }

        mock_processor1.process_query_async = mock_async1

        with patch("lambda_handler._init_processor", return_value=mock_processor1):
            response1 = lambda_handler(event1, mock_context)

            # Verify Turn 1 response
            assert response1["statusCode"] == 200
            body1 = response1["body"]
            assert '"success": true' in body1
            assert '"row_count": 5' in body1

        # Turn 2 - Reference previous context
        event2 = {
            "query": "Now show me the bottom 5",
            "prior_turns": [
                {
                    "content": "Show me the top 5 products by revenue",
                    "sql": "SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue DESC LIMIT 5",
                    "visualization": {"type": "bar"},
                }
            ],
            "export_to_s3": False,
            "generate_visualization": True,
        }

        # Mock the processor for Turn 2
        mock_processor2 = Mock()

        async def mock_async2(**kwargs):
            # Verify prior_turns were passed correctly
            assert "prior_turns" in kwargs
            assert len(kwargs["prior_turns"]) == 1
            assert (
                kwargs["prior_turns"][0]["content"]
                == "Show me the top 5 products by revenue"
            )

            return {
                "success": True,
                "sql": "SELECT product_name, SUM(revenue) FROM products GROUP BY product_name ORDER BY revenue ASC LIMIT 5",
                "row_count": 5,
                "cache_hit": False,
                "total_duration_ms": 400,
                "visualization": {
                    "chart": {
                        "type": "bar",
                        "config": {"x": "product_name", "y": "revenue"},
                    }
                },
            }

        mock_processor2.process_query_async = mock_async2

        with patch("lambda_handler._init_processor", return_value=mock_processor2):
            response2 = lambda_handler(event2, mock_context)

            # Verify Turn 2 response
            assert response2["statusCode"] == 200
            body2 = response2["body"]
            assert '"success": true' in body2
            assert '"row_count": 5' in body2

    @pytest.mark.integration
    def test_visualization_change_request(self):
        """Test changing visualization type in second turn."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Turn 1 - Initial query with bar chart
        event1 = {
            "query": "Show sales by region",
        }

        # Mock the processor for Turn 1
        mock_processor1 = Mock()

        async def mock_async1(**kwargs):
            return {
                "success": True,
                "sql": "SELECT region, SUM(sales) FROM sales_data GROUP BY region",
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 500,
                "visualization": {
                    "chart": {"type": "bar", "config": {"x": "region", "y": "sales"}}
                },
            }

        mock_processor1.process_query_async = mock_async1

        with patch("lambda_handler._init_processor", return_value=mock_processor1):
            response1 = lambda_handler(event1, mock_context)

            # Verify Turn 1 response
            assert response1["statusCode"] == 200

        # Turn 2 - Change to pie chart
        event2 = {
            "query": "Change to pie chart",
            "prior_turns": [
                {
                    "content": "Show sales by region",
                    "sql": "SELECT region, SUM(sales) FROM sales_data GROUP BY region",
                    "visualization": {"type": "bar"},
                }
            ],
        }

        # Mock the processor for Turn 2
        mock_processor2 = Mock()

        async def mock_async2(**kwargs):
            # Verify prior_turns were passed correctly
            assert "prior_turns" in kwargs
            prior_turns = kwargs["prior_turns"]
            assert prior_turns[0]["visualization"]["type"] == "bar"

            return {
                "success": True,
                "sql": "SELECT region, SUM(sales) FROM sales_data GROUP BY region",  # Same SQL
                "row_count": 10,
                "cache_hit": False,
                "total_duration_ms": 300,
                "visualization": {
                    "chart": {
                        "type": "pie",
                        "config": {"values": "sales", "names": "region"},
                    }
                },
            }

        mock_processor2.process_query_async = mock_async2

        with patch("lambda_handler._init_processor", return_value=mock_processor2):
            response2 = lambda_handler(event2, mock_context)

            # Verify Turn 2 response
            assert response2["statusCode"] == 200
            body2 = response2["body"]
            assert '"success": true' in body2
            # Should have pie chart in visualization
            assert '"type": "pie"' in body2

    @pytest.mark.integration
    def test_three_turn_conversation_with_truncation(self):
        """Test three-turn conversation with proper context handling."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Create a three-turn conversation
        prior_turns = [
            {
                "content": "Show me all customers",
                "sql": "SELECT * FROM customers",
                "visualization": {"type": "table"},
            },
            {
                "content": "Filter by active status",
                "sql": "SELECT * FROM customers WHERE status = 'active'",
                "visualization": {"type": "table"},
            },
        ]

        # Turn 3 - Reference the first turn (should use last 2 turns for context)
        event3 = {
            "query": "Show me more columns from the first query",
            "prior_turns": prior_turns,
        }

        # Mock the processor for Turn 3
        mock_processor3 = Mock()

        async def mock_async3(**kwargs):
            # Verify only last 2 turns are included
            assert "prior_turns" in kwargs
            assert len(kwargs["prior_turns"]) == 2
            # Should include both turns from our list
            assert kwargs["prior_turns"][0]["content"] == "Show me all customers"
            assert kwargs["prior_turns"][1]["content"] == "Filter by active status"

            return {
                "success": True,
                "sql": "SELECT customer_id, name, email, phone, address FROM customers",
                "row_count": 100,
                "cache_hit": False,
                "total_duration_ms": 600,
                "visualization": {
                    "chart": {
                        "type": "table",
                        "config": {
                            "columns": [
                                "customer_id",
                                "name",
                                "email",
                                "phone",
                                "address",
                            ]
                        },
                    }
                },
            }

        mock_processor3.process_query_async = mock_async3

        with patch("lambda_handler._init_processor", return_value=mock_processor3):
            response3 = lambda_handler(event3, mock_context)

            # Verify Turn 3 response
            assert response3["statusCode"] == 200
            body3 = response3["body"]
            assert '"success": true' in body3
            assert '"row_count": 100' in body3

    @pytest.mark.integration
    def test_conversation_with_invalid_prior_turns(self):
        """Test conversation with invalid prior_turns format."""
        # Mock context
        mock_context = Mock()
        mock_context.aws_request_id = "test-request-id"
        mock_context.get_remaining_time_in_millis.return_value = 30000

        # Event with invalid prior_turns (missing content)
        event = {
            "query": "Show me more data",
            "prior_turns": [
                {"sql": "SELECT * FROM table"}  # Missing required content field
            ],
        }

        # Should return 400 error without processing
        response = lambda_handler(event, mock_context)

        assert response["statusCode"] == 400
        body = response["body"]
        assert '"success": false' in body
        assert '"error": "Invalid prior_turns format"' in body
