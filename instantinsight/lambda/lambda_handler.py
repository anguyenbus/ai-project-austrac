"""
AWS Lambda handler for QueryProcessor with container deployment.

This handler is designed for Lambda container images and supports:
- Cold start optimization with global instance reuse
- Structured error handling and logging
- CloudWatch integration
- S3 result exports
- S3-backed session management
- Timeout management
- CORS support
"""

import asyncio
import json
import os
import uuid
from typing import Any

from loguru import logger

from src.query_processor import QueryProcessor
from src.session import SessionStore, query_result_to_turn

# Configure logger for CloudWatch
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),  # CloudWatch captures stdout
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# Global instances for Lambda container reuse
_processor: QueryProcessor | None = None
_session_store: SessionStore | None = None


def _init_processor() -> QueryProcessor:
    """
    Initialize QueryProcessor with Lambda configuration.

    This is called once per container lifecycle and reused
    across invocations (warm starts).

    Returns:
        Initialized QueryProcessor instance

    """
    global _processor

    if _processor is None:
        logger.info("Initializing QueryProcessor for Lambda container")

        # Lambda-specific configuration from environment
        config = {
            "use_pipeline": True,
            "enable_cache": True,
            "export_enabled": True,
            "s3_bucket": os.getenv("RESULTS_BUCKET", "instantinsight-query-results"),
            "s3_prefix": os.getenv("RESULTS_PREFIX", "query-results/"),
            "is_lambda": True,
            "query_timeout_seconds": float(os.getenv("QUERY_TIMEOUT", "120.0")),
            "max_result_rows": int(os.getenv("MAX_RESULT_ROWS", "50000")),
        }

        _processor = QueryProcessor(config=config)
        logger.info("✅ QueryProcessor initialized successfully")

    return _processor


def _init_session_store() -> SessionStore | None:
    """
    Initialize SessionStore with S3 configuration.

    Returns None if S3 session storage is disabled via feature flag.

    Returns:
        Initialized SessionStore instance or None

    """
    global _session_store

    # Check feature flag
    use_s3_session = os.getenv("USE_S3_SESSION", "true").lower() == "true"

    if not use_s3_session:
        logger.info("S3 session storage disabled via USE_S3_SESSION flag")
        return None

    if _session_store is None:
        logger.info("Initializing SessionStore for Lambda container")

        session_bucket = os.getenv("SESSION_BUCKET", "instantinsight-session")
        session_prefix = os.getenv("SESSION_PREFIX", "sessions/")

        _session_store = SessionStore(bucket=session_bucket, prefix=session_prefix)
        logger.info("✅ SessionStore initialized successfully")

    return _session_store


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda handler for query processing with S3-backed sessions.

    Event Structure:
    {
        "query": "Natural language query",
        "session_id": "optional-uuid-for-multi-turn",
        "user_id": "optional-user-id",
        "export_to_s3": true,
        "generate_visualization": true
    }

    Response Structure:
    {
        "statusCode": 200 | 400 | 500 | 504,
        "body": {
            "success": true/false,
            "session_id": "uuid-v4",
            "turn_count": 1-6,
            "sql": "Generated SQL",
            "row_count": 123,
            "export_path": "s3://bucket/key",
            "cache_hit": true/false,
            "duration_ms": 1234,
            "visualization": {
                "chart_type": "bar",
                "config": {...}
            },
            "error": "Optional error message",
            "error_type": "Optional error type"
        },
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
    }

    Note:
        - prior_turns parameter is NO LONGER SUPPORTED (use session_id instead)
        - Session state is managed server-side in S3
        - Timeout is controlled by QUERY_TIMEOUT environment variable

    Args:
        event: Lambda event containing query and optional session_id
        context: Lambda context with runtime information

    Returns:
        Lambda response with statusCode, body, and headers

    """
    try:
        # Log invocation details
        logger.info(f"Lambda invocation: {context.aws_request_id}")
        logger.info(f"Remaining time: {context.get_remaining_time_in_millis()}ms")

        # Initialize components (reused across warm starts)
        processor = _init_processor()
        session_store = _init_session_store()

        # Extract and validate parameters
        query = event.get("query")
        if not query:
            return _error_response(400, "Query parameter is required")

        # WARN: Reject legacy prior_turns parameter
        if "prior_turns" in event:
            return _error_response(
                400,
                "prior_turns parameter is no longer supported. Use session_id for multi-turn conversations.",
            )

        # Extract or generate session_id
        session_id = event.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session_id: {session_id}")
        else:
            logger.info(f"Continuing session: {session_id}")

        user_id = event.get("user_id")
        export = event.get("export_to_s3", True)
        viz = event.get("generate_visualization", True)

        # Load prior turns from S3 if session store is enabled
        prior_turns: list[dict[str, str]] = []
        if session_store:
            try:
                session = session_store.load(session_id)
                prior_turns = session["turns"]
                logger.info(f"Loaded {len(prior_turns)} prior turns from session")
            except Exception as e:
                logger.error(f"Failed to load session: {e}")
                return _error_response(
                    500,
                    f"Failed to load session state: {str(e)}",
                    error_type="session_load_error",
                )

        logger.info(f"Processing query: {query[:100]}...")
        if prior_turns:
            logger.info(f"Conversation context: {len(prior_turns)} prior turns")

        # Process query asynchronously
        # NOTE: Timeout is controlled by config, not passed as parameter
        result = asyncio.run(
            processor.process_query_async(
                query=query,
                prior_turns=prior_turns,
                export_results=export,
                generate_visualization=viz,
                user_id=user_id,
            )
        )

        # Check remaining time
        remaining_ms = context.get_remaining_time_in_millis()
        if remaining_ms < 5000:  # Less than 5 seconds
            logger.warning(f"Lambda approaching timeout: {remaining_ms}ms remaining")

        # Build response
        if result["success"]:
            logger.info(
                f"✅ Query processed successfully in {result.get('total_duration_ms')}ms"
            )

            # Save turn to session if session store is enabled
            turn_count = len(prior_turns) + 1
            if session_store:
                try:
                    turn = query_result_to_turn(result)
                    session_store.append(session_id, turn)
                    logger.info(f"Saved turn to session (total: {turn_count} turns)")
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")
                    # WARN: Session save failure is fatal - return error
                    return _error_response(
                        500,
                        f"Failed to save session state: {str(e)}",
                        error_type="session_save_error",
                    )

            # Extract visualization chart if present, else None
            visualization = result.get("visualization", {})
            chart = visualization.get("chart") if visualization else None

            if chart:
                logger.info(json.dumps(chart, indent=2))
            else:
                logger.info("No visualization chart found")

            response_body = {
                "success": True,
                "sql": result.get("sql"),
                "row_count": result.get("row_count"),
                "export_path": result.get("export_path"),
                "cache_hit": result.get("cache_hit", False),
                "duration_ms": result.get("total_duration_ms"),
                "visualization": chart,  # Always include (None if not generated)
            }

            return _success_response(response_body, session_id, turn_count)
        else:
            logger.error(f"Query processing failed: {result.get('error')}")
            return _error_response(
                500,
                result.get("error", "Query processing failed"),
                error_type=result.get("error_type"),
            )

    except asyncio.TimeoutError:
        logger.error("Lambda function timed out")
        return _error_response(504, "Query processing timed out", error_type="timeout")

    except Exception as e:
        logger.exception(f"Unexpected error in Lambda handler: {e}")
        return _error_response(
            500, f"Internal server error: {str(e)}", error_type="unexpected_error"
        )


def _success_response(
    body: dict[str, Any], session_id: str, turn_count: int
) -> dict[str, Any]:
    """
    Build successful Lambda response with session tracking.

    Args:
        body: Response body dictionary
        session_id: UUIDv4 session identifier
        turn_count: Current turn number (1-6)

    Returns:
        Formatted Lambda response with 200 status code and session metadata

    """
    # Add session metadata to response
    body["session_id"] = session_id
    body["turn_count"] = turn_count

    return {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Configure CORS as needed
        },
    }


def _error_response(
    status_code: int,
    error_message: str,
    error_type: str | None = None,
) -> dict[str, Any]:
    """
    Build error Lambda response.

    Args:
        status_code: HTTP status code (400, 500, 504, etc.)
        error_message: Human-readable error message
        error_type: Optional error type classification

    Returns:
        Formatted Lambda response with error details

    """
    body = {
        "success": False,
        "error": error_message,
    }

    if error_type:
        body["error_type"] = error_type

    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }


# For local testing
if __name__ == "__main__":
    """Test handler locally with sample event."""

    class MockContext:
        """Mock Lambda context for local testing."""

        aws_request_id = "local-test-123"

        def get_remaining_time_in_millis(self):
            """Return mock remaining time in milliseconds."""
            return 300000  # 5 minutes

    test_event = {
        "query": "Show me the top 10 products by revenue",
        "export_to_s3": False,  # Use local export for testing
        "generate_visualization": True,
    }

    response = lambda_handler(test_event, MockContext())

    print("\n" + "=" * 80)
    print("Lambda Response:")
    print("=" * 80)
    print(json.dumps(response, indent=2))
