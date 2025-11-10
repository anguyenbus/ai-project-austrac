"""Test client for instantinsight MCP Server.

This script demonstrates how to interact with the MCP server locally,
sending natural language queries and receiving structured results.

Usage:
    # Start the server first in another terminal:
    >>> uv run python -m src.mcp_server.server http

    # Then run this test client:
    >>> uv run python tests/mcp/test_mcp_client.py

Expected Flow:
    1. Client initializes MCP session
    2. Client sends POST requests with mcp-session-id header
    3. Server processes query through QueryProcessor pipeline
    4. Server returns SQL, data, and visualization
    5. Client displays results
"""

import json
from typing import Any, Final

import httpx
from loguru import logger


def parse_sse_response(text: str) -> dict[str, Any]:
    """Parse Server-Sent Events (SSE) response format.

    Args:
        text: Raw SSE response text

    Returns:
        Parsed JSON data from the SSE event

    Raises:
        json.JSONDecodeError: If the data cannot be parsed as JSON

    """
    # SSE format: "event: message\ndata: {json}\n\n"
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            return json.loads(data_str)

    # If no data line found, try parsing as JSON directly
    return json.loads(text)

# Server configuration
MCP_SERVER_URL: Final[str] = "http://localhost:8000"
TIMEOUT_SECONDS: Final[int] = 120


def initialize_session() -> str:
    """Initialize MCP session and return session ID.

    Returns:
        MCP session ID from server

    Raises:
        httpx.HTTPError: If the request fails

    """
    logger.info("🔌 Initializing MCP session")

    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 0,
        "params": {
            "protocolVersion": "0.1.0",
            "clientInfo": {
                "name": "instantinsight-test-client",
                "version": "1.0.0",
            },
        },
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            # Step 1: Send initialize request
            response = client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=init_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            # Extract session ID from response headers
            session_id = response.headers.get("mcp-session-id")
            if not session_id:
                msg = "Server did not return mcp-session-id header"
                raise RuntimeError(msg)

            logger.info(f"✅ Session ID obtained: {session_id}")

            # Step 2: Send initialized notification (required by MCP protocol)
            notification_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }

            client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=notification_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )

            logger.info(f"✅ Session fully initialized: {session_id}")
            return session_id

    except httpx.HTTPError as e:
        logger.error(f"❌ Failed to initialize session: {e}")
        raise


def test_query(
    query_text: str, mcp_session_id: str, session_id: str | None = None
) -> dict[str, Any]:
    """Send a query to the MCP server and return the result.

    Args:
        query_text: Natural language query to process
        mcp_session_id: MCP session ID from initialization
        session_id: Optional session ID for conversation tracking

    Returns:
        Dictionary with query results including SQL, data, and metadata

    Raises:
        httpx.HTTPError: If the request fails

    """
    logger.info(f"📤 Sending query: {query_text}")

    # Construct MCP tool call payload
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "text": query_text,
                "session_id": session_id,
                "export_results": True,
                "generate_visualization": True,
            },
        },
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": mcp_session_id,
                },
            )
            response.raise_for_status()

            # Parse SSE response
            result = parse_sse_response(response.text)

            # Extract the actual result from MCP response
            if "result" in result:
                actual_result = result["result"].get("content", [])
                if actual_result and len(actual_result) > 0:
                    # Parse the text content as JSON
                    result_data = json.loads(actual_result[0].get("text", "{}"))
                    return result_data

            return result

    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse response: {e}")
        raise


def test_list_capabilities(mcp_session_id: str) -> dict[str, Any]:
    """Query server capabilities.

    Args:
        mcp_session_id: MCP session ID from initialization

    Returns:
        Dictionary with server capabilities and configuration

    """
    logger.info("📤 Requesting server capabilities")

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "list_capabilities",
            "arguments": {},
        },
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{MCP_SERVER_URL}/mcp",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": mcp_session_id,
                },
            )
            response.raise_for_status()

            # Parse SSE response
            result = parse_sse_response(response.text)

            if "result" in result:
                actual_result = result["result"].get("content", [])
                if actual_result and len(actual_result) > 0:
                    result_data = json.loads(actual_result[0].get("text", "{}"))
                    return result_data

            return result

    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP request failed: {e}")
        raise


def display_result(result: dict[str, Any]) -> None:
    """Display query result in a formatted way.

    Args:
        result: Query result dictionary

    """
    print("\n" + "=" * 80)
    print("QUERY RESULT")
    print("=" * 80)

    print(f"\n📝 Original Query: {result.get('query', 'N/A')}")
    print(f"✅ Success: {result.get('success', False)}")
    print(f"⏱️  Duration: {result.get('total_duration_ms', 0)}ms")
    print(f"💾 Cache Hit: {result.get('cache_hit', False)}")

    if result.get("sql"):
        print(f"\n🔍 Generated SQL:\n{result['sql']}")

    if result.get("row_count") is not None:
        print(f"\n📊 Rows Returned: {result['row_count']}")

    if result.get("data"):
        print("\n📋 Data (first 5 rows):")
        data = result["data"][:5] if isinstance(result["data"], list) else result["data"]
        print(json.dumps(data, indent=2))

    if result.get("export_path"):
        print(f"\n💾 Exported to: {result['export_path']}")

    if result.get("visualization"):
        print("\n📈 Visualization schema available")

    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        print(f"   Type: {result.get('error_type', 'unknown')}")

    print("\n" + "=" * 80 + "\n")


def main() -> None:
    """Run the test client with example queries."""
    logger.info("🚀 Starting instantinsight MCP Client Test")
    logger.info(f"🔗 Connecting to: {MCP_SERVER_URL}")

    try:
        # Initialize MCP session
        mcp_session_id = initialize_session()

        # Test 1: Check server capabilities
        logger.info("\n--- Test 1: Server Capabilities ---")
        capabilities = test_list_capabilities(mcp_session_id)
        print("\n📋 Server Capabilities:")
        print(json.dumps(capabilities, indent=2))

        # Test 2: Example query - Discontinued products by category
        logger.info("\n--- Test 2: Query - Discontinued Products by Category ---")
        query_text = "How many products are discontinued for each category?"
        result = test_query(query_text, mcp_session_id, session_id="test-session-001")
        display_result(result)

        # Test 3: Another example query
        logger.info("\n--- Test 3: Query - Top Products by Revenue ---")
        query_text = "Show me the top 10 products by revenue"
        result = test_query(query_text, mcp_session_id, session_id="test-session-001")
        display_result(result)

        logger.info("✅ All tests completed successfully")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/mcp_client_test.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )

    main()