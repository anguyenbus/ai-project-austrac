"""instantinsight MCP Server - Natural Language to SQL Query Processing.

This module provides an MCP (Model Context Protocol) server that exposes
the QueryProcessor functionality via HTTP endpoints. It allows clients to
send natural language queries and receive SQL queries, executed results,
and visualizations.

Key Features:
    - Natural language to SQL conversion via QueryProcessor
    - Asynchronous query processing for Lambda compatibility
    - Session-based conversation tracking
    - Result caching and export capabilities
    - Visualization generation

Usage:
    Local Development:
        >>> uv run python -m src.mcp_server.server http

    This starts the server on http://0.0.0.0:8000

Architecture:
    Client → FastMCP Server → QueryProcessor → RAG Pipeline → Database
"""

import os
from datetime import datetime
from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict
from dotenv import load_dotenv
from fastmcp import FastMCP
from loguru import logger

from src.query_processor import QueryProcessor

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("instantinsight Natural Language Query Server")

# Global QueryProcessor instance
query_processor: QueryProcessor | None = None

# Server configuration
MCP_HOST: Final[str] = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT: Final[int] = int(os.environ.get("MCP_PORT", "8000"))


def _initialize_query_processor() -> None:
    """Initialize the QueryProcessor instance.

    This function creates a singleton QueryProcessor instance if it doesn't
    already exist. The processor is configured based on environment variables.

    Raises:
        RuntimeError: If QueryProcessor initialization fails

    """
    global query_processor

    if query_processor is None:
        try:
            logger.info("🔧 Initializing QueryProcessor...")
            query_processor = QueryProcessor()
            logger.info("✅ QueryProcessor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize QueryProcessor: {e}")
            raise RuntimeError(f"Failed to initialize QueryProcessor: {e}") from e


@mcp.tool()
@beartype
async def query(
    text: str,
    session_id: str | None = None,
    export_results: bool = True,
    generate_visualization: bool = True,
) -> Dict[str, Any]:
    """Process natural language query and return SQL, data, and visualization.

    This tool orchestrates the complete query processing pipeline:
    1. Converts natural language to SQL using RAG pipeline
    2. Executes the SQL query against the database
    3. Optionally exports results to CSV
    4. Optionally generates visualization schema

    Args:
        text: Natural language query (e.g., "How many products are discontinued?")
        session_id: Optional session ID for conversation tracking
        export_results: Whether to export results to CSV (default: True)
        generate_visualization: Whether to generate Plotly visualization (default: True)

    Returns:
        Dictionary containing:
            - query: Original natural language query
            - sql: Generated SQL statement
            - data: Query results as records (list of dicts)
            - success: Whether processing succeeded
            - row_count: Number of rows returned
            - visualization: Plotly chart schema (if requested)
            - export_path: Path to exported CSV (if requested)
            - cache_hit: Whether result came from cache
            - total_duration_ms: Total processing time
            - error: Error message (if failed)

    Example:
        >>> result = await query("How many products are discontinued for each category?")
        >>> print(result["sql"])
        >>> print(result["data"])
        >>> print(result["row_count"])

    """
    # Initialize QueryProcessor if needed
    _initialize_query_processor()

    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    logger.info(
        f"📥 Processing query: '{text[:50]}...' | Session: {session_id[:8] if session_id else 'none'}"
    )

    try:
        # Process query through QueryProcessor pipeline
        result = await query_processor.process_query_async(
            query=text,
            prior_turns=None,  # NOTE: Multi-turn support can be added later
            export_results=export_results,
            generate_visualization=generate_visualization,
        )

        # Convert DataFrame to records for JSON serialization
        if result.get("data") is not None:
            result["data"] = result["data"].to_dict(orient="records")

        # Add session tracking
        result["session_id"] = session_id

        if result.get("success"):
            logger.info(
                f"✅ Query processed successfully | "
                f"Rows: {result.get('row_count', 0)} | "
                f"Duration: {result.get('total_duration_ms', 0)}ms | "
                f"Cache: {result.get('cache_hit', False)}"
            )
        else:
            logger.warning(
                f"⚠️ Query processing failed: {result.get('error', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"❌ Unexpected error processing query: {e}")
        return {
            "query": text,
            "success": False,
            "error": str(e),
            "error_type": "unexpected_error",
            "session_id": session_id,
            "cache_hit": False,
        }


@mcp.tool()
@beartype
async def list_capabilities() -> Dict[str, Any]:
    """List server capabilities and configuration.

    Returns:
        Dictionary with server information:
            - version: Server version
            - features: Available features
            - cache_enabled: Whether semantic caching is enabled
            - export_enabled: Whether result export is enabled

    """
    _initialize_query_processor()

    return {
        "server": "instantinsight MCP Server",
        "version": "0.1.0",
        "features": {
            "natural_language_to_sql": True,
            "query_execution": True,
            "visualization_generation": True,
            "result_export": True,
            "semantic_caching": query_processor.config.get("enable_cache", False),
            "session_tracking": True,
        },
        "processor_config": {
            "cache_enabled": query_processor.config.get("enable_cache", False),
            "export_enabled": query_processor.config.get("export_enabled", False),
            "is_lambda": query_processor.config.get("is_lambda", False),
        },
    }


async def run_server() -> None:
    """Run the MCP server.

    Starts the FastMCP HTTP server on the configured host and port.
    Uses the same pattern as the reference fastmcp-example.

    """
    logger.info("🚀 Starting instantinsight MCP Server")
    logger.info(f"📡 Server will be available at http://{MCP_HOST}:{MCP_PORT}")
    logger.info("✅ Natural Language to SQL query processing enabled")

    # Run with HTTP transport for local testing
    await mcp.run_http_async(
        host=MCP_HOST,
        port=MCP_PORT,
    )


if __name__ == "__main__":
    import sys

    import anyio

    # Configure logging
    logger.add(
        "logs/mcp_server.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )

    # Get transport from command line or use default
    transport = sys.argv[1] if len(sys.argv) > 1 else "http"

    if transport == "stdio":
        logger.warning(
            "stdio transport is for desktop integration. "
            "HTTP transport recommended for testing."
        )
        mcp.run(transport="stdio", stateless_http=False)
    else:
        anyio.run(run_server)