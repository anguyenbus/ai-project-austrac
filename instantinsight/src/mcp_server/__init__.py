"""MCP Server package for instantinsight.

This package provides an MCP (Model Context Protocol) server implementation
that exposes instantinsight's natural language to SQL query processing capabilities
via HTTP endpoints.

Main Components:
    - server: FastMCP server with query processing tools
    
Usage:
    >>> uv run python -m src.mcp_server.server
"""

from src.mcp_server.server import mcp, query, list_capabilities

__all__ = ["mcp", "query", "list_capabilities"]