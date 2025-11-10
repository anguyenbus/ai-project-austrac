# Makefile for instantinsight Project

.PHONY: help mcp-local-run mcp-local-test

# Default target - show help
help:
	@echo "Available commands:"
	@echo "  make mcp-local-run   - Start the MCP server on http://0.0.0.0:8000"
	@echo "  make mcp-local-test  - Run the MCP client test suite"

# Start the MCP server
mcp-local-run:
	@echo "🚀 Starting instantinsight MCP Server..."
	@echo "📡 Server will be available at http://0.0.0.0:8000/mcp"
	@echo "Press Ctrl+C to stop"
	@echo ""
	uv run python -m src.mcp_server.server http

# Run the MCP client tests
mcp-local-test:
	@echo "🧪 Running MCP Client Tests..."
	@echo ""
	uv run python tests/mcp/test_mcp_client.py