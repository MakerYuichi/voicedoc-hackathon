"""
MongoDB MCP Server integration for VoiceDoc Intelligence.

The MCP client spawns the official mongodb-mcp-server as a subprocess
and communicates with it over JSON-RPC (stdio transport).
"""
from app.mcp.mcp_client import MCPClient, mcp_client

__all__ = ["MCPClient", "mcp_client"]
