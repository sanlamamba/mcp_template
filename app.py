"""Main application entry point for the MCP server."""

import uvicorn
from mcp_server.api.server import app
from mcp_server.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "mcp_server.api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
