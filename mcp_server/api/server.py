"""FastAPI server for the MCP server."""

import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from mcp_server.config import settings
from mcp_server.api.routes import router as api_router

# Initialize the FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model-Context-Protocol Server with GPT and LangChain",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key."""
    if settings.DEBUG:
        return True

    if not settings.SECRET_KEY:
        return True

    if api_key == settings.SECRET_KEY:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
    )


# Include API routes
app.include_router(api_router, dependencies=[Depends(get_api_key)])


# Health check route
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("MCP Server is starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print("MCP Server is shutting down...")
