"""API routes for the MCP server with fixed tools endpoint and optimizations."""

import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body, status
from pydantic import BaseModel, Field, validator

from mcp_server.agents.gpt_agent import GPTAgent
from mcp_server.langchain_adapters.tools import create_tools_toolkit
from mcp_server.utils.helpers import parse_json_safely

# Set up logger
logger = logging.getLogger(__name__)

# Initialize router with tags for better API documentation
router = APIRouter(prefix="/api")

# --- Models ---


class Message(BaseModel):
    """Chat message model with validation."""

    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")

    @validator("role")
    def validate_role(cls, v):
        """Validate message role."""
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


class ChatRequest(BaseModel):
    """Chat request model with enhanced validation."""

    prompt: str = Field(..., description="User prompt")
    model: str = Field("gpt-4o", description="Model name")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Temperature (0.0 to 2.0)"
    )
    system_prompt: Optional[str] = Field(None, description="System prompt")
    use_tools: bool = Field(False, description="Use tools")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    history: Optional[List[Message]] = Field(None, description="Conversation history")

    class Config:
        """Additional configuration."""

        schema_extra = {
            "example": {
                "prompt": "What's the weather like today?",
                "model": "gpt-4o",
                "temperature": 0.7,
                "system_prompt": "You are a helpful weather assistant.",
                "use_tools": True,
                "conversation_id": None,
                "history": None,
            }
        }


class ToolCall(BaseModel):
    """Tool call model."""

    tool: str = Field(..., description="Tool name")
    tool_input: Dict[str, Any] = Field(..., description="Tool inputs")
    executed: bool = Field(False, description="Whether the tool was executed")
    result: Optional[Any] = Field(None, description="Tool execution result")
    note: Optional[str] = Field(None, description="Additional note about the tool call")


class ChatResponse(BaseModel):
    """Chat response model with enhanced fields."""

    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")


class ModelInfo(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")


class AgentInfo:
    """Agent information with TTL for memory management."""

    def __init__(self, agent: GPTAgent, ttl_minutes: int = 60):
        self.agent = agent
        self.last_access = time.time()
        self.ttl_seconds = ttl_minutes * 60

    def update_access_time(self):
        """Update the last access time."""
        self.last_access = time.time()

    def is_expired(self) -> bool:
        """Check if the agent has expired."""
        return (time.time() - self.last_access) > self.ttl_seconds


# --- Agent Store ---


class AgentStore:
    """Enhanced agent store with TTL and memory management."""

    def __init__(self, cleanup_interval_minutes: int = 30):
        self._agents: Dict[str, AgentInfo] = {}
        self.cleanup_interval = cleanup_interval_minutes * 60
        self.last_cleanup = time.time()

    def get(self, agent_id: str) -> Optional[GPTAgent]:
        """Get an agent by ID and update its access time."""
        self._maybe_cleanup()

        if agent_id in self._agents:
            agent_info = self._agents[agent_id]
            agent_info.update_access_time()
            return agent_info.agent
        return None

    def add(self, agent_id: str, agent: GPTAgent, ttl_minutes: int = 60) -> None:
        """Add an agent to the store."""
        self._agents[agent_id] = AgentInfo(agent, ttl_minutes)

    def remove(self, agent_id: str) -> bool:
        """Remove an agent from the store."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def size(self) -> int:
        """Get the number of agents in the store."""
        return len(self._agents)

    def _maybe_cleanup(self) -> None:
        """Clean up expired agents if needed."""
        now = time.time()
        if (now - self.last_cleanup) > self.cleanup_interval:
            self._cleanup()
            self.last_cleanup = now

    def _cleanup(self) -> None:
        """Remove expired agents."""
        expired_ids = [
            agent_id
            for agent_id, agent_info in self._agents.items()
            if agent_info.is_expired()
        ]

        for agent_id in expired_ids:
            del self._agents[agent_id]


# Initialize the agent store
agent_store = AgentStore()


# --- Helper Functions ---


def parse_tool_calls(response: str) -> List[ToolCall]:
    """Parse tool calls from a response string."""
    tool_calls = []

    # Try to extract JSON from the response
    try:
        # Look for JSON objects in the response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            tool_data = parse_json_safely(json_str)

            if "tool" in tool_data and "tool_input" in tool_data:
                tool_calls.append(
                    ToolCall(
                        tool=tool_data["tool"],
                        tool_input=tool_data["tool_input"],
                        executed=False,
                        result=None,
                    )
                )
    except Exception as e:
        logger.warning(f"Error parsing tool calls: {str(e)}")
        # If parsing fails, fall back to simple string matching
        if '"tool":' in response and '"tool_input":' in response:
            tool_calls.append(
                ToolCall(
                    tool="unknown",
                    tool_input={},
                    executed=False,
                    result=None,
                    note="Could not parse tool call format",
                )
            )

    return tool_calls


async def get_or_create_agent(request: ChatRequest) -> Tuple[str, GPTAgent, bool]:
    """Get or create an agent based on the request."""
    agent_id = request.conversation_id or f"agent_{agent_store.size() + 1}"
    is_new = False

    # Get existing agent or create new one
    agent = agent_store.get(agent_id)
    if agent is None:
        # Create a new agent
        agent = GPTAgent(
            model_name=request.model,
            temperature=request.temperature,
            system_prompt=request.system_prompt or "You are a helpful assistant.",
        )
        agent_store.add(agent_id, agent)
        is_new = True

    return agent_id, agent, is_new


def get_available_tools() -> List[Dict[str, str]]:
    """Get available tools with error handling."""
    try:
        tools = create_tools_toolkit()

        # Format as dictionaries
        tool_infos = []
        for tool in tools:
            # Handle both LangChain tools and BasicTool objects
            if hasattr(tool, "name") and hasattr(tool, "description"):
                tool_infos.append({"name": tool.name, "description": tool.description})

        # Ensure we always return at least one tool
        if not tool_infos:
            # Add fallback tools if no tools were found
            tool_infos.append(
                {"name": "echo", "description": "Repeats back what you say to it."}
            )
            tool_infos.append(
                {
                    "name": "calculator",
                    "description": "Performs basic mathematical calculations.",
                }
            )

        return tool_infos
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")

        # Return fallback tools
        return [
            {"name": "echo", "description": "Repeats back what you say to it."},
            {
                "name": "calculator",
                "description": "Performs basic mathematical calculations.",
            },
        ]


# --- Routes ---


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with the AI agent",
    response_description="The AI agent's response",
)
async def chat(request: ChatRequest = Body(...)):
    """
    Chat with the MCP agent.

    This endpoint processes a chat request and returns the agent's response.
    If tools are enabled, the agent may call tools to help answer the query.

    - **prompt**: The user's input message
    - **model**: The model to use (default: gpt-4o)
    - **temperature**: Controls randomness (0.0-2.0)
    - **system_prompt**: Optional system instructions
    - **use_tools**: Whether to allow tool usage
    - **conversation_id**: Optional ID for continuing a conversation
    - **history**: Optional conversation history
    """
    try:
        # Get or create agent
        agent_id, agent, is_new = await get_or_create_agent(request)

        # Add history if provided and not a new agent
        if request.history and not is_new:
            conversation_history = [
                {"role": message.role, "content": message.content}
                for message in request.history
            ]
            agent.context.add("conversation_history", conversation_history)

        # Process the request
        if request.use_tools:
            # Get tools
            tool_infos = get_available_tools()

            if not tool_infos:
                # No tools available, process without tools
                response = await agent.run(request.prompt)
                tool_calls_dict = None
            else:
                # Process with tools
                response = await agent.run_with_tools(request.prompt, tool_infos)

                # Parse tool calls from the response
                tool_calls = parse_tool_calls(response)

                # Convert tool calls to dict for response
                tool_calls_dict = (
                    [tool_call.dict() for tool_call in tool_calls]
                    if tool_calls
                    else None
                )
        else:
            # Process without tools
            response = await agent.run(request.prompt)
            tool_calls_dict = None

        # Return the response
        return ChatResponse(
            response=response,
            conversation_id=agent_id,
            tool_calls=tool_calls_dict,
            created_at=datetime.now(),
        )

    except Exception as e:
        # Log the error
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        # Raise HTTP exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}",
        )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation",
    response_description="Deletion status",
)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation by ID.

    This endpoint removes an agent and its conversation history from the store.

    - **conversation_id**: The ID of the conversation to delete
    """
    # Try to remove the agent
    if agent_store.remove(conversation_id):
        return {"status": "deleted", "conversation_id": conversation_id}

    # Agent not found
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Conversation with ID {conversation_id} not found",
    )


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="List available models",
    response_description="Available models",
)
async def list_models():
    """
    List available language models.

    This endpoint returns the available language models that can be used with the agent.
    """
    # Define available models
    models = [
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            description="Latest GPT-4 model with improved capabilities",
        ),
        ModelInfo(
            id="gpt-4-turbo", name="GPT-4 Turbo", description="Faster version of GPT-4"
        ),
        ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            description="Fast and cost-effective model",
        ),
    ]

    return {"models": [model.dict() for model in models]}


@router.get(
    "/tools",
    status_code=status.HTTP_200_OK,
    summary="List available tools",
    response_description="Available tools",
)
async def list_tools():
    """
    List available tools.

    This endpoint returns the available tools that can be used by the agent.
    """
    tool_infos = get_available_tools()
    return {"tools": tool_infos}


@router.get(
    "/conversations/{conversation_id}/context",
    status_code=status.HTTP_200_OK,
    summary="Get conversation context",
    response_description="Conversation context",
)
async def get_conversation_context(conversation_id: str):
    """
    Get the context for a conversation.

    This endpoint returns the current context for a conversation.

    - **conversation_id**: The ID of the conversation
    """
    # Get the agent
    agent = agent_store.get(conversation_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found",
        )

    # Get the context
    try:
        context = agent.context.to_dict()

        # Remove sensitive or internal fields
        if "conversation_history" in context:
            # Only return the last 10 messages
            history = context["conversation_history"][-10:]
            context["conversation_history"] = history

        return {"context": context}
    except Exception as e:
        logger.error(f"Error getting conversation context: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation context: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    response_description="Health status",
    tags=["System"],
)
async def health_check():
    """
    Health check endpoint.

    This endpoint returns the health status of the API and its components.
    """
    # Check if tools are available
    tools_available = len(get_available_tools()) > 0

    # Check agent store
    agent_store_status = "ok"
    try:
        agent_store._maybe_cleanup()
    except Exception as e:
        agent_store_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "components": {
            "tools": "ok" if tools_available else "degraded",
            "agent_store": agent_store_status,
            "server": "ok",
        },
        "timestamp": datetime.now().isoformat(),
    }
