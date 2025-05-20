"""Base agent class for MCP implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcp_server.core.context import Context
from mcp_server.core.model import Model
from mcp_server.core.protocol import Protocol


class Agent(ABC):
    """Base Agent class for the MCP pattern."""

    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        """Run the agent with input data."""
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Reset the agent's state."""
        pass


class BaseAgent(Agent):
    """Base implementation of an Agent."""

    def __init__(
        self,
        model: Model,
        protocol: Protocol,
        context: Optional[Context] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        """Initialize the base agent."""
        self.model = model
        self.protocol = protocol
        self.context = context or self._create_default_context()
        self.context.add("system_prompt", system_prompt)

    async def run(self, input_data: str) -> str:
        """Run the agent with input data."""
        return await self.protocol.process(input_data, self.context)

    async def reset(self) -> None:
        """Reset the agent's state."""
        self.context.clear()
        self.context.add("system_prompt", "You are a helpful assistant.")
        self.context.add("conversation_history", [])

    def _create_default_context(self) -> Context:
        """Create a default context."""
        from mcp_server.core.context import MemoryContext

        context = MemoryContext()
        context.add("conversation_history", [])
        return context
