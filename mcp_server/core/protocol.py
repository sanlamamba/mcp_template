"""Protocol component of the MCP pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcp_server.core.context import Context
from mcp_server.core.model import Model


class Protocol(ABC):
    """Base Protocol class for the MCP pattern."""

    @abstractmethod
    async def process(self, input_data: Any, context: Context) -> Any:
        """Process input data using the protocol."""
        pass


class ChatProtocol(Protocol):
    """Chat-based implementation of the Protocol component."""

    def __init__(self, model: Model):
        """Initialize the chat protocol."""
        self.model = model

    async def process(self, input_data: str, context: Context) -> str:
        """Process chat input using the protocol."""
        # Get conversation history from context
        history = context.get("conversation_history", [])

        # Add user message to history
        history.append({"role": "user", "content": input_data})

        # Format the conversation for the model
        formatted_prompt = self._format_conversation(history)

        # Get response from model
        system_prompt = context.get("system_prompt", "You are a helpful assistant.")
        full_prompt = f"{system_prompt}\n\n{formatted_prompt}"

        response = await self.model.generate(full_prompt)

        # Add assistant response to history
        history.append({"role": "assistant", "content": response})

        # Update context with new history
        context.add("conversation_history", history)

        return response

    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        """Format the conversation history for the model."""
        formatted = ""
        for message in history:
            role = message["role"].capitalize()
            content = message["content"]
            formatted += f"{role}: {content}\n\n"

        formatted += "Assistant: "
        return formatted
