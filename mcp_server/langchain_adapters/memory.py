"""LangChain memory implementation for the MCP server."""

from typing import Any, Dict, List, Optional

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)

from mcp_server.core.context import Context, MemoryContext


class LangChainContextAdapter(Context):
    """LangChain memory adapter for the Context component."""

    def __init__(self, memory: Any, context: Optional[Context] = None):
        """Initialize with a LangChain memory."""
        self.memory = memory
        self.context = context or MemoryContext()

    def add(self, key: str, value: Any) -> None:
        """Add an item to the context."""
        if key == "conversation_history":
            # Handle conversation history specially for LangChain
            self._update_memory_from_history(value)

        # Always update internal context
        self.context.add(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from the context."""
        # For conversation history, combine memory and context
        if key == "conversation_history":
            return self._get_combined_history()

        return self.context.get(key, default)

    def remove(self, key: str) -> None:
        """Remove an item from the context."""
        if key == "conversation_history":
            # Clear memory
            if hasattr(self.memory, "clear"):
                self.memory.clear()

        self.context.remove(key)

    def clear(self) -> None:
        """Clear all items from the context."""
        if hasattr(self.memory, "clear"):
            self.memory.clear()

        self.context.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary."""
        result = self.context.to_dict()

        # Add memory variables
        memory_vars = self.memory.load_memory_variables({})
        for key, value in memory_vars.items():
            result[key] = value

        return result

    def _update_memory_from_history(self, history: List[Dict[str, str]]) -> None:
        """Update LangChain memory from conversation history."""
        if not history:
            return

        # Get last user and AI messages
        user_msg = None
        ai_msg = None

        for msg in reversed(history):
            if msg["role"] == "user" and user_msg is None:
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and ai_msg is None:
                ai_msg = msg["content"]

            if user_msg and ai_msg:
                break

        # Update memory if we found both messages
        if user_msg and ai_msg:
            self.memory.save_context({"input": user_msg}, {"output": ai_msg})

    def _get_combined_history(self) -> List[Dict[str, str]]:
        """Get combined history from context and memory."""
        # Get history from context
        history = self.context.get("conversation_history", [])

        # Try to get additional history from memory
        if hasattr(self.memory, "chat_memory") and hasattr(
            self.memory.chat_memory, "messages"
        ):
            for msg in self.memory.chat_memory.messages:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = "user" if msg.type == "human" else "assistant"
                    # Check if this message is already in history
                    if not any(
                        h["role"] == role and h["content"] == msg.content
                        for h in history
                    ):
                        history.append({"role": role, "content": msg.content})

        return history


def create_buffer_memory() -> ConversationBufferMemory:
    """Create a LangChain conversation buffer memory."""
    return ConversationBufferMemory()


def create_summary_memory(llm: Any) -> ConversationSummaryMemory:
    """Create a LangChain conversation summary memory."""
    return ConversationSummaryMemory(llm=llm)


def create_window_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """Create a LangChain conversation buffer window memory."""
    return ConversationBufferWindowMemory(k=k)
