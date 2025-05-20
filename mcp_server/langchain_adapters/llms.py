"""LangChain LLM adapters for the MCP server."""

from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from mcp_server.core.model import Model, GPTModel


class LangChainModel(Model):
    """LangChain-based Model implementation."""

    def __init__(self, llm: Any):
        """Initialize with a LangChain LLM."""
        self.llm = llm

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the LangChain LLM."""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages, **kwargs)
        return response.content

    async def generate_with_context(
        self, prompt: str, context: Dict[str, Any], **kwargs
    ) -> str:
        """Generate a response with additional context."""
        # Check if context contains a conversation history
        messages = []

        # Add system message if present
        if "system_prompt" in context:
            messages.append(SystemMessage(content=context["system_prompt"]))

        # Add conversation history if present
        if "conversation_history" in context:
            for msg in context["conversation_history"]:
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

        # Add the current prompt
        messages.append(HumanMessage(content=prompt))

        # Generate response
        response = await self.llm.ainvoke(messages, **kwargs)
        return response.content


def create_openai_model(
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs
) -> LangChainModel:
    """Create a LangChain model using OpenAI."""
    llm = ChatOpenAI(
        model_name=model_name, temperature=temperature, max_tokens=max_tokens, **kwargs
    )
    return LangChainModel(llm)
