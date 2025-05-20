"""Model component of the MCP pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Model(ABC):
    """Base Model class for the MCP pattern."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass

    @abstractmethod
    async def generate_with_context(
        self, prompt: str, context: Dict[str, Any], **kwargs
    ) -> str:
        """Generate a response with additional context."""
        pass


class GPTModel(Model):
    """GPT implementation of the Model component."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize the GPT model."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the GPT model."""
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage

        # Create params by merging instance defaults with method kwargs
        params = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update(self.additional_params)
        params.update(kwargs)

        # Initialize the LLM
        llm = ChatOpenAI(**params)

        # Generate response
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    async def generate_with_context(
        self, prompt: str, context: Dict[str, Any], **kwargs
    ) -> str:
        """Generate a response with additional context."""
        # Format the prompt with context
        formatted_prompt = self._format_with_context(prompt, context)

        # Use the standard generate method
        return await self.generate(formatted_prompt, **kwargs)

    def _format_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Format a prompt with context information."""
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        return f"Context:\n{context_str}\n\nPrompt: {prompt}"
