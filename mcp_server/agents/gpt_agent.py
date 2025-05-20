"""GPT-based agent implementation."""

from typing import Any, Dict, List, Optional

from mcp_server.agents.base import BaseAgent
from mcp_server.core.context import Context, MemoryContext
from mcp_server.core.model import GPTModel
from mcp_server.core.protocol import ChatProtocol


class GPTAgent(BaseAgent):
    """GPT-specific implementation of an Agent."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.6,
        system_prompt: str = "You are a helpful assistant.",
        context: Optional[Context] = None,
    ):
        """Initialize the GPT agent."""
        # Create the model
        model = GPTModel(model_name=model_name, temperature=temperature)

        # Create the protocol
        protocol = ChatProtocol(model)

        # Create the context if not provided
        if context is None:
            context = MemoryContext(
                {
                    "conversation_history": [],
                    "system_prompt": system_prompt,
                    "model_config": {
                        "model_name": model_name,
                        "temperature": temperature,
                    },
                }
            )

        # Initialize the base agent
        super().__init__(model, protocol, context, system_prompt)

    async def run_with_tools(self, input_data: str, tools: List[Dict[str, Any]]) -> str:
        """Run the agent with tools."""
        # Add tools to the context
        self.context.add("tools", tools)

        # Update the system prompt to include tool instructions
        system_prompt = self.context.get("system_prompt")
        tool_descriptions = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in tools]
        )

        tool_system_prompt = f"""{system_prompt}
        
You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
{{
  "tool": "tool_name",
  "tool_input": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}

Wait for the tool response before providing your final answer.
"""
        self.context.add("system_prompt", tool_system_prompt)

        # Run the agent
        response = await super().run(input_data)

        # Restore the original system prompt
        self.context.add("system_prompt", system_prompt)

        return response
