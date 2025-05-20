"""LangChain chains implementation for the MCP server."""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain_core.language_models import BaseChatModel

from mcp_server.core.context import Context
from mcp_server.core.protocol import Protocol


class LangChainProtocol(Protocol):
    """LangChain-based Protocol implementation."""

    def __init__(self, chain: Any):
        """Initialize with a LangChain chain."""
        self.chain = chain

    async def process(self, input_data: Any, context: Context) -> Any:
        """Process input data using the LangChain chain."""
        # Check if the chain expects additional variables from context
        # This is a simplistic approach; might need customization based on chain type
        variables = {"input": input_data}

        # Add context variables to chain inputs if they exist
        context_dict = context.to_dict()
        for key, value in context_dict.items():
            if (
                key != "conversation_history"
            ):  # Skip conversation history as it's handled differently
                variables[key] = value

        # Run the chain
        response = await self.chain.ainvoke(variables)

        # Extract output based on chain type
        output = response.get("output", response.get("response", response))

        # Update context with response if needed
        if hasattr(self.chain, "memory") and self.chain.memory:
            # If chain has memory, update context with memory variables
            memory_variables = self.chain.memory.load_memory_variables({})
            for key, value in memory_variables.items():
                context.add(key, value)

        return output


def create_conversation_chain(
    llm: BaseChatModel, memory: Optional[Any] = None
) -> LangChainProtocol:
    """Create a LangChain conversation chain protocol."""
    if memory is None:
        memory = ConversationBufferMemory()

    chain = ConversationChain(llm=llm, memory=memory, verbose=True)

    return LangChainProtocol(chain)


def create_agent_chain(
    llm: BaseChatModel,
    tools: List[Any],
    memory: Optional[Any] = None,
    agent_type: AgentType = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
) -> LangChainProtocol:
    """Create a LangChain agent chain protocol."""
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    agent = initialize_agent(
        tools=tools, llm=llm, agent=agent_type, memory=memory, verbose=True
    )

    return LangChainProtocol(agent)
