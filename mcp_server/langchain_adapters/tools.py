"""LangChain tools implementation with improved error handling and default tools."""

from typing import Any, Dict, List, Optional, Callable
import logging

from langchain.agents import Tool
from langchain.tools import BaseTool

# Set up logger
logger = logging.getLogger(__name__)


class BasicTool:
    """Basic tool class for fallback when LangChain tools fail."""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_text: str) -> str:
        """Run the tool with input text."""
        try:
            return self.func(input_text)
        except Exception as e:
            return f"Error running tool: {str(e)}"


def create_simple_tool(name: str, func: Callable, description: str) -> Tool:
    """Create a simple LangChain tool with error handling."""
    try:
        return Tool(name=name, func=func, description=description)
    except Exception as e:
        logger.warning(f"Error creating tool {name}: {str(e)}")
        # Return a basic tool as fallback
        return BasicTool(name, description, func)


def create_calculator_tool() -> Any:
    """Create a calculator tool with error handling."""

    def calculator(query: str) -> str:
        """Calculate a mathematical expression."""
        try:
            return str(eval(query))
        except Exception as e:
            return f"Error calculating: {str(e)}"

    try:
        return Tool(
            name="calculator",
            description="Useful for performing mathematical calculations. Input should be a mathematical expression.",
            func=calculator,
        )
    except Exception as e:
        logger.warning(f"Error creating calculator tool: {str(e)}")
        return BasicTool(
            "calculator",
            "Useful for performing mathematical calculations. Input should be a mathematical expression.",
            calculator,
        )


def create_web_search_tool() -> Any:
    """Create a web search tool with error handling."""
    try:
        from langchain_community.utilities import GoogleSearchAPIWrapper

        search = GoogleSearchAPIWrapper()

        return Tool(
            name="web_search",
            description="Search the web for information. Useful for finding current or factual information.",
            func=search.run,
        )
    except Exception as e:
        logger.warning(f"Error creating web search tool: {str(e)}")

        def fallback_search(query: str) -> str:
            return (
                "Web search is not available. Please configure the Google Search API."
            )

        return BasicTool(
            "web_search",
            "Search the web for information. Useful for finding current or factual information.",
            fallback_search,
        )


def create_wikipedia_tool() -> Any:
    """Create a Wikipedia tool with error handling."""
    try:
        from langchain.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        return Tool(
            name="wikipedia",
            description="Search Wikipedia for information. Useful for finding facts and definitions.",
            func=wikipedia.run,
        )
    except Exception as e:
        logger.warning(f"Error creating Wikipedia tool: {str(e)}")

        def fallback_wikipedia(query: str) -> str:
            return "Wikipedia search is not available. Please install the required packages."

        return BasicTool(
            "wikipedia",
            "Search Wikipedia for information. Useful for finding facts and definitions.",
            fallback_wikipedia,
        )


def create_datetime_tool() -> Any:
    """Create a datetime tool."""
    from datetime import datetime

    def get_datetime(query: str = "") -> str:
        """Get the current date and time."""
        now = datetime.now()

        if "date" in query.lower():
            return now.strftime("Today's date is %Y-%m-%d")
        elif "time" in query.lower():
            return now.strftime("The current time is %H:%M:%S")
        else:
            return now.strftime("Current date and time: %Y-%m-%d %H:%M:%S")

    return BasicTool(
        "datetime",
        "Get the current date and time. You can ask for just the date or just the time.",
        get_datetime,
    )


def create_tools_toolkit() -> List[Any]:
    """Create a toolkit of common tools with guaranteed fallbacks."""
    tools = []

    # Always add calculator tool (pure Python implementation)
    tools.append(create_calculator_tool())

    # Always add datetime tool (pure Python implementation)
    tools.append(create_datetime_tool())

    # Try to add Wikipedia tool
    try:
        tools.append(create_wikipedia_tool())
    except Exception:
        logger.warning("Failed to create Wikipedia tool")

    # Try to add web search tool
    try:
        tools.append(create_web_search_tool())
    except Exception:
        logger.warning("Failed to create web search tool")

    # If no tools were added (very unlikely), add a dummy tool
    if not tools:

        def echo(input_text: str) -> str:
            return f"Echo: {input_text}"

        tools.append(BasicTool("echo", "Repeats back what you say to it.", echo))

    return tools
