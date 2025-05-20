# MCP Server Template

A streamlined template for building AI agents using the Model-Context-Protocol pattern with GPT models and LangChain.

## Overview

This project provides a foundation for building AI applications using the Model-Context-Protocol (MCP) design pattern. The MCP pattern separates the core components of an AI system:

- **Model**: Handles interactions with language models like GPT
- **Context**: Manages state and memory for the system
- **Protocol**: Defines how the system processes inputs and generates outputs

By separating these concerns, the template provides a flexible architecture that can be adapted to various AI applications.

## Features

- Modular architecture based on the MCP pattern
- GPT integration via OpenAI's API
- LangChain adapters for quick integration with LangChain components
- FastAPI-based REST API
- Support for conversation history and context management
- Tool integration for enabling agents with external capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sanlamamba/mcp_template.git
cd mcp_template
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

5. Edit the `.env` file with your API keys and configuration.

## Usage

### Starting the Server

```bash
python app.py
```

The server will start at http://localhost:8000 by default.

### API Endpoints

- `POST /api/chat`: Chat with an AI agent
- `DELETE /api/conversations/{conversation_id}`: Delete a conversation
- `GET /api/models`: List available models
- `GET /health`: Health check endpoint

### Example Request

```python
import requests
import json

url = "http://localhost:8000/api/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key-here"  # If configured
}
data = {
    "prompt": "Tell me about artificial intelligence",
    "model": "gpt-4o",
    "temperature": 0.7,
    "system_prompt": "You are a helpful AI assistant.",
    "use_tools": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## Extending the Template

### Adding a New Model

1. Create a new class in `mcp_server/core/model.py` that inherits from `Model`
2. Implement the required methods: `generate` and `generate_with_context`

### Adding a New Tool

1. Add a new function in `mcp_server/langchain_adapters/tools.py`
2. Update the `create_tools_toolkit` function to include your new tool

### Adding Custom Protocols

1. Create a new class in `mcp_server/core/protocol.py` that inherits from `Protocol`
2. Implement the required `process` method

## Project Structure

```
mcp-server/
├── .env.example              # Environment variables template
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── app.py                    # Main application entry point
├── mcp_server/               # Main package
    ├── __init__.py           # Package initialization
    ├── config.py             # Configuration management
    ├── core/                 # Core MCP implementation
    │   ├── __init__.py
    │   ├── model.py          # Model component
    │   ├── context.py        # Context component
    │   └── protocol.py       # Protocol component
    ├── agents/               # Agent implementations
    │   ├── __init__.py
    │   ├── base.py           # Base agent class
    │   └── gpt_agent.py      # GPT-specific agent
    ├── langchain_adapters/   # LangChain integration
    │   ├── __init__.py
    │   ├── llms.py           # LLM adapters
    │   ├── chains.py         # Chain implementations
    │   ├── memory.py         # Memory implementations
    │   └── tools.py          # Tool implementations
    ├── api/                  # API server
    │   ├── __init__.py
    │   ├── server.py         # FastAPI server
    │   └── routes.py         # API endpoints
    └── utils/                # Utility functions
        ├── __init__.py
        └── helpers.py        # Helper functions
```

## License

MI