"""Utility helper functions for the MCP server."""

import json
import os
import uuid
from typing import Any, Dict, List, Optional


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def format_json_response(data: Any) -> str:
    """Format data as a JSON string."""
    return json.dumps(data, indent=2)


def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Parse JSON with error handling."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown text."""
    import re

    pattern = r"```(?:\w+)?\n([\s\S]+?)\n```"
    matches = re.findall(pattern, text)
    return matches


def execute_python_safely(code: str) -> Dict[str, Any]:
    """Execute Python code in a safe sandbox (simplified version)."""
    # In a real implementation, this would use a proper sandbox
    # This is a simplified version for demonstration purposes
    import ast

    # Check for unsafe operations
    try:
        parsed = ast.parse(code)
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    if name.name in ["os", "sys", "subprocess", "eval", "exec"]:
                        return {"error": f"Unsafe import: {name.name}", "result": None}
    except SyntaxError as e:
        return {"error": f"Syntax error: {str(e)}", "result": None}

    # Execute in a temporary namespace
    namespace = {}
    try:
        exec(code, {"__builtins__": {}}, namespace)
        return {"error": None, "result": namespace.get("result", None)}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}", "result": None}


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
