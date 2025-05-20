"""Context component of the MCP pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


class Context(ABC):
    """Base Context class for the MCP pattern."""

    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """Add an item to the context."""
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from the context."""
        pass

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove an item from the context."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from the context."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary."""
        pass


class MemoryContext(Context):
    """In-memory implementation of the Context component."""

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """Initialize the memory context."""
        self._data = initial_data or {}

    def add(self, key: str, value: Any) -> None:
        """Add an item to the context."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from the context."""
        return self._data.get(key, default)

    def remove(self, key: str) -> None:
        """Remove an item from the context."""
        if key in self._data:
            del self._data[key]

    def clear(self) -> None:
        """Clear all items from the context."""
        self._data.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary."""
        return self._data.copy()

    def update(self, data: Dict[str, Any]) -> None:
        """Update the context with a dictionary."""
        self._data.update(data)

    def keys(self) -> Set[str]:
        """Get all keys in the context."""
        return set(self._data.keys())
