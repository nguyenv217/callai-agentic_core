from abc import ABC, abstractmethod
from typing import Literal, TypedDict, Mapping, Any

class ToolSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: Mapping[str, Any]

class ToolSchema(TypedDict):
    type: Literal['function']
    funtion: ToolSchemaFunction


class BaseTool(ABC):
    """Interface for all executable tools."""
    def __init__(self):
        self._name = None
        self._schema = None
        
    @property
    def name(self) -> str:
        """The function name expected by the LLM."""
        return self._name
    
    @abstractmethod
    def execute(self, args: dict, context: dict) -> str:
        """Executes the tool logic and returns a string result."""
        pass
     
    @property
    def schema(self) -> ToolSchema:
        """The JSON schema associated with this tool"""
        return self._schema
    
    @staticmethod
    def _is_allowed_path(file_path: str, base_dir: str = ".") -> bool:
        """Return True only if *file_path* resolves inside *base_dir*."""
        if not file_path:
            return False

        from pathlib import Path
        # Reject absolute paths and null bytes
        if Path(file_path).is_absolute() or "\0" in file_path:
            return False

        base = Path(base_dir).resolve()  # canonical base
        try:
            target = (base / file_path).resolve()  # follow symlinks, collapse '..'
        except FileNotFoundError:
            # If you allow creating new files, use resolve(strict=False) instead
            return False

        # Python 3.9+: Path.is_relative_to
        return target.is_relative_to(base)
