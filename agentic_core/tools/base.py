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
    
