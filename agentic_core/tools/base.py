from abc import ABC, abstractmethod
from ..interfaces import ToolSchema

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
    async def execute(self, args: dict, context: dict) -> str:
        """Executes the tool logic and returns a string result."""
        pass
     
    @property
    def schema(self) -> ToolSchema:
        """The JSON schema associated with this tool"""
        return self._schema
    
