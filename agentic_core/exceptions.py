class AgenticError(Exception):
    """Base exception for agentic_core"""
    pass

class MCPConnectionError(AgenticError):
    """Raised when connection to an MCP server fails"""
    pass

class ContextLimitExceededError(AgenticError):
    """Raised when the conversation context exceeds the LLM limit"""
    pass

class ProviderAuthenticationError(AgenticError):
    """Raised when the provider authentication fails"""
    pass

class ProviderRateLimitError(AgenticError):
    """Raised when the provider rate limit is exceeded"""
    pass

class ProviderTimeoutError(AgenticError):
    """Raised when the provider request times out"""
    pass

class IterationLimitReachedError(Exception):
    """Raised when the agent exceeds the maximum number"""
    pass

class NodeValidationError(Exception):
    """Raised when a node in the DAG is not properly configured"""
    pass

class NodeExecutionError(Exception):
    """Raised when a node fails to execute"""
    def __init__(self, node_id: str, message: str, original_exception: Exception | None = None):
        self.node_id = node_id
        self.original_exception = original_exception
        super().__init__(f"Node {node_id} failed: {message}")
