from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from ...config import ConfigurationError
from ..base import BaseTool

if TYPE_CHECKING:
    from ..manager import ToolManager

logger = logging.getLogger(__name__)

class ListMCPTools(BaseTool):
    """Returns a lightweight catalog of available external MCP tools."""
    
    def __init__(self, tool_manager: ToolManager, preview_limit: int = 3):
        self._tool_manager = tool_manager
        self.preview_limit = preview_limit
    
    @property
    def name(self) -> str:
        return "list_mcp_catalog"
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_mcp_catalog",
                "description": "Returns a catalog of available MCP tools grouped by server. Use this to discover external tools. Pass a specific server_name to see its full list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_name": {
                            "type": "string",
                            "description": "The specific MCP server domain (e.g., 'github', 'playwright') to expand and list all of its tools. Don't supply this argument to list all servers."
                        }
                    },
                    "required": []
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        server_filter = args.get("server_name")
        registry = getattr(self._tool_manager, '_mcp_standby_registry', {})
        
        if not registry:
            return "No MCP tools available in standby registry."
            
        # Group tools by their source server
        servers = {}
        for tool_name, adapter in registry.items():
            server = getattr(adapter, 'server_name', 'unknown')
            if server not in servers:
                servers[server] = []
            servers[server].append(adapter)
            
        lines = []
        
        if server_filter:
            # The agent requested a specific server's full catalog
            if server_filter not in servers:
                return f"Error: Server '{server_filter}' not found. Available servers: {list(servers.keys())}"
            
            lines.append(f"--- Available tools for server '{server_filter}' ({len(servers[server_filter])} total) ---")
            for adapter in servers[server_filter]:
                desc = adapter.schema['function'].get('description', 'No description')
                lines.append(f"- {adapter.name}: {desc}")
        else:
            # Default view: Overview with truncation
            lines.append("--- Available MCP Servers and Tools Overview ---")
            for server, adapters in servers.items():
                lines.append(f"\n[{server}] ({len(adapters)} tools total):")
                
                # Show up to 3 tools as a preview
                preview_limit = self.preview_limit
                for adapter in adapters[:preview_limit]:
                    # Grab description and truncate it to a single line for neatness
                    raw_desc = adapter.schema['function'].get('description', 'No description')
                    short_desc = raw_desc.split('\n')[0][:100]
                    if len(raw_desc) > 100: 
                        short_desc += "..."
                    lines.append(f"  - {adapter.name}: {short_desc}")
                
                if len(adapters) > preview_limit:
                    lines.append(f"  ... and {len(adapters) - preview_limit} more. -> Call list_mcp_catalog with server_name='{server}' to view all.")
                    
        lines.append("\nTo load tools into active context, call 'load_mcp_tool' providing a list of exact tool names in the 'tool_names' array.")
        return "\n".join(lines)

class LoadMCPTool(BaseTool):
    """Moves specified tool schemas from standby registry into active execution context."""
    
    def __init__(self, tool_manager: ToolManager):
        self._tool_manager = tool_manager
    
    @property
    def name(self) -> str:
        return "load_mcp_tool"
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "load_mcp_tool",
                "description": "Loads one or more MCP tools from the catalog into the active toolset, making them available for use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_names": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of exact names of the tools to load (from the catalog)."
                        }
                    },
                    "required": ["tool_names"]
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        tool_names = args.get("tool_names")
        
        # Fallback in case the LLM passes a single string or uses the old argument name
        if not tool_names and "tool_name" in args:
            tool_names = [args["tool_name"]]
        elif isinstance(tool_names, str):
            tool_names = [tool_names]
            
        if not tool_names or not isinstance(tool_names, list):
            return "Error: 'tool_names' (array of strings) is required."
        
        registry = getattr(self._tool_manager, '_mcp_standby_registry', {})
        
        results = []
        loaded = []
        
        for tool_name in tool_names:
            if tool_name not in registry:
                results.append(f"Error: Tool '{tool_name}' not found in catalog.")
                continue
            
            adapter = registry[tool_name]
            self._tool_manager.register_tool(adapter, load_mcp=True)
            
            loaded.append(tool_name)
            
        if loaded:
            results.append(f"Success: Loaded {len(loaded)} tool(s): {', '.join(loaded)}")
            
        return "\n".join(results)


class MCPToolAdapter(BaseTool):
    """
    A dynamic subclass of BaseTool that acts as a proxy to MCP server tools.
    Its execute method forwards calls via JSON-RPC to the respective MCP server.
    """
    
    def __init__(
        self, 
        mcp_tool_def: dict[str, Any], 
        session: Any, 
        server_name: str,
        timeout: float = 30.0,
        clean_schema: bool = True
    ):
        """
        Initialize the MCP tool adapter.
        
        Args:
            mcp_tool_def: The tool definition from MCP server (contains name, description, inputSchema)
            session: The MCP ClientSession connected to the server
            server_name: Name identifier for the MCP server (used for namespacing)
            timeout: Timeout in seconds for tool execution
        """
        super().__init__()
        self._session = session
        self.server_name = server_name
        self._timeout = timeout
        
        # Prefix the tool name to prevent collisions across servers
        # e.g., "sqlite_query" instead of just "query"
        self._actual_name = mcp_tool_def.get('name', 'unnamed')
        self._name = f"{server_name}_{self._actual_name}"
        
        if clean_schema:
            cleansed_schema = {k: v for k, v in mcp_tool_def.get('inputSchema', {}).items() if k in ['type', 'required', 'properties']}
        else:
            cleansed_schema = mcp_tool_def.get('inputSchema', {})

        # Map MCP schema to OpenAI function calling schema format
        self._schema = {
            "type": "function",
            "function": {
                "name": self._name,
                "description": mcp_tool_def.get('description', ''),
                "parameters": cleansed_schema
            }
        }
    
    @property
    def name(self) -> str:
        """The function name expected by the LLM."""
        return self._name
    
    @property
    def schema(self) -> dict:
        """The JSON schema associated with this tool."""
        return self._schema
    
    async def execute(self, args: dict, context: dict) -> str:
        """
        Execute the tool by calling the MCP server.
        
        Args:
            args: The arguments to pass to the tool
            context: The execution context (controller, callbacks, etc.)
            
        Returns:
            str: The result from the MCP server formatted as a string
        """
        import anyio
        actual_tool_name = self._actual_name

        try:
            # Apply timeout to prevent indefinite hangs
            logger.info(f"Executing tool {actual_tool_name} in mcp {self.server_name}")
            result = await asyncio.wait_for(
                self._session.call_tool(actual_tool_name, arguments=args),
                timeout=self._timeout
            )
            
            # Format MCP ToolResult array into string
            # MCP returns content as a list of Content objects (text, image, etc.)
            formatted_results = []
            for content in result.content:
                if hasattr(content, 'text'):
                    formatted_results.append(content.text)
                elif isinstance(content, dict):
                    formatted_results.append(json.dumps(content))
                else:
                    formatted_results.append(str(content))
            
            return "\n".join(formatted_results)
            
        except anyio.ClosedResourceError:
            logger.error(f"MCP server {self.server_name} disconnected unexpectedly.")
            return f"Error: The external server process for '{self.server_name}' crashed or disconnected. Please check the server logs and restart the tool."
        except asyncio.TimeoutError:
            return f"Error: Timeout after {self._timeout}s"
        except Exception as e:
            logger.exception(f"Exception at MCP tool {self._name}")
            return f"Error: {str(e)}"
