
import asyncio
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("TestServer")

@mcp.tool()
def echo(text: str) -> str:
    """Echoes the input text."""
    return f"Echo: {text}"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run()
