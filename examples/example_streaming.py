"""Example usage of `AgentStreamRunner` to stream agent events in real-time."""
from agentic_core.observers import SilentObserver
from dotenv import load_dotenv

from agentic_core.config import RunnerConfig
env_path = ".env"
if not load_dotenv(dotenv_path=env_path):
    raise RuntimeError("No .env file found. Please create one in the project root directory and try again")

import asyncio
import os

from agentic_core.engines.stream_engine import AgentStreamRunner
from agentic_core.llm_providers.openai import OpenAILLM
from agentic_core.tools import BaseTool, ToolManager
from agentic_core.memory import MemoryManager
from agentic_core.interfaces import StreamEventType

# Define a simple tool for testing
import json
class CalculatorTool(BaseTool):
    """A simple tool for testing the dispatch engine."""
    def __init__(self):
        super().__init__()
        self._name = "add_numbers"
        self._schema = {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Adds two numbers together",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        if isinstance(args, str):
            args = json.loads(args)
        result = args.get('a', 0) + args.get('b', 0)
        return json.dumps({"result": result})

async def main():
    # ============================================================
    # Example: OpenAI-compatible Streaming example (just pass your key and OpenAI-compatible endpoint that supports streaming)
    # ============================================================

    # 1. Setup tools
    async with ToolManager() as tools:
        # Note: In a real scenario, you would add your own tools here.
        # For this example, we assume some tools are already configured 
        # or will be discovered/loaded by the ToolManager.

        tools.register_tool(CalculatorTool())

        # 2. Setup LLM and Memory
        api_key = os.getenv("FIREWORK_API_KEY", "")
        model="fireworks/minimax-m2p5"
        base_url="https://api.fireworks.ai/inference/v1"

        llm = OpenAILLM(api_key=api_key, base_url=base_url, model=model)
        memory = MemoryManager()

        # 3. Initialize the streaming runner
        runner = AgentStreamRunner(
            llm_client=llm, 
            tools=tools, 
            memory=memory, 
            config=RunnerConfig(tools=[CalculatorTool().schema]),
            observer=SilentObserver())

        user_input = "What is the weather in Tokyo?"
        print(f"User: {user_input}\n")
        print("Agent: ", end="", flush=True)

        started_reason = False
        stopped_reason = False
        # 4. Stream the response
        async for event in await runner.stream(user_input):
            if event.type == StreamEventType.TEXT:
                if not stopped_reason:
                    print("=== [End of Reasoning] ===")
                    stopped_reason = True
                print(event.content, end="", flush=True)
            
            elif event.type == StreamEventType.REASONING:
                if not started_reason:
                    print(f"\n=== [Reasoning] ===:\n{event.content}", end="", flush=True)
                    started_reason=True
                else:
                    print(event.content, end="", flush=True)
            
            elif event.type == StreamEventType.TOOL_CALL:
                tool_name = event.content['function']['name']
                print(f"\n[Tool Call]: {tool_name}", end="", flush=True)
            
            elif event.type == StreamEventType.TOOL_RESULT:
                print(f"\n[Tool Result]: {event.content['result']}", end="", flush=True)
            
            elif event.type == StreamEventType.ERROR:
                print(f"\n[Error]: {event.content}", end="", flush=True)
            
            elif event.type == StreamEventType.FINAL_RESPONSE:
                print(f"\n\n[Complete] Final text: {event.content.text}")

if __name__ == "__main__":
    asyncio.run(main())
