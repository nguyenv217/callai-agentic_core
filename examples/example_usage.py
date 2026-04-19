"""
ULTRA-SIMPLE EXAMPLE

Before: You needed to understand protocols, classes, generators...
After: Just call chat() with your API key!
"""
import asyncio
from agentic_core.agents import chat, create_openai_agent  # That's it!

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
if not load_dotenv(dotenv_path=env_path):
    raise RuntimeError("No .env file found. Please create one in the project root directory and try again")

import os

from pathlib import Path
os.environ['MCP_SERVER_MEMORY_FILE'] = str((Path(__file__).parent.parent / "memory.jsonl").resolve())
os.environ['MCP_CONFIG_PATH'] = str((Path(__file__).parent / "mcp_config.json").resolve())

async def main():
    # ============================================================
    # Example: OpenAI-compatible (just pass your key and OpenAI-compatible endpoint)
    # ============================================================
    print("=" * 50)
    print("Simple OpenAI Agent")
    print("=" * 50)
    
    runner = create_openai_agent(
        api_key=os.getenv("FIREWORK_API_KEY", ""),                          # let's suppose you use a provider like FIREWORK
        model="fireworks/minimax-m2p5",                                     # refer to the provider-specific model slug 
        base_url="https://api.fireworks.ai/inference/v1",                   # open-ai compatible endpoint
        mcp_config_path=os.getenv("MCP_CONFIG_PATH", "memory.jsonl")        # path to your MCP configuration file
    )

    from agentic_core.engine import RunnerConfig
    
    system_prompt = f"""You are a helpful assistant. Keep your answer short and concise. 
You have access to MCP servers to invoke tools when appropriate. 
Among these servers, you can use the 'memory' server to access past conversations. 
Only use these tools when necessary to conserve resources.
"""
    
    # RunnerConfig provides easy way to config agentic loop
    config = RunnerConfig(
        system_prompt=system_prompt,
        max_iterations=20,
        # mcp_preload_tools=[]
        )

    try:
        while True:
            result = await chat(
                message=input(">> "),
                runner=runner,
                verbose=True,                 # Set True to see what's happening
                config=config
            )
            print(f"\n📝 Response: {result}")

    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    asyncio.run(main())
