# Agentic Core - Autonomous Skills Evolution

This package provides passive monitoring and autonomous synthesis of agent skills. It observes execution traces and, when an agent successfully recovers from a highly error-prone sequence, condenses that trial-and-error experience into a reusable, declarative `SKILL.md` file without human intervention.

## Architecture

The module adheres to the Decorator pattern, wrapping existing `AgentEventHandler` instances to intercept the engine's telemetry without mutating the core execution loop.

- **`AutoSkillObserver`**: An `AgentEventHandler` decorator that intercepts tool and error events to calculate struggle density.
- **`SkillExtractor`**: An LLM-backed synthesizer that formats the raw execution trace into an optimal, declarative markdown skill, stripping away the mistakes while highlighting the correct tool sequence.

## Integration

To integrate autonomous skill evolution into your application, initialize the extractor and wrap your primary event handler with the observer.

```python
import asyncio
from pathlib import Path
from agentic_core.engines import AgentRunner
from agentic_core.handlers import PrintHandler
from agentic_core_skills import AutoSkillObserver, SkillExtractor

async def main():
    # ... initialize llm, tools, memory, and runner ...

    # 1. Setup the backend synthesizer
    extractor = SkillExtractor(
        llm_client=runner.llm, 
        output_dir=Path("./.callai/skills")
    )

    # 2. Wrap your standard event handler
    # `error_threshold` defines how many transient errors must occur before 
    # the agent's successful recovery is deemed "skill-worthy"
    skill_aware_handler = AutoSkillObserver(
        extractor=extractor,
        error_threshold=2,
        base_handler=PrintHandler()
    )

    # 3. Execute the turn
    await runner.run_turn(
        user_input="Execute a highly complex, error-prone workflow", 
        handler=skill_aware_handler
    )

asyncio.run(main())
```

The extraction runs as a detached asynchronous background task (`asyncio.create_task`) upon `on_turn_complete`, ensuring the primary execution thread is never blocked during the synthesis phase.
