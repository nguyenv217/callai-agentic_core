import uuid
import logging
from pathlib import Path
from agentic_core.llm_providers.base import ILLMClient
from agentic_core.utils import heuristic_json_parse

logger = logging.getLogger(__name__)

class SkillExtractor:
    """
    Synthesizes conversation history into reusable SKILL.md rules using structured tool calling.
    """
    def __init__(self, llm_client: ILLMClient, output_dir: Path | str):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_skill(self, messages: list[dict]) -> bool:
        # Appending directly to the existing sequence guarantees a prefix cache hit on supported providers
        prompt = (
            "Meta-Cognitive Reflection: You have just successfully completed a task that required some trial-and-error. "
            "Review the conversation history above. Extract the optimal, error-free path into a reusable skill "
            "for future agents. Identify the exact tool sequences that worked and explicitly note the mistakes/pitfalls to avoid. "
            "You MUST use the `save_skill` tool to output your structural synthesis."
        )

        extraction_messages = list(messages) + [{"role": "user", "content": prompt}]

        schema = {
            "type": "function",
            "function": {
                "name": "save_skill",
                "description": "Saves a synthesized skill definition.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short, descriptive skill name (e.g. 'React Developer', 'Postgres Querying')"},
                        "description": {"type": "string", "description": "Brief description of what this skill achieves."},
                        "triggers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that should trigger this skill."
                        },
                        "instructions": {"type": "string", "description": "Detailed markdown instructions, optimal tool sequence, and pitfalls to avoid."}
                    },
                    "required": ["name", "description", "triggers", "instructions"]
                }
            }
        }

        try:
            tool_calls = []
            async for response in self.llm.ask(messages=extraction_messages, tools=[schema], stream=False):
                if response.tool_calls:
                    tool_calls.extend(response.tool_calls)
                    
            if not tool_calls:
                logger.warning("Agent failed to call `save_skill` during extraction.")
                return False

            target_call = next((tc for tc in tool_calls if tc["function"]["name"] == "save_skill"), None)
            if not target_call:
                return False
                
            args_str = target_call["function"].get("arguments", "{}")
            try:
                args = heuristic_json_parse(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                logger.error("Failed to parse save_skill arguments.")
                return False

            name = args.get("name", "Unknown Skill")
            desc = args.get("description", "")
            triggers = args.get("triggers", [])
            instructions = args.get("instructions", "")

            import re
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
            dir_name = f"{safe_name}_{str(uuid.uuid4())[:4]}"

            skill_content = (
                f"---\n"
                f"name: \"{name}\"\n"
                f"description: \"{desc}\"\n"
                f"triggers:\n"
            )
            for t in triggers:
                skill_content += f"  - \"{t}\"\n"
            skill_content += f"---\n\n{instructions}\n"

            skill_path = self.output_dir / dir_name
            skill_path.mkdir(parents=True, exist_ok=True)
            
            with open(skill_path / "SKILL.md", "w", encoding="utf-8") as f:
                f.write(skill_content)
                
            logger.info(f"Successfully extracted new skill to {skill_path}")
            return True
            
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            return False
