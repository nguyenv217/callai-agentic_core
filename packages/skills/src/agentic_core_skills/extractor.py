import re
import uuid
import logging
from pathlib import Path
from agentic_core.llm_providers.base import ILLMClient

logger = logging.getLogger(__name__)

class SkillExtractor:
    """
    Synthesizes execution traces into reusable SKILL.md rules using an LLM.
    """
    def __init__(self, llm_client: ILLMClient, output_dir: Path | str):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_skill(self, execution_trace: str) -> bool:
        prompt = (
            "You are an expert meta-cognitive agent. Review the following execution trace of an agent "
            "attempting a complex task. The trace contains errors, retries, and trial-and-error tool usage, "
            "but ultimately succeeded.\n"
            "Extract the optimal, error-free path into a reusable SKILL.md file for future agents to follow.\n"
            "Identify the exact tool sequences that worked and explicitly warn against the mistakes made.\n\n"
            "Format your response EXACTLY as follows (including the markdown code block):\n"
            "```markdown\n"
            "---\n"
            "name: \"<Short, descriptive skill name>\"\n"
            "description: \"<Brief description of what this skill achieves>\"\n"
            "triggers:\n"
            "  - \"<trigger_keyword_1>\"\n"
            "  - \"<trigger_keyword_2>\"\n"
            "---\n"
            "<Detailed instructions, optimal tool sequence, and pitfalls to avoid>\n"
            "```\n\n"
            f"EXECUTION TRACE:\n{execution_trace}"
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text = ""
            async for response in self.llm.ask(messages=messages, stream=False):
                if response.text:
                    response_text += response.text

            match = re.search(r'```markdown\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if not match:
                logger.warning("Failed to extract markdown block from LLM response.")
                return False

            skill_content = match.group(1).strip()
            
            # Extract name to create a sensible directory
            name_match = re.search(r'name:\s*"?([^"]+)"?', skill_content)
            dir_name = "extracted_skill_" + str(uuid.uuid4())[:8]
            if name_match:
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name_match.group(1).lower())
                dir_name = f"{safe_name}_{str(uuid.uuid4())[:4]}"

            skill_path = self.output_dir / dir_name
            skill_path.mkdir(parents=True, exist_ok=True)
            
            with open(skill_path / "SKILL.md", "w", encoding="utf-8") as f:
                f.write(skill_content)
                
            logger.info(f"Successfully extracted new skill to {skill_path}")
            return True
            
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            return False
