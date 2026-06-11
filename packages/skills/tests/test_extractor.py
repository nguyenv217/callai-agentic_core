import pytest
import tempfile
from pathlib import Path
from agentic_core_skills.extractor import SkillExtractor
from agentic_core.llm_providers.base import ILLMClient, LLMResponse

class MockLLMClient(ILLMClient):
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def ask(self, messages, tools=None, stream=False, **kwargs):
        yield LLMResponse(text=self.response_text)

@pytest.mark.asyncio
async def test_extract_skill_success():
    mock_response = """
    Here is the skill you requested:
    ```markdown
    ---
    name: "Test Skill"
    description: "A test skill"
    triggers:
      - "test"
    ---
    Do this, not that.
    ```
    """
    llm = MockLLMClient(mock_response)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = SkillExtractor(llm_client=llm, output_dir=tmpdir)
        result = await extractor.extract_skill("Mock execution trace")
        
        assert result is True
        
        out_dir = Path(tmpdir)
        created_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
        assert len(created_dirs) == 1
        
        skill_dir = created_dirs[0]
        assert "test_skill_" in skill_dir.name
        
        skill_file = skill_dir / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "name: \"Test Skill\"" in content
        assert "Do this, not that." in content

@pytest.mark.asyncio
async def test_extract_skill_no_markdown():
    mock_response = "I couldn't find a skill to extract. I failed."
    llm = MockLLMClient(mock_response)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = SkillExtractor(llm_client=llm, output_dir=tmpdir)
        result = await extractor.extract_skill("Mock execution trace")
        
        assert result is False
