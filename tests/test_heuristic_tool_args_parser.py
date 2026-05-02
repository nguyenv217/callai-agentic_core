import pytest
from agentic_core.utils import HeuristicFailedToParse, heuristic_json_parse

def test_happy_path_valid_json():
    """Test that perfectly valid JSON parses immediately."""
    raw = '{"name": "agent", "iterations": 5}'
    result = heuristic_json_parse(raw)
    assert result == {"name": "agent", "iterations": 5}

def test_markdown_wrapper():
    """Test that Markdown code blocks are stripped out."""
    raw = '''```json
{
  "command": "search",
  "query": "quantum computing"
}
```'''
    result = heuristic_json_parse(raw)
    assert result == {"command": "search", "query": "quantum computing"}

def test_conversational_filler():
    """Test that conversational filler surrounding the JSON is ignored."""
    raw = '''Here is the tool call you requested:
{"action": "read_file", "path": "/tmp/test.txt"}
Let me know if you need anything else!'''
    result = heuristic_json_parse(raw)
    assert result == {"action": "read_file", "path": "/tmp/test.txt"}

def test_trailing_commas():
    """Test that trailing commas in objects and arrays are fixed."""
    # Object trailing comma
    raw_obj = '{"a": 1, "b": 2,}'
    assert heuristic_json_parse(raw_obj) == {"a": 1, "b": 2}

    # Array trailing comma
    raw_arr = '{"items": [1, 2, 3, ]}'
    assert heuristic_json_parse(raw_arr) == {"items": [1, 2, 3]}

def test_unescaped_newlines():
    """Test that literal newlines inside strings are properly escaped."""
    raw = '''{
  "text": "This is line 1.
This is line 2."
}'''
    result = heuristic_json_parse(raw)
    # The parser should convert the literal newline to an escaped \n
    assert result == {"text": "This is line 1.\nThis is line 2."}

def test_python_syntax_fallback():
    """Test that Python-style dictionaries (single quotes, True/False) are parsed."""
    # Note the single quotes and capital 'T' for True
    raw = "{'is_active': True, 'name': 'Ollama', 'value': None}"
    result = heuristic_json_parse(raw)
    assert result == {"is_active": True, "name": "Ollama", "value": None}

def test_unsalvageable_garbage():
    """Test that completely broken syntax raises a ValueError."""
    # Missing closing brackets and quotes
    raw = '{"action": "broken_tool", "data": [1, 2, 3'
    
    with pytest.raises(HeuristicFailedToParse, match="Could not parse arguments into JSON"):
        heuristic_json_parse(raw)

def test_empty_string():
    """Test that an empty string returns an empty dictionary."""
    assert heuristic_json_parse("") == {}
    assert heuristic_json_parse("   \n   ") == {}