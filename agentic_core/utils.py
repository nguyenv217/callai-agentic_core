from typing import Any
import re
import json
import re
import ast
import logging
logger = logging.getLogger(__name__)


def clean_context_for_downstream(raw_response: str) -> str:
    """Removes <think> blocks to prevent context pollution."""
    cleaned = re.sub(r'<(think|thinking|thought)>.*?</(think|thinking|thought)>\n?', '', raw_response, flags=re.DOTALL)
    
    return cleaned.strip()

def convert_exception_to_message(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"

class HeuristicFailedToParse(Exception):
    """Raised by heuristic_json_parse when it fails to parse"""
    pass

def heuristic_json_parse(raw_text: str) -> dict[str, Any]:
    """
    Attempts to parse a string into JSON, applying heuristics to fix common LLM formatting errors.
    1. Strip markdown and conversational filler, e.g. ```json blocks or try to find valid json brackets
    2. `json.loads` normally
    3. if failed, apply heuristics to fix common LLM formatting errors
    """
    text = raw_text.strip()
    if not text:
        return {}

    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # Isolate the outermost brackets
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # Move to heuristics

    # Trailing commas (e.g., {"a": 1, } -> {"a": 1 })
    repaired_text = re.sub(r',\s*([\]}])', r'\1', text)
    
    try:
        return json.loads(repaired_text)
    except json.JSONDecodeError:
        pass

    # Unescaped newlines inside strings (breaks standard JSON). Replaced with escaped literal '\n'
    repaired_text = repaired_text.replace('\n', '\\n')
    
    # Reversing the previous escape for actual structural newlines (hacky)
    repaired_text = re.sub(r'\\n\s*([\{\}\[\]\,\:])', r'\n\1', repaired_text)
    repaired_text = re.sub(r'([\{\}\[\]\,\:])\s*\\n', r'\1\n', repaired_text)

    try:
        return json.loads(repaired_text)
    except json.JSONDecodeError:
        pass

    # Final fallback: local LLMs (like Llama 3 or Mistral) often output Python dicts instead of strict JSON 
    # (e.g., using single quotes 'key': 'value' or booleans like True/False).
    try:
        # ast.literal_eval is safe; it only parses literal structures, not executable code.
        parsed_ast = ast.literal_eval(text)
        if isinstance(parsed_ast, (dict, list)):
            return parsed_ast
    except (ValueError, SyntaxError):
        pass

    # If we reach here, it's completely unsalvageable. 
    logger.warning("Failed to heuristically repair LLM JSON output.")
    raise HeuristicFailedToParse(f"Could not parse arguments into JSON. Raw text: {raw_text[:100]}...")